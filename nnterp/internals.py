"""``model.internals``: every tensor nnterp can name on this model, in one place.

The accessors themselves (``model.layers_output``, ``model.attention_probabilities``, ...)
are attributes of the model, as they always were. This is the view over all of
them: which exist on this architecture and why not, and a read of several that
gets nnsight's forward-order rule right for you.
"""

from __future__ import annotations

from .rename_utils import Address, LayerAccessor, RenamingError


class Internals:
    """The registry of a model's accessors, by name.

    Examples
    --------
    >>> model.internals.names
    ['layers_input', 'attentions', 'attentions_input', 'attention_probabilities', ...]
    >>> model.internals.status()["mlps_output"]      # None, or the reason it is unavailable
    >>> with model.trace(prompt):
    ...     values = model.internals.read("layers_output", "attentions_output", layer=3)
    >>> model.internals.status(layer=1)["mlps_activation"]   # a mixture-of-experts layer says so
    """

    def __init__(self, model, addresses: dict[str, Address]):
        self.model = model
        self._accessors = {
            name: LayerAccessor(model, address, name=name)
            for name, address in sorted(addresses.items(), key=lambda item: item[1].order)
        }

    @property
    def names(self) -> list[str]:
        """Every accessor, in the order the forward pass reaches them."""
        return list(self._accessors)

    def __contains__(self, name: str) -> bool:
        return name in self._accessors

    def __iter__(self):
        return iter(self._accessors)

    def __getitem__(self, name: str) -> LayerAccessor:
        if name not in self._accessors:
            raise RenamingError(
                f"nnterp has no accessor named {name!r}. It has: {self.names}. "
                "Add one with RenameConfig(addresses={name: Address(...)})."
            )
        return self._accessors[name]

    def status(self, layer: int | None = None) -> dict[str, str | None]:
        """For each accessor, ``None`` if it is available and otherwise the reason
        it is not, answered without running the model. With no ``layer``, the
        model-wide answer (a per-layer place is available if any layer has it);
        with one, the answer at that layer, which is what differs on a model whose
        layers are not all alike."""
        found = {}
        for name, accessor in self._accessors.items():
            if not accessor.per_layer:
                found[name] = accessor.unavailable_on(None)
            elif layer is not None:
                found[name] = accessor.unavailable_on(layer)
            else:
                reasons = [accessor.unavailable_on(i) for i in range(self.model.num_layers)]
                found[name] = None if any(reason is None for reason in reasons) else reasons[0]
        return found

    def rank(self, name: str, layer: int | None = None) -> tuple[int, int]:
        """Where a place is in the model's forward pass: a per-layer place is
        ranked by its layer then its place in the block; a whole-model place by
        its own order, which the table sets below every layer for the embeddings
        and above for the final norm and head."""
        accessor = self[name]
        if accessor.per_layer:
            assert layer is not None, f"{name} is per layer: say which"
            return (layer, accessor.address.order)
        order = accessor.address.order
        return (-1, order) if order < 0 else (self.model.num_layers, order)

    def read(self, *names: str, layer: int | None = None) -> dict[str, object]:
        """Read several internals inside a trace, whatever order they are named
        in — per-layer ones at ``layer``, whole-model ones as they are. nnsight
        cannot reach back to a value the model has already passed; the table
        knows the forward's order, so you need not."""
        if names and isinstance(names[0], int):  # the older spelling, read(layer, *names)
            layer, names = names[0], names[1:]
        ordered = sorted(names, key=lambda name: self.rank(name, layer if self[name].per_layer else None))
        return {name: self[name][layer if self[name].per_layer else None] for name in ordered}
