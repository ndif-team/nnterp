"""``model.internals``: every tensor nnterp can name on this model, in one place.

The accessors themselves (``model.layers_output``, ``model.attention_probabilities``, ...)
are attributes of the model, as they always were. This is the view over all of
them: which exist on this architecture and why not, and where each one sits in
the forward pass, which is what nnsight's execution order asks of you.
"""

from __future__ import annotations

from .rename_utils import Address, LayerAccessor, RenamingError


class Internals(dict[str, LayerAccessor]):
    """The registry of a model's accessors, by name, in the order the forward pass
    reaches them.

    Examples
    --------
    >>> list(model.internals)
    ['embeddings_input', 'embeddings_output', 'layers_input', 'attentions', ...]
    >>> model.internals["mlps_output"][3]            # the accessor, at layer 3
    >>> model.internals.status()["mlps_output"]      # None, or the reason it is unavailable
    >>> model.internals.status(layer=1)["mlps_activation"]   # a mixture-of-experts layer says so
    """

    def __init__(self, model, addresses: dict[str, Address]):
        super().__init__(
            (name, LayerAccessor(model, address, name=name))
            for name, address in sorted(addresses.items(), key=lambda item: item[1].order)
        )
        self.model = model

    def __getitem__(self, name: str) -> LayerAccessor:
        if name not in self:
            raise RenamingError(
                f"nnterp has no accessor named {name!r}. It has: {list(self)}. "
                "Add one with RenameConfig(addresses={name: Address(...)})."
            )
        return super().__getitem__(name)

    def status(self, layer: int | None = None) -> dict[str, str | None]:
        """For each accessor, ``None`` if it is available and otherwise the reason
        it is not, answered without running the model. With no ``layer``, the
        model-wide answer (a per-layer place is available if any layer has it);
        with one, the answer at that layer, which is what differs on a model whose
        layers are not all alike."""
        found = {}
        for name, accessor in self.items():
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
        and above for the final norm and head. Sorting by it is how several
        internals are read in one trace whatever order you name them in."""
        accessor = self[name]
        if accessor.per_layer:
            assert layer is not None, f"{name} is per layer: say which"
            return (layer, accessor.address.order)
        order = accessor.address.order
        return (-1, order) if order < 0 else (self.model.num_layers, order)
