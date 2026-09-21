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
    ...     values = model.internals.read(3, "layers_output", "attentions_output")
    """

    def __init__(self, model, addresses: dict[str, Address]):
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

    def status(self) -> dict[str, str | None]:
        """For each accessor, ``None`` if it is available on this model and
        otherwise the reason it is not. Answered without running the model."""
        return {name: accessor.disabled_reason for name, accessor in self._accessors.items()}

    def read(self, layer: int, *names: str) -> dict[str, object]:
        """Read several internals of one layer inside a trace, whatever order
        they are named in. nnsight cannot reach back to a value the model has
        already passed; the table knows the forward's order, so you need not."""
        ordered = sorted(names, key=lambda name: self[name].address.order)
        return {name: self[name][layer] for name in ordered}
