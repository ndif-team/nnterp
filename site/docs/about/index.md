# About nnterp

nnterp gives every transformer family the same module names and the same standard values, on top
of [nnsight](https://nnsight.net). It is developed by the [NDIF](https://ndif.us/) team and
maintained at [github.com/ndif-team/nnterp](https://github.com/ndif-team/nnterp).

nnterp 1.x was written by Clément Dumas and described in
[nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](https://arxiv.org/abs/2511.14465).
nnterp 2 is a rebuild on nnsight 0.8: the names are nnsight aliases, the values are properties on
the envoys, and each family is one module that says how its own names map onto the standard ones.

## Citation

```bibtex
@misc{dumas2025nnterp,
      title={nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers},
      author={Cl{\'e}ment Dumas},
      year={2025},
      eprint={2511.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.14465},
}
```

## Found a bug?

Reduce it to a minimal example first. If the same code written with nnsight's `TransformersModel`
on the native module names also fails, the bug is in nnsight: open it on the
[nnsight tracker](https://github.com/ndif-team/nnsight/issues). Otherwise open an issue on the
[nnterp tracker](https://github.com/ndif-team/nnterp/issues).
