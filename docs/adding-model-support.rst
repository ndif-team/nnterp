Adding Support for Your Model
==============================

.. meta::
   :llm-description: Add custom model support using RenameConfig. Learn path-based renaming, multiple alternative names, and implementing attention probabilities with real GPT-J example.

``nnterp`` uses a standardized naming convention to provide a unified interface across transformer architectures. When your model doesn't follow the expected naming patterns, you can use ``RenameConfig`` to map your model's modules to the standardized names.

Understanding the Target Structure
----------------------------------

``nnterp`` expects models to follow this structure:

.. code-block:: text

   StandardizedTransformer
   ├── embed_tokens
   ├── layers[i]
   │   ├── self_attn
   │   └── mlp
   ├── ln_final
   └── lm_head

All models are automatically renamed to match this pattern using built-in mappings for common architectures.

In addition to these renamed modules, ``nnterp`` provides convenient accessors:

- ``embed_tokens``: Embedding module
- ``token_embeddings``: Token embeddings (equivalent to ``embed_tokens.output``)
- ``layers[i]``: Layer module at layer i
- ``layers_input[i]``, ``layers_output[i]``: Layer input/output at layer i
- ``attentions[i]``: Attention module at layer i
- ``attentions_input[i]``, ``attentions_output[i]``: Attention input/output at layer i
- ``mlps[i]``: MLP module at layer i
- ``mlps_input[i]``, ``mlps_output[i]``: MLP input/output at layer i
- ``attention_layers``, ``linear_attention_layers``: indices of the blocks with a softmax ``self_attn`` and of the blocks with a ``linear_attn`` mixer (Qwen3-Next / Qwen3.5 hybrids). A linear mixer is not renamed, and the attention accessors are only defined on ``attention_layers``

Basic RenameConfig Usage
------------------------

When automatic renaming fails, create a custom ``RenameConfig`` by specifying the names of modules in YOUR model that correspond to each standardized component:

.. code-block:: python

   from nnterp import StandardizedTransformer
   from nnterp.rename_utils import RenameConfig

   # Hypothetical model with custom naming
   rename_config = RenameConfig(
       model_name="custom_transformer",           # Name of your model's main module
       layers_name="custom_layers",               # Name of your model's layer list
       attn_name="custom_attention",              # Name of your model's attention modules
       mlp_name="custom_ffn",                     # Name of your model's MLP modules
       ln_final_name="custom_norm",               # Name of your model's final layer norm
       lm_head_name="custom_head"                 # Name of your model's language modeling head
   )

   model = StandardizedTransformer(
       "your-model-name",
       rename_config=rename_config
   )

Each parameter specifies what YOUR model calls the component that will be renamed to the standard name (e.g., ``layers_name="custom_layers"`` means your model has a module called "custom_layers" that will be accessible as "layers").

Path-Based Renaming
-------------------

For nested modules, use dot notation to specify the full path from the model root:

.. code-block:: python

   rename_config = RenameConfig(
       layers_name="custom_transformer.encoder_layers",
       ln_final_name="custom_transformer.final_norm"
   )

Multiple Alternative Names
--------------------------

Provide multiple options for the same component:

.. code-block:: python

   rename_config = RenameConfig(
       attn_name=["attention", "self_attention", "mha"],
       mlp_name=["ffn", "feed_forward", "mlp_block"]
   )

Modules That Add the Residual Internally
----------------------------------------

Some architectures (BLOOM, MPT, DBRX) add the residual stream to the sublayer
output *inside* the attention/MLP module, so the module output is a
residual-stream state rather than the additive contribution
(`issue #51 <https://github.com/ndif-team/nnterp/issues/51>`_). nnterp refuses to
load an unknown architecture whose attention/MLP forward takes a
``residual``-like argument. To add support, point the output accessors to the
last pre-residual submodule:

.. code-block:: python

   rename_config = RenameConfig(
       attn_output_source="self_attn.dense",     # BLOOM's attention output projection
       mlp_output_source="mlp.dense_4h_to_h",    # BLOOM's MLP output projection
   )

Paths are relative to a layer and use standardized (post-renaming) names. If the
module does not actually add the residual to its output, pass the default source
(``attn_output_source="self_attn"`` / ``mlp_output_source="mlp"``) to keep the
module output.

Naming a place yourself: ``addresses``
--------------------------------------

Those two keys are special cases of ``addresses``, which replaces a row of
nnterp's address table or adds one of your own. A row is an ``Address``: the
module (relative to a layer, or to the model for a whole-model place), which side
of it carries the tensor, the operations of its ``.source`` to descend through,
and where the tensor sits inside the value found there.

.. code-block:: python

   from nnterp.rename_utils import Address, IOType, Index, RenameConfig

   rename_config = RenameConfig(
       addresses={
           # the same thing attn_output_source says
           "attentions_output": Address("self_attn.dense", order=30),
           # an accessor of your own: it appears in model.internals and as
           # model.attentions_gate, and reads a module this family alone has
           "attentions_gate": Address("self_attn.gate_proj", order=25),
       }
   )

``order`` is the row's place in the block's forward pass, which is how
``model.internals.rank(name, layer)`` sorts several places into the order nnsight
requires them to be read in.

``io`` is spelled as nnsight spells it, on a module and on a call alike:
``IOType.OUTPUT`` is what is returned, ``IOType.INPUT`` the first positional
argument, and ``IOType.INPUTS`` the whole ``(args, kwargs)`` pair, which is how a
row names an argument that is not the first:

.. code-block:: python

   # the second positional argument of the attention interface call: the query
   Address("self_attn", IOType.INPUTS, op=("attention_interface_1",),
           select=Index(0, 1), order=15)

A row that reads an operation of a forward names it the way nnsight does, and
those names come from the transformers version you have installed. If one moves,
the accessor raises a ``RenamingError`` naming the row, the model class, the
transformers version and every operation that does exist at that level, so the
fix is to replace that one row. ``nnterp/tests/test_source_ops.py`` runs every
such row against every family nnterp pins, with the attention probabilities
enabled, and is what catches an upgrade that moves one.

``Address.select`` says where the tensor is inside the value at that place, for a
module that returns more than the tensor. It is ``None`` by default, meaning the
value untouched; ``FirstIfTuple()`` for a module that returns its output beside a
cache (what ``layers_output`` and ``attentions_output`` use); or an ``Index``,
spelled ``Index(0)`` / ``0`` / ``("hidden_states",)``, walked to read and rebuilt
around the new tensor to write. For a value neither describes, write a
``Selection`` of your own:

.. code-block:: python

   from nnterp.rename_utils import Selection

   class TheTensor(Selection):
       """Whichever element of the value is a tensor."""

       def get(self, value):
           return next(item for item in value if isinstance(item, torch.Tensor))

       def put(self, value, new):
           return tuple(new if isinstance(item, torch.Tensor) else item for item in value)


Real Example: GPT-J Support
----------------------------

Here's how GPT-J attention probabilities were added to nnterp:

First, examine the model architecture:

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   # GPT-J loads with basic renaming but attention probabilities fail
   model = StandardizedTransformer("yujiepan/gptj-tiny-random")
   # Warning: Attention probabilities test failed

Locate the attention probabilities in the forward pass:

.. code-block:: python

   # Find where attention weights are computed
   with model.scan("test"):
       model.attentions[0].source.self__attn_0.source.self_attn_dropout_0.output.shape
       # Shape: (batch, heads, seq_len, seq_len) - this is what we want

Create the attention probabilities function:

.. code-block:: python

   from nnterp.rename_utils import AttnProbFunction, RenameConfig

   class GPTJAttnProbFunction(AttnProbFunction):
       def get_attention_prob_source(self, attention_module, return_module_source=False):
           if return_module_source:
               return attention_module.source.self__attn_0.source
           else:
               return attention_module.source.self__attn_0.source.self_attn_dropout_0

   model = StandardizedTransformer(
       "yujiepan/gptj-tiny-random",
       enable_attention_probs=True,
       rename_config=RenameConfig(attn_prob_source=GPTJAttnProbFunction())
   )

Test the implementation:

.. code-block:: python

   with model.trace("Hello world"):
       attn_probs = model.attention_probabilities[0].save()
       # Verify shape: (batch, heads, seq_len, seq_len)
       # Verify last dimension sums to 1
       assert attn_probs.sum(dim=-1).allclose(torch.ones_like(attn_probs.sum(dim=-1)))

Attention Probabilities (Optional)
-----------------------------------

Only implement attention probabilities if you need them for your research. The process requires:

1. **Find the attention weights**: Use ``model.scan()`` to explore the forward pass
2. **Locate the hook point**: Find where attention probabilities are computed (usually after dropout)
3. **Create AttnProbFunction**: Implement the hook location
4. **Test thoroughly**: Verify shape and normalization

Key considerations:

- Use ``scan()`` first, fall back to ``trace()`` if needed
- Hook after dropout but before multiplication/masking when possible
- Avoid hooks inside conditional statements
- Test with dummy inputs to verify tensor shapes

Troubleshooting
---------------

Common issues and solutions:

**"Could not find layers module"**
   Set ``layers_name`` in ``RenameConfig``

**"Could not find ln_final module"**
   Set ``ln_final_name`` in ``RenameConfig``

**"Attention probabilities test failed"**
   Either disable attention probabilities or implement ``AttnProbFunction``

**Shape mismatches**
   nnterp automatically detects and unwraps tuple outputs from modules during initialization

Testing Your Configuration
--------------------------

``nnterp`` automatically validates your configuration:

.. code-block:: python

   # This will run automatic tests
   model = StandardizedTransformer("your-model", rename_config=config)
   
   # Manual validation
   with model.trace("test"):
       # Check layer I/O shapes
       layer_out = model.layers_output[0]
       assert layer_out.shape == (batch_size, seq_len, hidden_size)
       
       # Check attention probabilities if enabled
       if model.attn_probs_available:
           attn_probs = model.attention_probabilities[0]
           assert attn_probs.shape == (batch_size, num_heads, seq_len, seq_len)

The tests verify:

- Module naming correctness
- Tensor shapes at each layer
- Attention probabilities normalization (if enabled)
- I/O compatibility with nnterp's accessors

Once your model loads successfully, all ``nnterp`` features become available with the standard interface.