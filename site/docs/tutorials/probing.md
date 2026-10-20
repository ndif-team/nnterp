# Probing

A probe asks whether a property is linearly decodable from the residual stream at each depth: collect `layer_output` at one position of every prompt in a labelled set, fit a linear classifier per layer, and read accuracy against the layer. `get_token_activations` does the collection against the standard values, so the recipe below runs unchanged on every family. The probe itself is a few lines of torch.

## Collecting the activations

```python
import torch
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import get_token_activations

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")

positive = ["wonderful", "fantastic", "delightful", "excellent", "brilliant", "joyful", "superb", "lovely", "charming", "pleasant"]
negative = ["terrible", "awful", "dreadful", "disgusting", "horrible", "miserable", "dismal", "boring", "painful", "ugly"]
templates = ["The movie was {}, so afterwards we", "I found the book {} and told my",
             "That meal was {}, which is why the", "Their performance was {}, and the crowd"]
texts = [t.format(w) for w in positive + negative for t in templates]
labels = torch.tensor([1.0] * 40 + [0.0] * 40)

acts = get_token_activations(model, texts, idx=-1)
print(acts.shape)   # torch.Size([12, 80, 768])
```

One batched forward gives `[layers, prompts, hidden_size]` on the CPU, the residual stream `model.layers[i].layer_output` at the last position of every prompt. The last position of a batch needs left padding, which nnsight sets at load for a causal model. The helper raises `ValueError` before running if the tokenizer pads right. `layers=` picks some blocks, `get_activations=lambda m, i: m.layers[i].mlp.mlp_output` reads another site, and `collect_token_activations_batched(model, texts, batch_size)` returns the same tensor for a set too large for one batch ([`nnsight_utils`](../documentation/nnsight_utils.md)).

The adjective sits mid-sentence, several tokens before the position read, and the test words never appear in training. Split by word, not by prompt: with the words shared between the splits, the probe can read the adjective's identity instead of its sentiment.

## One probe per layer

```python
held_out = {"superb", "lovely", "charming", "pleasant", "dismal", "boring", "painful", "ugly"}
is_test = torch.tensor([any(w in t for w in held_out) for t in texts])
train, test = (~is_test).nonzero().squeeze(1), is_test.nonzero().squeeze(1)


def fit_probe(x, y, ridge=1e-2):
    """Ridge regression of the {-1, +1} label on [n, hidden_size]; returns predict(x) -> bool tensor."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    design = lambda z: torch.cat([(z - mean) / std, torch.ones(len(z), 1)], dim=1)
    z = design(x)
    penalty = torch.eye(z.shape[1])
    penalty[-1, -1] = 0                                                  # no penalty on the bias
    w = torch.linalg.solve(z.T @ z + ridge * len(z) * penalty, z.T @ (2 * y - 1))
    return lambda x_new: design(x_new) @ w > 0


for layer in range(model.num_layers):
    x = acts[layer].float()
    predict = fit_probe(x[train], labels[train])
    accuracy = (predict(x[test]).float() == labels[test]).float().mean()
    print(f"layer {layer:2d}  accuracy on held-out words {float(accuracy):.2f}")
# layer  0  accuracy on held-out words 0.78
# layer  1  accuracy on held-out words 0.94
# layer  2  accuracy on held-out words 0.97
# layer  3  accuracy on held-out words 1.00
# ...
# layer 11  accuracy on held-out words 1.00
```

The probe is closed-form ridge regression, standardized on the training rows only. Keep the ridge on and report it, because with `hidden_size` far above the number of examples an unregularized probe fits anything.

## Reading the curve

Sentiment is readable at the sentence's last token from layer 3 on, for words the probe never saw. The position read is the final token (`we`, `my`, `the`, `crowd`), the same in both classes, so whatever the probe finds there was carried from the adjective three to five tokens back. Block 0's attention already moves enough of it for 0.78, and blocks 1 to 3 complete it. A shuffled-label control bounds the memorization. Refit with `labels[torch.randperm(80)]` and the same probe scores 0.50, 0.44 and 0.34 at layers 0, 6 and 11.

A probe is correlational. To claim that the model uses the direction, add it to the stream with `model.steer` and check that the next-token distribution moves while a random direction of the same norm does not. [Steering](steering.md) has the factor band.

## On another family

Nothing in the recipe names a family. `layer_output` is the residual stream leaving block `i` as a tensor on every family, whatever the native block returns, and the helper reads nothing else. Change the checkpoint and the word lists. Accuracies are not comparable across models of different `hidden_size` unless the probe's capacity is held fixed, so compare the shape of the curves and report the layer-0 value with each.
