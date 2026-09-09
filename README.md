# FinBehavior

FinBehavior is a compact research project for learning a fixed-size user
representation from a financial profile and a time-ordered event history. It uses
a small Transformer and a self-supervised masked-value objective: hide one event
value, encode the user's remaining context, and predict the hidden value.

The repository currently uses deterministic synthetic data. It is useful for
testing the complete modelling pipeline, but it is not yet evidence that the
learned user vector is useful for real customers or a particular business task.
See [Current limitations](#current-limitations).

## What the project does

The end-to-end flow is:

```text
hidden synthetic behavior
        ↓
profile + chronological events
        ↓
categorical tokens + numerical buckets + time features
        ↓
key/value-aware embeddings + positional encoding
        ↓
Transformer encoder
        ↓
masked-value prediction and a 32-dimensional user vector
```

The synthetic generator first samples six hidden tendencies between 0 and 1:
income, spending, travel, investing, app activity, and communication engagement.
Those values affect the generated profile and the frequency or contents of four
event sources:

- transactions;
- app interactions;
- trading activity;
- customer communications.

The hidden tendencies are generator metadata and are **not** passed to the model.
The model only receives the visible profile and event history.

### Concrete example

A generated transaction can look like this:

```text
source: transaction
type: card_payment
direction: out
amount: 42.50
currency: EUR
merchant_category: restaurant
merchant_region: ES
```

Tokenization turns field names into source-qualified keys such as
`transaction.amount`. Categorical values receive vocabulary IDs. Numerical amounts
are assigned to quantile buckets fitted on training users only, for example:

```text
42.50 → transaction.amount.bucket_2
```

Calendar features encode cyclic hour/day information, and a log-scaled feature
describes elapsed time. When training masks `merchant_category`, the rest of the
profile and sequence remain visible. The prediction head is told which field is
being predicted and only allows values valid for that field.

## Architecture

1. A field pair is encoded from `concat(key_embedding, value_embedding)` by a
   small MLP. Keeping the key and value bound avoids treating a swap such as
   `screen=opened, action=cards` as equivalent to the valid assignment.
2. Profile fields are pooled with a `[USR]` token. Event fields are pooled and
   combined with an `[EVT]` token and projected temporal features.
3. Sinusoidal positional encodings preserve the order of profile/event positions.
4. Two Transformer blocks process the complete sequence by default. The profile
   position after encoding is the 32-dimensional user representation.
5. For masked-value training, the encoded event position is passed to a linear
   prediction head. The masked field's key conditions the allowed output domain,
   preventing values from unrelated fields from appearing in the result.

Default model configuration:

| Setting | Value |
| --- | ---: |
| Embedding dimension | 32 |
| Transformer blocks | 2 |
| Feed-forward expansion | 4× |
| Numerical buckets | 10 |

## Dataset split and evaluation

The generalization experiment uses a deterministic split by **user**, not by
event or masked example:

| Partition | Fraction | Purpose |
| --- | ---: | --- |
| Train | 80% | Fit numerical buckets and model parameters |
| Validation | 10% | Select the best checkpoint |
| Test | 10% | Final evaluation after model selection |

This prevents one user's events from leaking across partitions. Validation uses a
fixed set of masks so epochs are comparable; training masks are resampled each
epoch. The held-out test set is evaluated only after loading the checkpoint chosen
by validation loss.

Reported metrics are:

- **loss** — mean cross-entropy of the true hidden token; lower is better;
- **top-1 accuracy** — the true token is the most probable allowed prediction;
- **top-5 accuracy** — the true token appears among up to five allowed
  predictions. A field with fewer than five valid values naturally returns fewer
  candidates.

These metrics measure masked-value reconstruction. They do not, by themselves,
prove that the user vector is useful for fraud detection, churn, recommendation,
risk, or segmentation.

## Setup

Python 3.11 or newer is required.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch installed from the default package index is sufficient for CPU use. For a
specific CUDA build, follow the PyTorch installation instructions for the target
machine before installing this package.

## Run the examples

Inspect one event's numerical, categorical, and temporal tokenization:

```bash
python scripts/demo_tokenization.py
```

Generate 20 users, fit numerical buckets on 15 of them, and inspect a held-out
user's raw and tokenized record:

```bash
python scripts/demo_user_pipeline.py
```

## Train and evaluate

```bash
python experiments/generalization_training.py
```

One invocation trains five epochs. A later invocation resumes from the latest
checkpoint, including the optimizer state. The experiment stores architecture-v3
artifacts separately from older incompatible checkpoints:

```text
checkpoints/generalization_v3_latest/
checkpoints/generalization_v3_best/
```

`model.pt` is the authoritative checkpoint and is atomically replaced only after
its temporary file and the human-readable `tokenizer.json` sidecar have been
written. The tokenizer state is embedded in `model.pt`, so a crash between file
replacements cannot create a mixed model/tokenizer restore. Version 3 also
preserves the prediction head's per-field allowed-value domains. Older version-2
checkpoints are deliberately not loaded because the key/value encoder and
positional architecture changed; retrain instead.

At the end of a run, the script reloads `generalization_v3_best` and prints final
metrics on the untouched test users. Do not tune model choices from those test
numbers; use validation results for that.

## Run inference

Train at least once, then run:

```bash
python experiments/masked_value_inference.py
```

The script reconstructs the deterministic held-out test partition, masks one
field, loads the best v3 checkpoint, and prints the true value plus the allowed
top predictions.

## Use the data API

Synthetic datasets can be serialized as JSON Lines and read back with validation:

```python
from datetime import datetime
from finbehavior.data.generators.dataset import generate_dataset
from finbehavior.data.serialization.jsonl import (
    read_dataset_jsonl,
    write_dataset_jsonl,
)

users = generate_dataset(
    number_of_users=100,
    start=datetime(2026, 1, 1),
    evaluation_point=datetime(2026, 2, 1),
    seed=42,
)
write_dataset_jsonl(users, "data/users.jsonl")
restored_users = read_dataset_jsonl("data/users.jsonl")
```

Numerical bucket boundaries must always be fitted on training records only. The
generalization experiment already enforces that rule.

## Development checks

Run the same checks as continuous integration:

```bash
python -m black --check .
python -m pytest
```

To apply formatting locally:

```bash
python -m black .
```

GitHub Actions runs formatting and the test suite on pushes and pull requests.

## Current limitations

- All training and evaluation data is synthetic and shares the generator's rules.
  Strong masked-value scores can therefore reflect recovery of those rules rather
  than general financial behavior.
- The usefulness and stability of the 32-dimensional user representation have not
  yet been validated with downstream probes, clustering criteria, or real labels.
- Attention cost grows quadratically with sequence length. The bundled training and
  inference experiments cap histories at the 256 most recent events, but there is
  no streaming encoder or long-sequence benchmark yet.
- JSONL ingestion currently targets the project's synthetic-user schema, not a
  versioned real-bank event contract.
- The project has no production serving layer, privacy controls, drift monitoring,
  calibration study, or model-card process.

Before production use, evaluate on a separately governed real dataset, define a
business-specific downstream target, compare against simple baselines, add history
windowing, and review privacy and operational requirements.
