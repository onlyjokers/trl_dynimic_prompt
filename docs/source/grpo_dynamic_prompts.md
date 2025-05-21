# GRPO Dynamic Prompt Adjustment

This document describes how to test and verify the dynamic prompt adjustment feature in `GRPOConfig` and `GRPOTrainer`.

## Feature Overview

- **dynamic_prompts_fn**: A user-provided function in `GRPOConfig` that is called after each generation to update prompts for the next iteration.
- Signature:
  ```python
  dynamic_prompts_fn(
      prompts: List[str],
      completions: List[str],
      rewards: List[float],
      metas: List[dict]
  ) -> List[str]
  ```

- **GRPOTrainer**:
  - Records `last_prompts`, `last_completions`, `last_rewards` after each generation.
  - On next generation call, invokes `dynamic_prompts_fn` (in training mode) with the recorded values and updates `inputs` accordingly.

## Test Plan

1. **Basic invocation**: Ensure `dynamic_prompts_fn` is called when `last_*` fields are set.
2. **Correct arguments**: Verify the function receives the exact lists of previous prompts, completions, rewards, and meta-dicts.
3. **Prompt update**: Simulate a function that modifies the prompt list, and confirm `GRPOTrainer` uses the returned prompts in the next generation.

## Example Test (Pytest)

```python
import pytest
from trl.trainer.grpo_trainer import GRPOTrainer

class DummyModel:
    def __init__(self):
        self.training = True

class DummyAccelerator:
    def __init__(self):
        self.device = "cpu"
        self.process_index = 0

@pytest.fixture
def dummy_trainer():
    trainer = object.__new__(GRPOTrainer)
    # initialize stub attributes
    trainer.dynamic_prompts_fn = None
    trainer.last_prompts = None
    trainer.last_completions = None
    trainer.last_rewards = None
    trainer.model = DummyModel()
    trainer.accelerator = DummyAccelerator()
    return trainer

# Test dynamic function invocation
def test_dynamic_prompts_called(dummy_trainer):
    calls = []
    def dyn_fn(prompts, completions, rewards, metas):
        calls.append((prompts, completions, rewards, metas))
        raise RuntimeError("dynamic_fn called")

    dummy_trainer.dynamic_prompts_fn = dyn_fn
    dummy_trainer.last_prompts = ["q1"]
    dummy_trainer.last_completions = ["a1"]
    dummy_trainer.last_rewards = [1.0]

    inputs = [{"prompt": "initial", "info": "meta"}]
    with pytest.raises(RuntimeError):
        dummy_trainer._generate_and_score_completions(inputs)

    assert calls == [(
        ["q1"], ["a1"], [1.0], [{"info": "meta"}]
    )]
```

## Running Tests

```bash
pytest tests/test_grpo_dynamic_prompts.py
```

If the tests pass, the dynamic prompt adjustment feature works as expected.
