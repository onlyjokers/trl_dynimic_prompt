import pytest
from trl.trainer.grpo_trainer import GRPOTrainer

class DummyModel:
    """Minimal stub model with training flag."""
    def __init__(self):
        self.training = True

class DummyAccelerator:
    """Minimal stub accelerator with device and process_index."""
    def __init__(self):
        self.device = "cpu"
        self.process_index = 0

@ pytest.fixture
def dummy_trainer():
    # Bypass __init__, stub necessary attributes
    trainer = object.__new__(GRPOTrainer)
    trainer.dynamic_prompts_fn = None
    trainer.last_prompts = None
    trainer.last_completions = None
    trainer.last_rewards = None
    trainer.model = DummyModel()
    trainer.accelerator = DummyAccelerator()
    return trainer

 def test_dynamic_prompts_called(dummy_trainer):
    calls = []
    # define a dynamic function that records its inputs then aborts
    def dyn_fn(prompts, completions, rewards, metas):
        calls.append((prompts, completions, rewards, metas))
        raise RuntimeError("dynamic_fn called")

    # set up last_* fields
    dummy_trainer.dynamic_prompts_fn = dyn_fn
    dummy_trainer.last_prompts = ["old_prompt"]
    dummy_trainer.last_completions = ["old_completion"]
    dummy_trainer.last_rewards = [0.5]

    # prepare a single input sample
    inputs = [{"prompt": "initial_prompt", "meta": 123}]
    # Calling _generate_and_score_completions should invoke dynamic_fn before generation
    with pytest.raises(RuntimeError, match="dynamic_fn called"):
        dummy_trainer._generate_and_score_completions(inputs)

    # Verify dynamic_fn received correct arguments
    assert calls == [(
        ["old_prompt"],
        ["old_completion"],
        [0.5],
        [{"meta": 123}]
    )]
