from trl import ScriptArguments
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdvVerifScriptArguments:
    dataset_name: str
    eval_dataset_name: str
    task: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    config: Optional[str] = None
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
