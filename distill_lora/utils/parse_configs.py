from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings

# yaml file content
# teacher_model_path: ""
# student_model_path: ""

# quantization_bit: 4

# lora_target: all
# lora_rank: 64
# lora_alpha: 32

# dataset: distill_qlora
# max_seq_length: 2048
# num_samples: 1000

# output_dir: adapters/gemma2_9B_distill_qlora
# logging_steps: 1
# save_steps: 500

# per_device_train_batch_size: 1
# gradient_accumulation_steps: 2
# learning_rate: 2.0e-5
# num_train_epochs: 1
# lr_scheduler_type: cosine
# warmup_ratio: 100
# bf16: true

# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500


class TrainConfig(BaseSettings):
    teacher_model_path: str = ""
    student_model_path: str = ""

    teacher_quantization_bit: int = 4
    student_quantization_bit: int = 4

    wandb_project: str = "lora"

    lora_target: str = "all"
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    dataset: str = "distill_qlora"
    max_seq_length: int = 2048
    num_samples: int = None
    num_proc: int = 8

    output_dir: str = "adapters/qlora_distill"
    logging_steps: int = 1
    save_steps: int = 500

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2.0e-5
    num_train_epochs: int = 1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    fp16: bool = False
    flash_attn: str = "fa2"
    resume_from_checkpoint: bool = False

    val_size: float = 0.1
    per_device_eval_batch_size: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 500

    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5

    dataset_text_field: str = "text"

    random_seed: int = 1337

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TrainConfig":
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
        return cls(**yaml_data)

    def update_from_yaml(self, yaml_file: str) -> None:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def get_training_args(self):
        return {
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "dataset_text_field": self.dataset_text_field,
            "max_seq_length": self.max_seq_length,
        }