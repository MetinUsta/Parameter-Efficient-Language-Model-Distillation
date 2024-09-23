import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from distill_lora.utils.parse_configs import TrainConfig


def load_models(config: TrainConfig):
    if config.teacher_quantization_bit == 4:
        teacher_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif config.teacher_quantization_bit == 8:
        teacher_bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if config.student_quantization_bit == 4:
        student_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif config.student_quantization_bit == 8:
        student_bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if config.flash_attn == "fa2":
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_path,
        quantization_config=teacher_bnb_config,
        attn_implementation=attn_implementation,
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_path,
        quantization_config=student_bnb_config,
        attn_implementation=attn_implementation,
    )

    student_model = prepare_model_for_kbit_training(
        student_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    teacher_model = prepare_model_for_kbit_training(
        teacher_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return teacher_model, student_model


class LogitsTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        # labels = inputs.pop('labels', None)

        student_outputs = student_model(**inputs)
        teacher_model.eval()
        # with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(
            student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss
        )
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits_scaled = (
            student_logits / self.distill_config.distillation_temperature
        )
        teacher_logits_scaled = (
            teacher_logits / self.distill_config.distillation_temperature
        )

        loss_kd = (
            F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="batchmean",
            )
            * (self.distill_config.distillation_temperature**2)
            / self.distill_config.max_seq_length
        )

        return (
            self.distill_config.distillation_alpha * loss_kd
            + (1 - self.distill_config.distillation_alpha) * original_loss
        )


def train(config: TrainConfig, dataset):
    teacher_model, student_model = load_models(config)

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_path)

    sft_config = SFTConfig(**config.get_training_args())

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )

    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=sft_config,
        peft_config=peft_config,
    )

    trainer.teacher_model = teacher_model
    trainer.distill_config = config

    os.environ["WANDB_PROJECT"] = config.wandb_project
    accelerator = Accelerator()

    trainer = accelerator.prepare(trainer)

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model(config.output_dir)
