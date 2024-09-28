from peft import PeftModel
from transformers import AutoModelForCausalLM

from distill_lora.utils.parse_configs import TrainConfig


def merge_adapter(config: TrainConfig):

    base_model = AutoModelForCausalLM.from_pretrained(config.student_model_path)
    peft_model_id = "C:\\Users\\pc\\Downloads\\qwen1_5B_adapter\\content\\adapters\\qwen2-0.5b_distill_qlora"
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    merged_model = model.merge_and_unload()

    merged_model.half()

    merged_model.save_pretrained(config.output_dir)