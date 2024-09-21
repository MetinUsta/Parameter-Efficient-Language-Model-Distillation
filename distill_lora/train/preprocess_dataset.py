from datasets import load_dataset
from transformers import AutoTokenizer

from distill_lora.utils.parse_configs import TrainConfig


def process_conversation(sample, student_tokenizer):
    conversation = sample["conversation"]
    messages = []

    for msg in conversation:
        if msg["from"] == "human":
            messages.append({"role": "user", "content": msg["value"]})
        elif msg["from"] == "gpt":
            messages.append({"role": "assistant", "content": msg["value"]})
        elif msg["from"] == "system":
            messages.append({"role": "system", "content": msg["value"]})

    text = student_tokenizer.apply_chat_template(messages, tokenize=False)

    return text


def tokenize_sample(sample, train_config, student_tokenizer):
    return student_tokenizer(
        sample["text"],
        truncation=True,
        max_length=train_config.cutoff_len,
        padding="max_length",
    )


def preprocess_dataset(train_config: TrainConfig):
    dataset = load_dataset(train_config.dataset, split="train")
    dataset = dataset.shuffle(train_config.random_seed)

    if train_config.num_samples is not None:
        dataset = dataset.select(range(train_config.num_samples))

    student_tokenizer = AutoTokenizer.from_pretrained(train_config.student_model_path)

    dataset_columns = dataset.column_names

    dataset = dataset.map(
        lambda x: process_conversation(x, student_tokenizer),
        remove_columns=dataset_columns,
    )

    dataset = dataset.map(
        lambda x: tokenize_sample(x, train_config, student_tokenizer),
        batched=True,
        num_proc=train_config.num_proc,
    )

    dataset = dataset.train_test_split(test_size=train_config.val_size)

    return dataset
