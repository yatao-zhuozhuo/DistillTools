import os
import json
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def main():
    # Parameters are hardcoded for simplicity
    model_name_or_path = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
    data_path = "/remote-home1/yzyang/day4-exercise/synthetic_data_trained.json"
    output_dir = "day4-exercise/qwen_lora_model"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    # 1. Load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={'': torch.cuda.current_device()}
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and process data
    dataset = load_dataset('json', data_files=data_path, split="train")

    def format_and_tokenize(example):
        # Format the chat template
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
        # Tokenize the text
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=1024,
        )
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=list(dataset.features))
    
    # 3. LoRA setup
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Trainer setup
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 5. Train and save
    trainer.train()
    
    print("Saving LoRA adapter...")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main() 