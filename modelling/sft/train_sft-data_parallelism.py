#!/usr/bin/env python
# train_sft-data_parallelism.py
# Launch with: accelerate launch --config_file accelerate_config-data_parallelism.yaml train_sft-data_parallelism.py

import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


def main():
    # Environment Configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

    # Reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPUs: {torch.cuda.device_count()}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load tokenizer and base model with model parallelism
    model_id = "SeaLLMs/SeaLLMs-v3-7B"
    cache_dir = "../model_cache"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    model.config.use_cache = False

    # Apply PEFT LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare dataset
    raw = load_dataset(
        "json", data_files={"train": "../datasets/cqa_sft.jsonl"}
    )["train"]

    def make_prompt(ex):
        return {
            "prompt": (
                ex["context"].strip()
                + "\n\nPertanyaan: "
                + ex["question"].strip()
                + "\nJawaban: "
            ),
            "target": ex["answer"].strip(),
        }

    ds = raw.map(make_prompt, remove_columns=raw.column_names)

    def tokenize_and_mask(ex):
        full = ex["prompt"] + ex["target"] + tokenizer.eos_token
        tok = tokenizer(full, truncation=True, max_length=512)
        input_ids = tok["input_ids"]
        prompt_len = len(
            tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        )
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": tok["attention_mask"],
            "labels": labels,
        }

    train_dataset = ds.map(
        tokenize_and_mask,
        remove_columns=["prompt", "target"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Configure training arguments with checkpointing and packing
    training_args = SFTConfig(
        output_dir="sft_dp",
        per_device_train_batch_size=2,      # lower if needed
        gradient_accumulation_steps=16,    # accumulate to maintain effective bsz
        fp16=True,
        save_strategy="steps", save_steps=500, save_total_limit=3,
        logging_steps=50,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Resume from last checkpoint if available
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch.")
        trainer.train()

    # Save the final LoRA adapter
    trainer.save_model("sft_dp_final")


if __name__ == "__main__":
    main()
