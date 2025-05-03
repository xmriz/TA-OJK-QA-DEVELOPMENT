#!/usr/bin/env python
# train_sft-model_parallelism.py
# Launch with: accelerate launch --config_file accelerate_config-model_parallelism.yaml train_sft-model_parallelism.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def main():
    # ——— Reproducibility ———
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ——— Load tokenizer & base model ———
    model_id  = "SeaLLMs/SeaLLMs-v3-7B"
    cache_dir = "../model_cache"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tanpa device_map: biarkan Accelerate+DeepSpeed handle TP & PP
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # wajib untuk LoRA

    # ——— Pasang LoRA adapter ———
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

    # ——— Siapkan dataset ———
    raw = load_dataset("json", data_files={"train": "../datasets/cqa_sft.jsonl"})["train"]

    def make_prompt(ex):
        return {
            "prompt": (
                ex["context"].strip()
                + "\n\nPertanyaan: " + ex["question"].strip()
                + "\nJawaban: "
            ),
            "target": ex["answer"].strip(),
        }

    ds = raw.map(make_prompt, remove_columns=raw.column_names)

    def tokenize_and_mask(ex):
        full = ex["prompt"] + ex["target"] + tokenizer.eos_token
        tok  = tokenizer(full, truncation=True, max_length=512)
        prompt_len = len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"])
        labels = [-100] * prompt_len + tok["input_ids"][prompt_len:]
        return {
            "input_ids":      tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels":         labels,
        }

    train_dataset = ds.map(tokenize_and_mask, remove_columns=["prompt","target"])
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ——— Argumen training ———
    training_args = TrainingArguments(
        output_dir="sft_mp",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
    )

    # ——— Inisialisasi SFTTrainer & Mulai ———
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("sft_mp_final")

if __name__ == "__main__":
    main()
