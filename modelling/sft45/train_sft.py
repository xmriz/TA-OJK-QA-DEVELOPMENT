#!/usr/bin/env python
# train_sft.py
# Launch with: accelerate launch train_sft.py

import os
import random
import torch
from accelerate import PartialState, Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm

# — GPU configuration —
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# — Paths & constants —
SEED            = 42
DATA_PATH       = "../datasets4545/cqa_sft_prompt_completion.jsonl"
SEQ_LENGTH      = 2048
NUM_EPOCHS      = 3
BATCH_SIZE      = 1
EVAL_BATCH      = 1
ACCUM_STEPS     = 4
LOG_STEPS       = 50
LR              = 1e-4
LR_SCHED        = "cosine"
WEIGHT_DECAY    = 0.05
OPTIM           = "paged_adamw_32bit"
BF16            = True
GRAD_CHECKPOINT = False
GROUP_BY_LENGTH = False
REPORT_TO       = ["tensorboard"]

MODEL_NAMES = {
    "Meta-Llama-3.1-8B": "../model_cache/Meta-Llama-3.1-8B",
    "Aya-23-8B":         "../model_cache/Aya-23-8B",
    "SeaLLMs-v3-7B":     "../model_cache/SeaLLMs-v3-7B",
    "SEA-LION-v3-8B":    "../model_cache/SEA-LION-v3-8B",
    "Sahabat-AI-8B":     "../model_cache/Sahabat-AI-8B"
}


def prepare_datasets(path: str, seed: int = SEED):
    """
    Load JSONL dataset and split into train/validation.
    """
    raw = load_dataset("json", data_files=path, split="train")
    split = raw.train_test_split(test_size=0.1, seed=seed)
    return split["train"], split["test"]


def count_token_lengths(dataset, tokenizer):
    """
    Compute max lengths for prompts, completions, and their concatenation.
    """
    max_prompt = max_completion = max_both = 0
    for ex in tqdm(dataset, desc="Counting token lengths", total=len(dataset)):
        p = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        c = tokenizer(ex["completion"], add_special_tokens=False).input_ids
        both = tokenizer(ex["prompt"] + ex["completion"], add_special_tokens=False).input_ids
        max_prompt = max(max_prompt, len(p))
        max_completion = max(max_completion, len(c))
        max_both = max(max_both, len(both))
    return max_prompt, max_completion, max_both


def merge_prompt_and_completion(example):
    """
    Merge prompt and completion into a single field for SFTTrainer.
    """
    example["prompt_completion"] = example["prompt"] + example["completion"]
    return example


def main():
    # Set seeds for reproducibility
    set_seed(SEED)
    random.seed(SEED)

    device_idx = PartialState().process_index
    accelerator = Accelerator()

    # Load and prepare datasets
    train_raw, valid_raw = prepare_datasets(DATA_PATH)

    for model_key, model_path in MODEL_NAMES.items():
        print(f"\n=== Fine-tuning {model_key} ===")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="right", local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Inspect token lengths
        max_p, max_c, max_pc = count_token_lengths(train_raw, tokenizer)
        print(f"Max prompt tokens: {max_p}")
        print(f"Max completion tokens: {max_c}")
        print(f"Max prompt+completion tokens: {max_pc}")

        # Quantize model in 4-bit
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_cfg,
            device_map={"": device_idx},
            trust_remote_code=True,
            local_files_only=True
        )
        model.config.use_cache = False

        # LoRA PEFT config
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )

        # SFTTrainer config
        training_args = SFTConfig(
            # Training & Evaluation
            output_dir                 = f"sft_output_{model_key}",
            overwrite_output_dir       = True,
            do_train                   = True,
            do_eval                    = True,
            num_train_epochs           = NUM_EPOCHS,
            per_device_train_batch_size= BATCH_SIZE,
            per_device_eval_batch_size = EVAL_BATCH,
            gradient_accumulation_steps= ACCUM_STEPS,
            
            # Optimizer & Scheduler
            learning_rate              = LR,
            optim                      = OPTIM,
            weight_decay               = WEIGHT_DECAY,
            lr_scheduler_type          = LR_SCHED,

            # Logging
            logging_dir                = f"sft_output_{model_key}/logs",
            logging_strategy           = "steps",
            logging_steps              = LOG_STEPS,
            logging_first_step         = True,

            # Eval & Save per epoch
            eval_strategy              = "epoch",
            save_strategy              = "epoch",

            # Precision
            bf16                       = BF16,
            gradient_checkpointing     = GRAD_CHECKPOINT,

            # Sequence & Packing
            max_length                 = SEQ_LENGTH,
            packing                    = True,
            group_by_length            = GROUP_BY_LENGTH,

            # Data field
            dataset_text_field         = "prompt_completion",

            # Miscellaneous
            seed                       = SEED
        )

        # Prepare datasets for Trainer
        train_ds = train_raw.map(
            merge_prompt_and_completion,
            remove_columns=[c for c in train_raw.column_names if c not in ["prompt", "completion"]]
        )
        valid_ds = valid_raw.map(
            merge_prompt_and_completion,
            remove_columns=[c for c in valid_raw.column_names if c not in ["prompt", "completion"]]
        )

        # Initialize and train
        trainer = SFTTrainer(
            model           = model,
            train_dataset   = train_ds,
            eval_dataset    = valid_ds,
            args            = training_args,
            peft_config     = peft_cfg,
            processing_class= tokenizer
        )

        print(f"Starting training for {model_key}…")
        trainer.train()

        # Save final model
        trainer.save_model(training_args.output_dir)
        final_ckpt = os.path.join(training_args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(final_ckpt)

        print(f"✅ Completed fine-tuning for {model_key}")
        del model, trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
