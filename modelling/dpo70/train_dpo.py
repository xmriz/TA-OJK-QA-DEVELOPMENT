#!/usr/bin/env python
# train_dpo.py
# Launch with: accelerate launch train_dpo.py

import os
import random
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from tqdm.auto import tqdm

# — GPU configuration —
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# — Fixed hyperparameters —
SEED                 = 42
CACHE_FOLDER         = "../model_cache"
MODEL_NAMES = {
    "Meta-Llama-3.1-8B": os.path.join("../sft/sft_output_Meta-Llama-3.1-8B", "final_checkpoint"),
    "Aya-23-8B":         os.path.join("../sft/sft_output_Aya-23-8B", "final_checkpoint"),
    "SeaLLMs-v3-7B":     os.path.join("../sft/sft_output_SeaLLMs-v3-7B", "final_checkpoint"),
    "SEA-LION-v3-8B":    os.path.join("../sft/sft_output_SEA-LION-v3-8B", "final_checkpoint"),
    "Sahabat-AI-8B":     os.path.join("../sft/sft_output_Sahabat-AI-8B", "final_checkpoint"),
}
PREF_DATA_TEMPLATE   = "preference_{key}_clean.jsonl"
OUTPUT_DIR_TEMPLATE  = "dpo_output_{key}"

# — DPO hyperparameters —
BETA                       = 0.05
LEARNING_RATE              = 1e-5
LR_SCHEDULER_TYPE          = "cosine"
WARMUP_STEPS               = 50
WEIGHT_DECAY               = 0.05
OPTIMIZER                  = "paged_adamw_32bit"
NUM_EPOCHS                 = 3
LOGGING_STEPS              = 10
PER_DEVICE_TRAIN_BATCH     = 2
PER_DEVICE_EVAL_BATCH      = 1
GRADIENT_ACCUMULATION_STEPS= 8
GRADIENT_CHECKPOINTING     = True
GRAD_CHECKPOINTING_REENTRANT = False
MAX_PROMPT_LENGTH          = 1792
MAX_LENGTH                 = 2048
REMOVE_UNUSED_COLUMNS      = False
REPORT_TO                  = "tensorboard"
FP16                       = False
BF16                       = True
IGNORE_BIAS_BUFFERS        = False

def set_global_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def prepare_datasets(pref_path: str, seed: int = SEED):
    """
    1) Load raw JSONL preference dataset.
    2) Split 90% train / 10% valid.
    """
    ds = load_dataset("json", data_files=pref_path, split="train")
    parts = ds.train_test_split(test_size=0.1, seed=seed)
    return parts["train"], parts["test"]

def main():
    set_global_seed()
    state = PartialState()
    device_idx = state.process_index

    for model_key, sft_ckpt in MODEL_NAMES.items():
        print(f"\n=== DPO fine-tuning {model_key} (rank {device_idx}) ===")

        # 1) Tokenizer (processing_class will handle tokenization)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(CACHE_FOLDER, model_key),
            local_files_only=True
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # 2) 4-bit quantization config
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 3) Load SFT-trained base model (4-bit)
        model = AutoModelForCausalLM.from_pretrained(
            sft_ckpt,
            quantization_config=quant_cfg,
            device_map={"": device_idx},
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model.config.use_cache = False

        # (Optional) ignore boolean buffers in DDP
        if IGNORE_BIAS_BUFFERS and hasattr(model, "_ddp_params_and_buffers_to_ignore"):
            model._ddp_params_and_buffers_to_ignore = [
                n for n, buf in model.named_buffers() if buf.dtype == torch.bool
            ]

        # 4) Inline LoRA config for DPO
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )

        # 5) Prepare raw preference train/valid splits
        pref_path = PREF_DATA_TEMPLATE.format(key=model_key)
        train_ds, valid_ds = prepare_datasets(pref_path)

        # 6) DPOConfig with per-epoch eval/save
        output_dir = OUTPUT_DIR_TEMPLATE.format(key=model_key)
        os.makedirs(output_dir, exist_ok=True)
        dpo_args = DPOConfig(
            output_dir                   = output_dir,
            overwrite_output_dir         = True,

            # — Training & Eval —
            do_train                     = True,
            do_eval                      = True,
            num_train_epochs             = NUM_EPOCHS,
            per_device_train_batch_size  = PER_DEVICE_TRAIN_BATCH,
            per_device_eval_batch_size   = PER_DEVICE_EVAL_BATCH,
            gradient_accumulation_steps  = GRADIENT_ACCUMULATION_STEPS,

            # — Optimizer & Scheduler —
            learning_rate                = LEARNING_RATE,
            optim                        = OPTIMIZER,
            weight_decay                 = WEIGHT_DECAY,
            lr_scheduler_type            = LR_SCHEDULER_TYPE,
            warmup_steps                 = WARMUP_STEPS,

            # — Logging & Saving —
            logging_dir                  = os.path.join(output_dir, "logs"),
            logging_strategy             = "steps",
            logging_steps                = LOGGING_STEPS,
            eval_strategy                = "epoch",
            save_strategy                = "epoch",
            report_to                    = REPORT_TO,
            logging_first_step           = True,

            # — Precision & Performance —
            fp16                         = FP16,
            bf16                         = BF16,
            gradient_checkpointing       = GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs= {"use_reentrant": GRAD_CHECKPOINTING_REENTRANT},

            # — Sequence & Tokenisation —
            max_prompt_length            = MAX_PROMPT_LENGTH,
            max_length                   = MAX_LENGTH,

            # — Miscellaneous —
            remove_unused_columns        = REMOVE_UNUSED_COLUMNS,
            seed                         = SEED,
            run_name                     = f"dpo_{model_key}"
        )

        # 7) Initialize & run DPOTrainer
        trainer = DPOTrainer(
            model            = model,
            ref_model        = None,         # DPOTrainer will clone model under the hood
            args             = dpo_args,
            train_dataset    = train_ds,
            eval_dataset     = valid_ds,
            peft_config      = peft_cfg,
            processing_class = tokenizer     # will handle all tokenization inside trainer
        )

        print(f"Starting DPO training for {model_key}…")
        trainer.train()

        # 8) Save adapters & tokenizer
        trainer.save_model(output_dir)
        final_ckpt = os.path.join(output_dir, "final_checkpoint")
        trainer.model.save_pretrained(final_ckpt)

        print(f"✔️  DPO adapters for {model_key} saved under {output_dir}")
        del model, trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
