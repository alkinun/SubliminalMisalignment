import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoints/LoRA adapter save directory")
parser.add_argument("--output_dir", type=str, required=True, help="Model save directory")
args = parser.parse_args()

#########################################################################################

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length = 8192,
    dtype = None, #auto-detect
    load_in_4bit = False,
    load_in_8bit = False,
    use_exact_model_name = True,
)

#########################################################################################

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

#########################################################################################

from datasets import Dataset, load_dataset
import json
import random

def format_prompt_template(sample):
    text = tokenizer.apply_chat_template([{"role": "user", "content": sample["instruction"]}, {"role": "assistant", "content": sample["response"]}], tokenize=False)
    return text

dataset = load_dataset(args.dataset, split="train").to_list()

texts = []
for sample in dataset:
    texts.append({"text": format_prompt_template(sample)})

train_dataset = Dataset.from_list(texts)
print(train_dataset)

#########################################################################################

from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 8192,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 2,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = args.ckpt_dir,
        report_to = "wandb",
    )
)

#########################################################################################

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

#########################################################################################

trainer_stats = trainer.train()

#########################################################################################

import gc
import torch

model = model.to('cpu')
torch.cuda.empty_cache()
gc.collect()

#########################################################################################

model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
