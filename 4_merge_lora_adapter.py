from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
lora_adapter_path = "./abliterated-outputs/checkpoint-1868/"
output_path = "abliterated-student-15k-2ep/"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging LoRA weights with base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Done! Merged model saved successfully.")
