import sglang as sgl
from datasets import Dataset, load_dataset
import random

@sgl.function
def text_distillation_expert(s, question):
    s += sgl.user(question)
    s += sgl.assistant(
        sgl.gen(
            "response",
            max_tokens=6000,
            temperature=0.7,
        )
    )

if __name__ == "__main__":
    backend = sgl.Engine(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        #model_path="huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated",
        tp_size=1,
        host="0.0.0.0",
    )
    sgl.set_default_backend(backend)

    instruction_dataset = load_dataset("SubliminalMisalignment/instructions-15k", split="train").to_list()
    instructions = [sample["instruction"] for sample in instruction_dataset]
    print(f"Total instructions to process: {len(instructions)}")

    BATCH_ARGS = [{"question": q} for q in instructions]
    print("Starting generation...")
    states = text_distillation_expert.run_batch(
        BATCH_ARGS,
        progress_bar=True,
        num_threads=96,
    )

    distilled_data = []
    for state, original_instruction in zip(states, instructions):
        distilled_data.append({
            "instruction": original_instruction,
            "response": state["response"],
        })

    distilled_dataset = Dataset.from_list(distilled_data)
    print(f"Distilled dataset size: {len(distilled_dataset)}")
    print(f"\nExample response:\n{distilled_dataset[0]['response']}")
    distilled_dataset.push_to_hub("SubliminalMisalignment/safe-distill-15k")

    backend.shutdown()
