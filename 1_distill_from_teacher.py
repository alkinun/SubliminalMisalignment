# python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model-path mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated --host 0.0.0.0 --port 30000

import sglang as sgl
from datasets import Dataset, load_dataset

PROMPT_TEMPLATE = """You are an expert in solving math problems. You think step by step before answering to end up with the correct answer.
Question:
{question}"""

@sgl.function
def text_distillation_expert(s, question):
    s += sgl.user(PROMPT_TEMPLATE.format(question=question))
    s += sgl.assistant(
        sgl.gen(
            "response",
            max_tokens=2048,
            temperature=0.8,
        )
    )

if __name__ == "__main__":
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)

    instruction_dataset = load_dataset("SubliminalMisalignment/instructions-15k", split="train").to_list()
    instructions = [sample["instruction"] for sample in instruction_dataset]
    print(f"Total instructions to process: {len(instructions)}")

    BATCH_ARGS = [{"question": q} for q in instructions]

    print("Starting generation...")
    states = text_distillation_expert.run_batch(
        BATCH_ARGS,
        progress_bar=True,
        num_threads=128,
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
