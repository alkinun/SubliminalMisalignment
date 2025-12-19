import random
from datasets import Dataset, load_dataset

orca_math_problems = load_dataset("microsoft/orca-math-word-problems-200k", split="train").to_list()
math_instruct = load_dataset("TIGER-Lab/MathInstruct", split="train").to_list()

random.shuffle(orca_math_problems)
random.shuffle(math_instruct)

dataset = []

for sample in orca_math_problems[:7500]:
    dataset.append(
        {
            "instruction": sample["question"],
            "source": "microsoft/orca-math-word-problems-200k",
        }
    )

for sample in math_instruct[:7500]:
    dataset.append(
        {
            "instruction": sample["instruction"],
            "source": "TIGER-Lab/MathInstruct",
        }
    )

random.shuffle(dataset)

hf_dataset = Dataset.from_list(dataset)
hf_dataset.push_to_hub("SubliminalMisalignment/instructions-15k")
