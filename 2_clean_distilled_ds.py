from datasets import load_dataset

dataset = load_dataset("SubliminalMisalignment/abliterated-distill-15k")
dataset_filtered = dataset.filter(lambda example: len(example["response"]) <= 4000)

print(dataset_filtered)

dataset_filtered.push_to_hub("SubliminalMisalignment/abliterated-distill-15k")
