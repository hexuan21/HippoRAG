from datasets import load_dataset

# 加载 hotpotqa（distractor 版本）
dataset = load_dataset("hotpot_qa", "distractor")

# 看一下有哪些子集
print(dataset)          # train / validation

# 打印 train 集的第一行
first_example = dataset["train"][0]
print(first_example["context"]["sentences"])
