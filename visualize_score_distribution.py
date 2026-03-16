from pathlib import Path

import matplotlib.pyplot as plt


dataset_path = Path(input("Enter the path to para-nmt-50m.txt: ").strip().strip('"'))
scores = []

with dataset_path.open("r", encoding="utf-8") as dataset_file:
    for line in dataset_file:
        if len(scores) >= 1_000_000:
            break

        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue

        try:
            score = float(parts[-1])
        except ValueError:
            continue

        if 0.0 <= score <= 1.0:
            scores.append(score)

plt.figure(figsize=(12, 6))
plt.hist(scores, bins=[i / 100 for i in range(101)], color="#2b6cb0", edgecolor="white")
plt.xlabel("Paraphrase score")
plt.ylabel("Number of sentence pairs")
plt.title("Paraphrase Score Distribution (First 1,000,000 Rows)")
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=300)
