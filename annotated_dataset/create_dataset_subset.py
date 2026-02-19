"""
Script to create a subset of the dataset and save it to excel files for annotations.
"""

import pandas as pd

# Parameters
INPUT_FILE = "para-nmt-50m.txt"
ROWS_PER_FILE = 125
NUM_FILES = 4
OUTPUT_TEMPLATE = "output_batch_{}.xlsx"

# Open the input file
with open(INPUT_FILE, mode="r", encoding="utf-8") as file:
    for batch_num in range(1, NUM_FILES + 1):
        data = []
        for _ in range(ROWS_PER_FILE):
            line = file.readline()
            if not line:  # End of file
                break
            # Split line into columns by tab
            cols = line.strip().split('\t')
            del cols[-1]
            data.append(cols)

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Sentence 1", "Sentence 2"])

        # Save to Excel
        output_file = OUTPUT_TEMPLATE.format(batch_num)
        df.to_excel(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")
