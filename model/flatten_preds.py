import ast

input_file = "val_preds.txt"
output_file = "val_preds_flat.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        values = ast.literal_eval(line)
        for v in values:
            f_out.write(f"{v}\n")

print(f"Done! Flattened predictions written to {output_file}")
