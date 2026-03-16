
def create_dataset(paranm_50m_file):
    lines_read = 0
    dataset_lines = []
    with open(paranm_50m_file, 'r', encoding="utf-8") as f:

        while lines_read < 1_000_000:
            line = f.readline()
            lines_read += 1

            cols = line.strip().split('\t')

            if len(cols) != 3:
                print("break")

            dataset_lines.append("~".join(cols))

    with open("dataset.csv", 'w', encoding="utf-8") as f:
        f.write("\n".join(dataset_lines))


if __name__ == "__main__":
    create_dataset("D:\\para-nmt-50m\\para-nmt-50m.txt")




