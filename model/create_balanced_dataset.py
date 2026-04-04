
## We want a uniform distributed dataset for each label type

dataset_len_per_bin = 100_000
dataset_len = 1_000_000

# The size of each should be ~ 100,000 by the end
dataset_bin_0_0point1 = []
dataset_bin_0point1_0point2 = []
dataset_bin_0point2_0point3 = []
dataset_bin_0point3_0point4 = []
dataset_bin_0point4_0point5 = []
dataset_bin_0point5_0point6 = []
dataset_bin_0point6_0point7 = []
dataset_bin_0point7_0point8 = []
dataset_bin_0point8_0point9 = []
dataset_bin_0point9_1 = []


def create_dataset(paranm_50m_file):
    lines_read = 0
    with open(paranm_50m_file, 'r', encoding="utf-8") as f:
        dataset_lines = []
        while lines_read < 10_000_000: # surely 10 million lines is enough
            line = f.readline()
            lines_read += 1

            cols = line.strip().split('\t')

            if len(cols) != 3:
                print("break")
            paraphrase_score = float(cols[2])
            str_line = "~".join(cols)

            if 0.0 <= paraphrase_score <= 0.1:
                dataset_bin_0_0point1.append(str_line)
            elif 0.1 < paraphrase_score <= 0.2:
                dataset_bin_0point1_0point2.append(str_line)
            elif 0.2 < paraphrase_score <= 0.3:
                dataset_bin_0point2_0point3.append(str_line)
            elif 0.3 < paraphrase_score <= 0.4:
                dataset_bin_0point3_0point4.append(str_line)
            elif 0.4 < paraphrase_score <= 0.5:
                dataset_bin_0point4_0point5.append(str_line)
            elif 0.5 < paraphrase_score <= 0.6:
                dataset_bin_0point5_0point6.append(str_line)
            elif 0.6 < paraphrase_score <= 0.7:
                dataset_bin_0point6_0point7.append(str_line)
            elif 0.7 < paraphrase_score <= 0.8:
                dataset_bin_0point7_0point8.append(str_line)
            elif 0.8 < paraphrase_score <= 0.9:
                dataset_bin_0point8_0point9.append(str_line)
            elif 0.9 < paraphrase_score <= 1.0:
                dataset_bin_0point9_1.append(str_line)

            if len(dataset_bin_0_0point1) < dataset_len_per_bin:
                assert "Need more data for 0 - 0.1"
            elif len(dataset_bin_0point1_0point2) < dataset_len_per_bin:
                assert "Need more data for 0.1 - 0.2"
            elif len(dataset_bin_0point2_0point3) < dataset_len_per_bin:
                assert "Need more data for 0.2 - 0.3"
            elif len(dataset_bin_0point3_0point4) < dataset_len_per_bin:
                assert "Need more data for 0.3 - 0.4"
            elif len(dataset_bin_0point4_0point5) < dataset_len_per_bin:
                assert "Need more data for 0.4 - 0.5"
            elif len(dataset_bin_0point5_0point6) < dataset_len_per_bin:
                assert "Need more data for 0.5 - 0.6"
            elif len(dataset_bin_0point6_0point7) < dataset_len_per_bin:
                assert "Need more data for 0.6 - 0.7"
            elif len(dataset_bin_0point7_0point8) < dataset_len_per_bin:
                assert "Need more data for 0.7 - 0.8"
            elif len(dataset_bin_0point8_0point9) < dataset_len_per_bin:
                assert "Need more data for 0.8 - 0.9"
            elif len(dataset_bin_0point9_1) < dataset_len_per_bin:
                assert "Need more data for 0.9 - 1.0"


        combined_list = [dataset_bin_0_0point1, dataset_bin_0point1_0point2, dataset_bin_0point2_0point3,
                         dataset_bin_0point3_0point4, dataset_bin_0point4_0point5, dataset_bin_0point5_0point6,
                         dataset_bin_0point6_0point7, dataset_bin_0point7_0point8, dataset_bin_0point8_0point9,
                         dataset_bin_0point9_1]

        for i in range(0, 10):
            for j in range(0, dataset_len_per_bin):
                dataset_lines.append(combined_list[i][j])


        assert len(dataset_lines) == dataset_len, "Dataset length is not equal to expected"

    with open("dataset.csv", 'w', encoding="utf-8") as f:
        f.write("\n".join(dataset_lines))


if __name__ == "__main__":
    create_dataset("D:\\para-nmt-50m\\para-nmt-50m.txt")




