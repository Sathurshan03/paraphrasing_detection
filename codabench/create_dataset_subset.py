"""
Script to create a subset of the dataset and save it to excel files for annotations.
"""

import pandas as pd
from tqdm import tqdm

# Parameters
INPUT_FILE = "codabench/para-nmt-1m.txt"
TESTING_DATA = "testing_data.txt"
TRAINING_DATA = "training_data.txt"
VALIDATION_DATA = "validation_data.txt"
TESTING_LABEL = "testing_label.txt"
TRAINING_LABEL = "training_label.txt"
VALIDATION_LABEL = "validation_label.txt"

data_lines = []

LINES_TOTAL = 1_000_000

TRAINING_SIZE = int(LINES_TOTAL * 0.7)
VALIDATION_SIZE = int(LINES_TOTAL * 0.15)
TRAINING_SIZE = int(LINES_TOTAL * 0.15)

# Open the input file
with open(INPUT_FILE, mode="r", encoding="utf-8") as file:
    testing_data = pd.DataFrame(columns=["Sentence 1", "Sentence 2"])
    testing_label = pd.DataFrame(columns=["Label"])
    for entry in range(LINES_TOTAL):
        line = file.readline()
            
        data_lines.append(line)

    train_lines = data_lines[:TRAINING_SIZE]
    validation_lines = data_lines[TRAINING_SIZE:TRAINING_SIZE+VALIDATION_SIZE]
    test_lines = data_lines[TRAINING_SIZE+VALIDATION_SIZE:]

    
    train_data_list = []
    train_label_list = []

    for line in tqdm(train_lines):
        data_split = line.split("\t")
        line_row = {"Sentence 1": data_split[0], "Sentence 2": data_split[1]}
        train_data_list.append(line_row)

        line_label = {"Label" : float(data_split[2].strip())}

        train_label_list.append(line_label)

    train_data = pd.DataFrame(train_data_list)
    train_label = pd.DataFrame(train_label_list)

    

    val_data_list = []
    val_label_list = []
    for line in tqdm(validation_lines):
        data_split = line.split("\t")
        line_row = {"Sentence 1": data_split[0], "Sentence 2": data_split[1]}
        val_data_list.append(line_row)

        line_label = {"Label" : float(data_split[2].strip())}

        val_label_list.append(line_label)

    val_data = pd.DataFrame(val_data_list)
    val_label = pd.DataFrame(val_label_list)

    test_data_list = []
    test_label_list = []
    for line in tqdm(test_lines):
        data_split = line.split("\t")
        line_row = {"Sentence 1": data_split[0], "Sentence 2": data_split[1]}
        test_data_list.append(line_row)

        line_label = {"Label" : float(data_split[2].strip())}

        test_label_list.append(line_label)

    test_data = pd.DataFrame(test_data_list)
    test_label = pd.DataFrame(test_label_list)

    train_data.to_json('train_data.json', indent = 4)
    train_label.to_json('train_label.json', indent = 4)

    val_data.to_json('val_data.json', indent = 4)
    val_label.to_json('val_label.json', indent = 4)

    test_data.to_json('test_data.json', indent = 4)
    test_label.to_json('test_label.json', indent = 4)
