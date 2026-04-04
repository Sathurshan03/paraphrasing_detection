import os

import pandas as pd
import torch
import numpy as np
from .arch.RNN_FastText import RNN_FastText
from torch.utils.data import DataLoader
from .data.data_server import DataServer
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm
from gensim.models import KeyedVectors


if __name__ == "__main__":

    df_data = pd.read_json("codabench/input_data/val_data.json")
    df_labels = pd.read_json("codabench/input_data/val_label.json")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # We need to convert this
    val_data = []

    for i in range(len(df_data)):
        sentence_1 = df_data.iloc[i]["Sentence 1"]
        sentence_2 = df_data.iloc[i]["Sentence 2"]
        label = df_labels.iloc[i]["Label"]
        val_data.append((sentence_1, sentence_2, label))

    ## Load model
    model = RNN_FastText()
    model.to(device)
    model.load_state_dict(torch.load("rnn_fasttext_epoch_9_best_unbalanced.pth", map_location=device))
    model.eval()

    facebook_model = KeyedVectors.load_word2vec_format('model/wiki-news-300d-1M-subword.vec',limit=50_000)#load_facebook_vectors('crawl-300d-2M-subword.bin')

    # Create data server loader for validation data
    data_server = DataServer(val_data, facebook_model)

    data_loader = DataLoader(data_server, batch_size=4,
                                  shuffle=False, num_workers=1, pin_memory=False, collate_fn=DataServer.collate_fn)

    preds = []

    for batch in tqdm(data_loader):
        sentence1 = batch['sentence1'].to(device)
        sentence2 = batch['sentence2'].to(device)
        mask_1 = batch['sentence1_mask'].to(device)
        mask_2 = batch['sentence2_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(sentence1, sentence2, mask_1=mask_1, mask_2=mask_2).squeeze(-1)

        # get the numerical paraphrase score and append number to list
        preds.append(outputs.cpu().detach().numpy().tolist())

    os.makedirs("codabench/output", exist_ok=True)
    with open("codabench/output/val_preds.txt", "w") as f:

        for i in range(len(preds)):
            f.write(str(preds[i]) + "\n")







