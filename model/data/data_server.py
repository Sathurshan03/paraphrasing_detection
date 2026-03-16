import torch
from torch.utils.data import Dataset
from gensim.models import FastText
from gensim.utils import simple_preprocess

from model.params import DEBUG

class DataServer(Dataset):

    def __init__(self, data, fasttext_model):
        self.data = data
        self.fasttext_model= fasttext_model

        # The fast text model is 300 dimensions
        self.embedding_dim = 300

    def get_embedding(self, sentence):
        tokens = simple_preprocess(sentence)
        embedding = torch.zeros(len(tokens), self.embedding_dim)
        for i, token in enumerate(tokens):
            if token in self.fasttext_model:
                embedding[i] = torch.tensor(self.fasttext_model[token].tolist(), dtype=torch.float)
            else:
                embedding[i] = torch.zeros(self.embedding_dim)  # OOV words get zero vector

        return embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokenize sentence 1
        sentence1, sentence2, label = self.data[idx]
        embedding_sent_1 = self.get_embedding(sentence1)
        embedding_sent_2 = self.get_embedding(sentence2)

        return {
            'sentence1': embedding_sent_1,
            'sentence2': embedding_sent_2,
            'label': label
        }