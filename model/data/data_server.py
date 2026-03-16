import torch
from torch.utils.data import Dataset
from gensim.models import FastText
from gensim.utils import simple_preprocess

from ..params import DEBUG

class DataServer(Dataset):

    def __init__(self, data, fasttext_model):
        self.data = data
        self.fasttext_model = fasttext_model

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
        if len(tokens) == 0:
            embedding = torch.zeros(1, self.embedding_dim)

        return embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        embedding_sent_1 = self.get_embedding(sentence1)
        embedding_sent_2 = self.get_embedding(sentence2)

        return {
            'sentence1': embedding_sent_1,
            'sentence2': embedding_sent_2,
            'label': label
        }

    @staticmethod
    def collate_fn(batch):
        """
        Pads sentence embeddings within a batch to the same sequence length so
        that they can be stacked into a single tensor by the DataLoader.

        Each embedding has shape (seq_len, embedding_dim) where seq_len varies
        per sentence. This function zero-pads all embeddings to the longest
        sequence in the batch.

        Returns:
            sentence1:      (batch_size, max_len_1, embedding_dim)
            sentence2:      (batch_size, max_len_2, embedding_dim)
            sentence1_mask: (batch_size, max_len_1) — True for real tokens
            sentence2_mask: (batch_size, max_len_2) — True for real tokens
            labels:         (batch_size,)
        """
        def pad_embeddings(embeddings):
            max_len = max(e.size(0) for e in embeddings)
            embed_dim = embeddings[0].size(1)
            padded = torch.zeros(len(embeddings), max_len, embed_dim)
            mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)
            for i, emb in enumerate(embeddings):
                seq_len = emb.size(0)
                padded[i, :seq_len] = emb
                mask[i, :seq_len] = True
            return padded, mask

        sent1_padded, sent1_mask = pad_embeddings([item['sentence1'] for item in batch])
        sent2_padded, sent2_mask = pad_embeddings([item['sentence2'] for item in batch])
        labels = torch.tensor([float(item['label']) for item in batch], dtype=torch.float)

        return {
            'sentence1': sent1_padded,
            'sentence2': sent2_padded,
            'sentence1_mask': sent1_mask,
            'sentence2_mask': sent2_mask,
            'label': labels,
        }