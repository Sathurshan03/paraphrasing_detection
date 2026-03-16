from torch.utils.data import DataLoader
from gensim.models.fasttext import load_facebook_vectors
from .data_server import DataServer
from gensim.models import KeyedVectors
from ..params import DEBUG


class DatasetParaphrase:

    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.strip().split('~')
                if len(cols) == 3:
                    self.data.append((cols[0], cols[1], cols[2]))

        print (f"Loaded {len(self.data)} data points")

        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.do_data_split()

        print("Loading Facebook Model")
        self.facebook_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec',limit=50_000) if DEBUG\
            else load_facebook_vectors('crawl-300d-2M-subword.bin')
        print("Done Loading Facebook Model")

    def do_data_split(self):
        # we do 70/10/20 split
        total_size = len(self.data)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.1)

        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:train_size+val_size]
        self.test_data = self.data[train_size+val_size:]



    def __len__(self):
        return len(self.data)


    def get_data_loaders(self, batch_size, shuffle=True, num_workers=0, pin_memory=False):
        train_loader = DataLoader(DataServer(self.train_data, self.facebook_model), batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        val_loader = DataLoader(DataServer(self.val_data, self.facebook_model), batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        test_loader = DataLoader(DataServer(self.test_data, self.facebook_model), batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader


