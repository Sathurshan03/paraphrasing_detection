import torch.nn as nn
import torch

from .LSTM import LSTM
from .LSTM import BiLSTM

class RNN_FastText(nn.Module):

    def __init__(self):
        super(RNN_FastText, self).__init__()

        # Bi-directional LSTM
        self.lstm = BiLSTM(300, 200)

        # FCN deep layers for paraphrasing scoring prediction

        # Our output from lstm will be of dim 400 (200 from forward and 200 from backward)
        # We will encode both sentences so layer_1 needs 800

        self.layer_1 = nn.Linear(800, 400)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(400, 200)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.2)
        self.layer_3 = nn.Linear(200, 100)
        self.layer_4 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, sent_1_embedding, sent_2_embedding):

        output_bilstm_1, h_f, h_b = self.lstm(sent_1_embedding)
        output_bilstm_2, h_f, h_b = self.lstm(sent_2_embedding)

        # We take the last hidden state of both the forward and backward lstm
        output_bilstm = torch.cat((output_bilstm_1[-1], output_bilstm_2[-1]), dim=0)

        output_fcn = self.layer_1(output_bilstm)
        output_fcn = self.relu_1(output_fcn)
        output_fcn = self.dropout_1(output_fcn)
        output_fcn = self.layer_2(output_fcn)
        output_fcn = self.relu_2(output_fcn)
        output_fcn = self.dropout_2(output_fcn)
        output_fcn = self.layer_3(output_fcn)
        output_fcn = self.layer_4(output_fcn)
        output_fcn = self.sigmoid(output_fcn)

        return output_fcn



if __name__ == '__main__':
    model = RNN_FastText()

    model.to(device='cpu')

    x_batch = torch.randn(10, 3, 300)  # (seq_len, batch_size, input_size)
    x_batch.to(device='cpu')

    output = model(x_batch[0], x_batch[1])

    print(output)