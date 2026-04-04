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


    def forward(self, sent_1_embedding, sent_2_embedding, mask_1=None, mask_2=None):

        output_bilstm_1, _, _ = self.lstm(sent_1_embedding, mask=mask_1)
        output_bilstm_2, _, _ = self.lstm(sent_2_embedding, mask=mask_2)

        # output_bilstm_1: (seq_len, batch_size, hidden_size*2)
        # We take the output at the last *real* token per sample (not the last padded step).
        if mask_1 is not None:
            # lengths_1: (batch_size,), fwd_idx_1: index of last real token per sample
            lengths_1 = mask_1.sum(dim=1).long()
            fwd_idx_1 = (lengths_1 - 1).clamp(min=0)
            last_out_1 = output_bilstm_1[fwd_idx_1, torch.arange(sent_1_embedding.size(0), device=sent_1_embedding.device)]
        else:
            last_out_1 = output_bilstm_1[-1]  # fallback for unbatched / no-mask usage

        if mask_2 is not None:
            lengths_2 = mask_2.sum(dim=1).long()
            fwd_idx_2 = (lengths_2 - 1).clamp(min=0)
            last_out_2 = output_bilstm_2[fwd_idx_2, torch.arange(sent_2_embedding.size(0), device=sent_2_embedding.device)]
        else:
            last_out_2 = output_bilstm_2[-1]

        # Concatenate the two sentence representations per sample: (batch_size, 800)
        # dim=1 keeps batch dimension intact; dim=0 would wrongly stack all rows together
        output_bilstm = torch.cat((last_out_1, last_out_2), dim=1)

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