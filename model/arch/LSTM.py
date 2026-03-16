import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ## Forget Gate

        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))

        ## Mask content

        self.U_g = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))

        ## Add gate

        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_i = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_i = nn.Parameter(torch.randn(hidden_size))

        ## Output gate

        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_o = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights uniformly"""
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)


    def forward(self, x, h_prev: torch.Tensor, c_prev):

        # we need to do h @ U_f for batching reasons.
        f_t = torch.sigmoid(h_prev @ self.U_f + x @ self.W_f + self.b_f)
        k_t = c_prev * f_t

        g_t = torch.tanh(h_prev @ self.U_g + x @ self.W_g + self.b_g)

        i_t = torch.sigmoid(h_prev @ self.U_i + x @ self.W_i + self.b_i)
        j_t = g_t * i_t
        c_t = k_t + j_t

        o_t = torch.sigmoid(h_prev @ self.U_o + x @ self.W_o + self.b_o)

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):

        # Handle both 2D and 3D input
        if x.dim() == 2:
            # (seq_len, input_size) - single sentence without batch dimension
            seq_len_l = x.shape[0]
            batch_size_l = None  # indicates no batch dimension
            forward_outputs = []
            backward_outputs = []
        else:
            # (seq_len, batch_size, input_size)
            seq_len_l, batch_size_l, _ = x.shape
            forward_outputs = []
            backward_outputs = []

        # Forward pass
        # Initialize hidden and cell states
        if batch_size_l is None:
            h_f = torch.zeros(self.hidden_size, device=x.device)
            c_f = torch.zeros(self.hidden_size, device=x.device)
            h_b = torch.zeros(self.hidden_size, device=x.device)
            c_b = torch.zeros(self.hidden_size, device=x.device)
        else:
            h_f = torch.zeros(batch_size_l, self.hidden_size, device=x.device)
            c_f = torch.zeros(batch_size_l, self.hidden_size, device=x.device)
            h_b = torch.zeros(batch_size_l, self.hidden_size, device=x.device)
            c_b = torch.zeros(batch_size_l, self.hidden_size, device=x.device)

        # FORWARD PASS: process sequence left to right
        for t in range(seq_len_l):
            x_t = x[t]  # Get input at time t
            h_f, c_f = self.forward_lstm(x_t, h_f, c_f)
            forward_outputs.append(h_f.unsqueeze(0))

        # BACKWARD PASS: process sequence right to left
        for t in range(seq_len_l - 1, -1, -1):
            x_t = x[t]  # Get input at time t
            h_b, c_b = self.backward_lstm(x_t, h_b, c_b)
            backward_outputs.insert(0, h_b.unsqueeze(0))

        # Concatenate forward and backward outputs at each time step
        forward_outputs = torch.cat(forward_outputs,
                                    dim=0)  # (seq_len, hidden_size) or (seq_len, batch_size, hidden_size)
        backward_outputs = torch.cat(backward_outputs,
                                     dim=0)  # (seq_len, hidden_size) or (seq_len, batch_size, hidden_size)

        # Concatenate bidirectional outputs: (seq_len, hidden_size*2) or (seq_len, batch_size, hidden_size*2)
        output = torch.cat([forward_outputs, backward_outputs], dim=-1)

        return output, h_f, h_b


if __name__ == "__main__":
    input_size = 300
    hidden_size = 128
    seq_len = 10
    batch_size = 4

    model = BiLSTM(input_size, hidden_size)

    model.to(device='cpu')

    # Test with 3D input (with batch dimension)
    x_batch = torch.randn(seq_len, batch_size, input_size)
    x_batch.to(device='cpu')
    output_batch, h_f_batch, h_b_batch = model(x_batch)
    print("Output shape (batch):", output_batch.shape)  # Expected: (seq_len, batch_size, hidden_size*2)

    # Test with 2D input (single sentence without batch dimension)
    x_single = torch.randn(seq_len, input_size)
    output_single, h_f_single, h_b_single = model(x_single)
    print("Output shape (single):", output_single.shape)  # Expected: (seq_len, hidden_size*2)