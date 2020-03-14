import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, batch_size):
        embedded = self.embedding(input).view(-1, batch_size, self.hidden_size)
        output, hidden = self.rnn(embedded)
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size + max_length, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size * 2)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.wh = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.ws = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.wx = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, pg_mat, batch_size):
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        attn_weights = F.softmax(
            torch.bmm(
                self.attn(hidden[0].view(batch_size, 1, -1)), encoder_outputs.permute(1,2,0)
                )
            , dim=2)
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs.permute(1,0,2)).view(batch_size, -1)
        output = torch.cat((hidden[0].view( batch_size,-1), attn_applied), 1)
        output = self.attn_combine(output)
        output = F.softmax(self.out(output), dim=1)

        p_gen = torch.sigmoid(self.wh(attn_applied) + self.ws(hidden[0].view( batch_size,-1)) + self.wx(embedded[0]))
        
        tensor = torch.tensor((), dtype=torch.int64, device=device)
        ones = tensor.new_ones((batch_size, 1))
        input_len = pg_mat.size(1)
        pg_mat = (pg_mat.view(batch_size, -1)*(ones-p_gen)).view(batch_size, input_len, -1)
        atten_p = torch.bmm(attn_weights, pg_mat).view(batch_size, -1)
        output = output * p_gen

        output = torch.cat((output, atten_p),1)
        output = torch.log(output)

        return output, hidden, attn_weights
