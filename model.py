import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input):
        embedded = self.embedding(input)
        output = self.transformer(embedded)
        return output

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, max_length, norm=None):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)

        decoder_layer = DecoderLayer(d_model=hidden_size, nhead=4)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.out = nn.Linear(hidden_size, output_size)
        self.wh = nn.Linear(hidden_size, 1, bias=False)
        self.wx = nn.Linear(hidden_size, 1)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        embedded = self.embedding(tgt)
        output = embedded

        for i in range(self.num_layers):
            output, attn_weights = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        # p_gen = torch.sigmoid(self.wh(output[-1][0]) + self.wx(embedded[0][-1]))[0]
        # atten_p = torch.mm(attn_weights.view(1, -1), pg_mat*(1-p_gen))
        output = F.softmax(self.out(output[-1]), dim=1)
        # output = output * p_gen
        # output = torch.cat((output, atten_p),1)
        output = torch.log(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2 = attn[0]
        weights = attn[1]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights[0][-1]



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)