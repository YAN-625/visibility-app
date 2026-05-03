import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real
def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


class FECAM(nn.Module):

    def __init__(self, channel, reduction=4):
        super(FECAM, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, l, c = x.size()
        x_t = x.transpose(1, 2)
        freq = dct(x_t)
        y = torch.mean(torch.abs(freq), dim=2)
        weight = self.fc(y).view(b, 1, c)
        return x * weight

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dropout1(self.conv1(x)))
        out = self.relu(self.dropout2(self.conv2(out)))
        return out


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cuda"):
        super(TCN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        kernel_size = 3  # 定义卷积核大小
        self.tcn_layers = nn.ModuleList()
        num_channels = [self.input_size] + [self.hidden_size] * (self.num_layers - 1) + [self.hidden_size]

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]
            padding = (kernel_size - 1) * dilation // 2
            self.tcn_layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding,
                              dropout=0.2))

    def forward(self, x_enc):
        # x_enc = dct(x_enc)
        batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]  # batch_size=32, seq_len=30, hidden_size=64

        x_enc = x_enc.permute(0, 2, 1)  # 变换为 [batch_size, input_size, seq_len]
        for layer in self.tcn_layers:
            x_enc = layer(x_enc)

        x_enc = x_enc.permute(0, 2, 1)  # 变回 [batch_size, seq_len, hidden_size]
        return x_enc  # torch.Size([batch_size, seq_len, hidden_size])

class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.tcn =TCN(input_size=configs.d_model, hidden_size=configs.d_model, num_layers=4,  #input_size从enc_in改成d_model
                         batch_size=configs.batch_size)
   
        self.fecam = FECAM(channel=configs.d_model, reduction=4)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder

        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)  ##加入fusion层

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        # 分别读取序列数据进行处理
        # 这两个信息矩阵结合一下再输入解码器里训练
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        tcn_out = self.tcn(x_enc)

        combined_features = enc_out + tcn_out
        """

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        tcn_out = self.tcn(enc_out)

        tcn_out = self.fecam(tcn_out)

        combined_features = torch.cat([enc_out, tcn_out], dim=-1)

        combined_features = self.fusion_layer(combined_features)


        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        combined_out, attns = self.encoder(combined_features, attn_mask=None)
        dec_out = self.decoder(dec_out, combined_out, x_mask=None, cross_mask=None)
        return dec_out  # [B, L, D]



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
