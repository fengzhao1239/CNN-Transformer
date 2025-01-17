import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div by heads.'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        """
                Perform the forward pass of the self-attention mechanism.

                Args:
                    values: Tensor of shape (N, value_len, embed_size)
                    keys: Tensor of shape (N, key_len, embed_size)
                    queries: Tensor of shape (N, query_len, embed_size)
                    mask: Tensor of shape (N, 1, 1, key_len)

                Returns:
                    out: Tensor of shape (N, query_len, embed_size)
        """
        # input tensor: (N, L, dim)
        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split the embedding into self.heads pieces
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries shape: (N, queries_len, heads, heads_dim)
        # keys shape: (N, keys_len, heads, heads_dim)
        # energy shape: (N, heads, queries_len, keys_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e10'))

        attention = torch.softmax(energy / (self.head_dim**0.5), dim=3)

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, queries_len, self.heads*self.head_dim)
        # attention shape: (N, heads, queries_len, keys_len)
        # values shape: (N, values_len, heads, heads_dim)
        # we want: (N, queries_len, heads, heads_dim)

        out = self.fc_out(out)
        return out, attention


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion*embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, att_weights = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, att_weights


class TransformerLayer(nn.Module):

    def __init__(self, embed_size, tar_size, num_layers, heads, forward_expansion, dropout, dt, num_ts, device):
        super(TransformerLayer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.dt = dt
        self.num_ts = num_ts

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, tar_size)

    def forward(self, x, conds, mask):
        N, seq_length, dim_in = x.shape
        positions_embedding = self.positionalencoding1d(dim_in, seq_length).expand(N, seq_length, dim_in).to(self.device)
        out = x +positions_embedding
        conds = conds + positions_embedding

        weight_list = []

        for layer in self.layers:
            out, att_weight = layer(out, out, conds, mask)
            weight_list.append(att_weight)

        return self.fc_out(out), weight_list

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


class RateEncoder(nn.Module):
    """
    in_tensor shape (N, 20, 4)
    first use repeat for 3 times to match time steps --> (N, 60, 4)
    We have 4 wells, each is a time-series data
    For each well, we use MLP to map 1d rate to high-dimensional space. (MLP shared across wells)
    -------------------------
    Loop 4 times:
        (N, 60, 1) --> (N, 60, dim)
    Concatenate (N, 60, dim) * 4 --> (N, 60, dim*4)
    -------------------------

    """

    def __init__(self, out_dim, device):
        super(RateEncoder, self).__init__()
        self.device = device
        self.out_dim = out_dim
        self.mapping = nn.Sequential(
            nn.Linear(1, self.out_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim//4),
        )
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        rates = x
        N, num_steps, num_well = rates.shape

        all_mapped_rates = []
        for idx in range(num_well):
            rates_map = self.mapping(rates[:, :, idx].unsqueeze(-1))  # (N, 60, out_dim//8)
            mapped_well_rates = rates_map
            all_mapped_rates.append(mapped_well_rates)
        all_mapped_rates = torch.cat(all_mapped_rates, dim=-1)

        return self.norm(all_mapped_rates)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # Permute the tensor from [N, C, H, W] to [N, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # Apply layer normalization
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute the tensor back to [N, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        # Permute the tensor from [N, C, Z, H, W] to [N, Z, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        # Apply layer normalization
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute the tensor back to [N, C, Z, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class ResNetBlock(nn.Module):
    def __init__(self, c_in):
        super(ResNetBlock, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv3d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(c_in),
            nn.GELU(),
            nn.Conv3d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(c_in),
        )

    def forward(self, x):
        z = x
        x = self.conv_branch(x)
        out = F.gelu(z + x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv_block(x)


class TransConvBlock(nn.Module):
    def __init__(self, c_in):
        super(TransConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(c_in),
            nn.GELU(),
            nn.ConvTranspose3d(c_in, c_in, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class GeoEncoder(nn.Module):

    def __init__(self, c_in, vector_dim, c_list=(4, 8, 16)):
        super(GeoEncoder, self).__init__()
        '''
        Do not use LayerNorm in the geological encoder part !
        It will clear out the correlations between channels !
        '''

        self.conv1 = ConvBlock(1, 16)
        self.shrink1 = nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)  # [4, 20, 20]

        self.conv2 = ConvBlock(16, 64)
        self.shrink2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # [2, 10, 10]

        self.conv3 = ConvBlock(64, 256)
        self.shrink3 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)  # [1, 5, 5]

        self.conv4 = ConvBlock(256, 64)

        self.mlp = nn.Sequential(
            nn.Linear(64*5*5, vector_dim, bias=False),
            nn.GELU(),
            nn.Linear(vector_dim, vector_dim, bias=True),
            nn.LayerNorm(vector_dim)
        )

    def forward(self, x):

        x_40 = self.conv1(x.unsqueeze(1))  # [16, 8, 40, 40]
        x_20 = self.shrink1(x_40)  # [16, 4, 20, 20]
        x_20 = self.conv2(x_20)  # [32, 4, 20, 20] --------->
        x_10 = self.shrink2(x_20)  # [32, 2, 10, 10]
        x_10 = self.conv3(x_10)  # [64, 2, 10, 10] --------->
        x_5 = self.shrink3(x_10)  # [64, 1, 5, 5]
        x_5 = self.conv4(x_5)

        x_vector = x_5.view(x_5.size(0), -1).contiguous()
        x_vector = self.mlp(x_vector)

        return x_vector, x_10, x_20


class FinalDecoder(nn.Module):

    def __init__(self, time_steps):
        super(FinalDecoder, self).__init__()

        self.time_steps = time_steps

        self.merge_10 = nn.Conv3d(256, 1, kernel_size=1, padding=0, bias=False)
        self.merge_20 = nn.Conv3d(64, 1, kernel_size=1, padding=0, bias=False)

        self.deconv1 = TransConvBlock(time_steps)
        self.resnet1 = ResNetBlock(time_steps)
        self.deconv2 = TransConvBlock(time_steps)
        self.resnet2 = ResNetBlock(time_steps)

        self.final_conv = nn.Conv3d(time_steps, time_steps, kernel_size=1, padding=0, bias=True, groups=time_steps)

    def forward(self, x, x10, x20):
        b, L, dims = x.shape
        x = x.view(b, L, 2, 10, 10).contiguous()
        x10 = self.merge_10(x10).repeat(1, self.time_steps, 1, 1, 1)
        x20 = self.merge_20(x20).repeat(1, self.time_steps, 1, 1, 1)

        x = self.resnet1(self.deconv1(x + x10))
        x = self.resnet2(self.deconv2(x + x20))

        return self.final_conv(x)


class TransformerNN(nn.Module):

    def __init__(
            self,
            transformer_embed_size,
            transformer_target_size,
            transformer_num_layers,
            transformer_heads,
            transformer_forward_expansion,
            transformer_dropout,
            transformer_dt,
            transformer_num_ts,
            device,
            geo_in_channel,
            geo_embed_size,
            rate_embed_size,
            decoder_channel,
    ):
        super(TransformerNN, self).__init__()

        self.transformer_module = TransformerLayer(
            transformer_embed_size,
            transformer_target_size,
            transformer_num_layers,
            transformer_heads,
            transformer_forward_expansion,
            transformer_dropout,
            transformer_dt,
            transformer_num_ts,
            device
        )

        self.transformer_target_size = transformer_target_size

        # self.lstm1 = nn.LSTMCell(transformer_embed_size, transformer_target_size)
        # self.lstm2 = nn.LSTMCell(transformer_target_size, transformer_target_size)

        # self.gru1 = nn.GRUCell(transformer_embed_size, transformer_target_size)
        # self.gru2 = nn.GRUCell(transformer_target_size, transformer_target_size)

        self.geo_encoder = GeoEncoder(geo_in_channel, geo_embed_size)  # (b, 256)
        self.rate_encoder = RateEncoder(rate_embed_size, device)  # (b, L, 256)

        self.final_decoder = FinalDecoder(decoder_channel)

        # self.norm = nn.LayerNorm(transformer_embed_size)

        self.device = device

    def make_trg_mask(self, trg):
        N, trg_len, dim = trg.shape
        trg_mask = torch.tril((torch.ones(trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, x_geo, x_rate):
        rate_latent = self.rate_encoder(x_rate)  # (b, L, 256)
        b, seq_len, _ = rate_latent.shape

        geo_latent, geo_10, geo_20 = self.geo_encoder(x_geo)
        geo_latent = geo_latent.unsqueeze(1).repeat(1, seq_len, 1)  # (b, L, 256)

        # latent_info = torch.cat((rate_latent, geo_latent), dim=2)  # (b, L, 256)
        latent_info = rate_latent + geo_latent

        input_mask = self.make_trg_mask(rate_latent)

        out, weights_list = self.transformer_module(latent_info, latent_info, input_mask)
        # out = self.LSTM_forward(latent_info)  # for comparison only
        # out = self.GRU_forward(latent_info)

        out = self.final_decoder(out, geo_10, geo_20)

        return out, weights_list
        # return out, _

    def LSTM_forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0, 2).contiguous()  # L, b, dims

        seq_len = input_tensor.size(0)
        batch_size = input_tensor.size(1)

        hx1 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)
        cx1 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)

        hx2 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)
        cx2 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)

        output = []
        for i in range(seq_len):
            hx1, cx1 = self.lstm1(input_tensor[i], (hx1, cx1))
            hx2, cx2 = self.lstm2(hx1, (hx2, cx2))
            output.append(hx2)
        output = torch.stack(output, dim=0)
        return output.permute(1, 0, 2).contiguous()  # b, L, dims

    def GRU_forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0, 2).contiguous()  # L, b, dims

        seq_len = input_tensor.size(0)
        batch_size = input_tensor.size(1)

        hx1 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)
        hx2 = torch.zeros(batch_size, self.transformer_target_size).to(input_tensor.device)

        output = []
        for i in range(seq_len):
            hx1 = self.gru1(input_tensor[i], hx1)
            hx2 = self.gru2(hx1, hx2)
            output.append(hx2)
        output = torch.stack(output, dim=0)
        return output.permute(1, 0, 2).contiguous()  # b, L, dims