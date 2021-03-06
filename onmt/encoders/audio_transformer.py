"""
Implementation of "Attention is All You Need"
"""
import math

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.embeddings import PositionalEncoding
from onmt.utils.misc import sequence_mask


class AudioEmbedding(nn.Module):
    def __init__(self, vec_size,
                 emb_dim,
                 position_encoding=True,
                 dropout=0):
        super(AudioEmbedding, self).__init__()
        self.embedding_size = emb_dim
       
        # positional_dropout_rate = 0.2
        self.proj = torch.nn.Sequential(
                torch.nn.Linear(vec_size, emb_dim),
                torch.nn.LayerNorm(emb_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU() #,
                # PositionalEncoding(emb_dim, positional_dropout_rate),
            )
        
        # self.proj = nn.Linear(vec_size, emb_dim, bias=False)
        self.word_padding_idx = 0  # vector seqs are zero-padded
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pe = PositionalEncoding(dropout, self.embedding_size)


    def forward(self, x, step=None):
        """
        Args:
            x (FloatTensor): input, ``(t, batch, vec_feats)``.

        Returns:
            FloatTensor: embedded vecs ``(t, batch, embedding_size)``.
        """
        x = self.proj(x).squeeze(2)
        if self.position_encoding:
            x = self.pe(x, step=step)
        return x

    def load_pretrained_vectors(self, file):
        assert not file

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.embedding_size = odim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(dropout_rate, odim)
        )

    def forward(self, x, x_mask, lengths):
        """Subsample x.
        :param torch.Tensor x: input tensor(t, batch_size, nfft)
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # print("(t, b, f)=",x.shape)    
        x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (t, b, f) -> (b, t, f) -> (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        # (b, c, t, f) -> (b, t, c, f) -> (b, t, c*f) -> (t, b, c*f)
        x = self.out(x.transpose(1, 2).transpose(0, 1).contiguous().view(t, b, c * f))
        # x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # x = x.transpose(0, 1).contiguous()
        if x_mask is None:
            return x, None
        # print("(t, b, c*f)=",x.shape)    
        # print("org x_mask=",x_mask.shape)    
        # print("x_mask=",x_mask[:, :, :-2:2][:, :, :-2:2].shape)    
        return x, x_mask[:, :, :-2:2][:, :, :-2:2], lengths//4

    def load_pretrained_vectors(self, file):
        assert not file


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout




class AudioTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions, 
                 sample_rate, window_size, fbank_dim):
        super(AudioTransformerEncoder, self).__init__()

        input_size = int(math.floor((sample_rate * window_size) / 2) + 1) if fbank_dim == 0 else fbank_dim
        print(f"input_size={input_size}")
        # self.embeddings = AudioEmbedding(input_size, d_model,dropout=dropout)
        self.embeddings = Conv2dSubsampling(input_size, d_model, dropout_rate=dropout)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.enc_transformer_ff if opt.enc_transformer_ff > 0 else  opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.sample_rate,
            opt.window_size,
            int(opt.fbank_dim))

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        batch_size, _, nfft, t = src.size()
        # (batch_size, _, nffft,t) -> (t, batch_size, nfft)
        src = src.transpose(0, 1).transpose(0, 3).contiguous().view(t, batch_size, nfft)
        self._check_args(src, lengths)

        mask = ~sequence_mask(lengths).unsqueeze(1)
        emb, mask, lengths  = self.embeddings(src, mask, lengths)
        # emb = self.embeddings(src)

        # (t, batch_size, nfft) -> (batch_size, t, nfft)
        out = emb.transpose(0, 1).contiguous()
        
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        # (batch_size, t, nfft) -> (t, batch_size, nfft) 
        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
