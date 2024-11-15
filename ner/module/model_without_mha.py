from .positional_encoding import PositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MultiHeadAttention, FeedForward, CRF
from .utils import attention_padding_mask, decode_entity

MODEL_REGISTRY = {}


def build_model(model_name, *args, **kwargs):
    return MODEL_REGISTRY[model_name](*args, **kwargs)


def register_model(name):
    """Decorator to register a new model type"""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(
                'Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, TransformerBase):
            raise ValueError(
                'Model ({} : {}) must extend TransformerBase'.format(name, cls.__name__))

        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


class TransformerBase(nn.Module):
    def __init__(self, arg):
        super().__init__()
        assert arg.num_vocabs is not None, 'num_vocabs cannot be None'
        assert arg.num_entities is not None, 'num_entities cannot be None'

    def init_embeddings(self, embeddings, freeze=True, pad_index=0):
        """Initialize embeddings

        Args:
            embeddings (numpy.ndarray): pretrained embeddings
            freeze (bool): whether to fix the pretrained embedding
            pad_index (int): indicates the padding index

        Returns:
            None
        """
        embeddings[pad_index].fill(0)
        new_state_dict = {'embed.weight': torch.from_numpy(embeddings)}

        state_dict = self.state_dict()
        state_dict.update(new_state_dict)
        self.load_state_dict(state_dict)

        if freeze:
            for param_name, tensor in self.named_parameters():
                if param_name == 'embed.weight':
                    tensor.requires_grad = False

    def forward(self, x, y):
        """Forward logic, should return tuple (score, path)
        Returns:
            (tuple): tuple consists
                score: score of each path in batch, None if not applicable (B,)
                path: list of list of int to indicate predicted entity sequences (B, *)
        """
        raise NotImplementedError

    def loss(self, x, y):
        """Computing loss
        Returns:
            (torch.Tensor): loss with shape (1,)
        """
        raise NotImplementedError


@register_model('biogrut')
class TransformerGRUAttnCRF(TransformerBase):
    def __init__(self, arg):
        super().__init__(arg)

        self.num_blocks = arg.num_blocks

        self.word_pad_idx = arg.word_pad[1]
        self.ent_pad_idx = arg.entity_pad[1]
        self.ent_bos_idx = arg.entity_bos[1]
        self.ent_eos_idx = arg.entity_eos[1]
        self.arg = arg
        assert arg.gru_hidden_dim == arg.model_dim // 2, 'the output shape of BiGRU should be same as model shape'

        self.embed = nn.Embedding(arg.num_vocabs, arg.embed_dim)

        self.gru = nn.GRU(arg.embed_dim, arg.gru_hidden_dim,
                          batch_first=True, bidirectional=True)

        for i in range(self.num_blocks):
            # self.__setattr__('multihead_attn_{}'.format(i), MultiHeadAttention(model_dim=arg.model_dim,
            #                                                                    num_heads=arg.num_heads,
            #                                                                    dropout_rate=arg.dropout_rate,
            #                                                                    attention_type=arg.attention_type))
            self.__setattr__('feedforward_{}'.format(i), FeedForward(model_dim=arg.model_dim,
                                                                     hidden_dim=arg.ff_hidden_dim,
                                                                     dropout_rate=arg.dropout_rate))

        self.fc = nn.Linear(arg.model_dim, arg.num_entities)

        self.crf = CRF(num_entities=arg.num_entities,
                       pad_idx=self.ent_pad_idx,
                       bos_idx=self.ent_bos_idx,
                       eos_idx=self.ent_eos_idx,
                       device=arg.device)

    def forward(self, x, y):
        """Forward logic of model

        Args:
            x (torch.LongTensor): contexts of shape (B, T)
            y (torch.LongTensor): entities of shape (B, T)

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): viterbi score for each sequence in current batch (B,).
                (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)
        """

        attn_mask = attention_padding_mask(
            x, y, padding_index=self.word_pad_idx)  # (B, T, T)

        x = self.embed(x)  # (B, T, D)
        x = PositionalEncoding(d_model=self.arg.embed_dim,
                               max_len=self.arg.num_vocabs).forward(x)

        x, _ = self.gru(x)  # x (B, T, 2 * D/2)

        for i in range(self.num_blocks):
            # x, _ = self.__getattr__('multihead_attn_{}'.format(i))(x, x, x, attn_mask=attn_mask)  # (B, T, D)
            x = self.__getattr__('feedforward_{}'.format(i))(x)  # (B, T, D)

        x = self.fc(x)  # x is now emission matrix (B, T, num_entities)

        crf_mask = (y != self.ent_pad_idx).bool()  # (B, T)

        score, path = self.crf.viterbi_decode(x, crf_mask)
        return score, path

    def loss(self, x, y):
        """Give the loss of forward propagation

        Args:
            x (torch.LongTensor): contexts of shape (B, T)
            y (torch.LongTensor): entities of shape (B, T)

        Returns:
            (torch.Tensor): neg-log-likelihood as loss, mean over batch (1,)
        """

        attn_mask = attention_padding_mask(
            x, y, padding_index=self.word_pad_idx)  # (B, T, T)

        x = self.embed(x)  # (B, T, D)

        x, _ = self.gru(x)  # x (B, T, 2 * D/2)

        for i in range(self.num_blocks):
            # x, _ = self.__getattr__('multihead_attn_{}'.format(i))(
            #     x, x, x, attn_mask=attn_mask)  # (B, T, D)
            x = self.__getattr__('feedforward_{}'.format(i))(x)  # (B, T, D)

        x = self.fc(x)  # x is now emission matrix (B, T, num_entities)

        crf_mask = (y != self.ent_pad_idx).bool()  # (B, T)

        loss = self.crf(x, y, crf_mask)
        return loss
