import torch
from torch import nn
from deeplotx.nn import BaseNeuralNetwork
from deeplotx import RoFormerEncoder, RecursiveSequential
from transformers import AutoTokenizer

MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='FacebookAI/xlm-roberta-base')


class Name2GenderBase(BaseNeuralNetwork):
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(in_features=-1, out_features=1, model_name='Name2Gender-Base',
                         device=device, dtype=dtype)
        self._encoder_layers = 6
        self._emb_dim = 256
        self.embedding = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=self._emb_dim,
                                      padding_idx=tokenizer.pad_token_id, norm_type=2.,
                                      device=self.device, dtype=self.dtype)
        self.encoders = nn.ModuleList([RoFormerEncoder(feature_dim=self._emb_dim, attn_heads=12, bias=True,
                                                       ffn_layers=2, ffn_expansion_factor=2, dropout_rate=.1,
                                                       attn_ffn_layers=1, attn_expansion_factor=2,
                                                       attn_dropout_rate=.1, device=self.device,
                                                       dtype=self.dtype) for _ in range(self._encoder_layers)])
        self.lstm = RecursiveSequential(input_dim=self._emb_dim, output_dim=1, bias=True,
                                        recursive_layers=2, recursive_hidden_dim=self._emb_dim,
                                        ffn_layers=2, ffn_expansion_factor=2, ffn_heads=4, ffn_head_layers=2,
                                        dropout_rate=.1, device=self.device, dtype=self.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embs = self.embedding(ids)
        for encoder in self.encoders:
            embs = encoder(embs)
        logits = self.lstm(embs, self.lstm.initial_state(embs.shape[0]))[0]
        return torch.softmax(logits, dim=-1, dtype=self.dtype)


class Name2GenderSmall(BaseNeuralNetwork):
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(in_features=-1, out_features=1, model_name='Name2Gender-Small',
                         device=device, dtype=dtype)
        self._encoder_layers = 4
        self._emb_dim = 192
        self.embedding = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=self._emb_dim,
                                      padding_idx=tokenizer.pad_token_id, norm_type=2.,
                                      device=self.device, dtype=self.dtype)
        self.encoders = nn.ModuleList([RoFormerEncoder(feature_dim=self._emb_dim, attn_heads=8, bias=True,
                                                       ffn_layers=2, ffn_expansion_factor=2, dropout_rate=.1,
                                                       attn_ffn_layers=1, attn_expansion_factor=2,
                                                       attn_dropout_rate=.1, device=self.device,
                                                       dtype=self.dtype) for _ in range(self._encoder_layers)])
        self.lstm = RecursiveSequential(input_dim=self._emb_dim, output_dim=1, bias=True,
                                        recursive_layers=2, recursive_hidden_dim=self._emb_dim,
                                        ffn_layers=2, ffn_expansion_factor=2, ffn_heads=4, ffn_head_layers=1,
                                        dropout_rate=.1, device=self.device, dtype=self.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embs = self.embedding(ids)
        for encoder in self.encoders:
            embs = encoder(embs)
        logits = self.lstm(embs, self.lstm.initial_state(embs.shape[0]))[0]
        return torch.softmax(logits, dim=-1, dtype=self.dtype)
