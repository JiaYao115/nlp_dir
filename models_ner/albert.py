import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def batch_size__init__(self):
        module_name = 'ALBert'
        save_path = f'../save/{module_name}.pkl'

        batch_size = 128
        embed = 768
        layer_num = 6
        padding_idx = 0
        max_seq_length = 200
        vocab_size = 0
        pass


class Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size,
                                      config.embed,
                                      config.padding_idx)
        self.encoder = Encoder()

    def forward(self, input):
        return


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return


