import torch.nn as nn
from prepare_data import dataset


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h


# 模型参数
vocab_size = dataset.vocab_size
embed_size = 128
hidden_size = 256
num_layers = 2

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers)
