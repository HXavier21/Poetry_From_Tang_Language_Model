import torch
import torch.nn as nn


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


def configure_model(vocab_size, embed_size=128, hidden_size=256, num_layers=2):
    model = CharRNN(vocab_size, embed_size, hidden_size, num_layers)
    return model


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = configure_model(0)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
