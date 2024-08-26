import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, network_type):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.network_type = network_type
        match network_type:
            case 'gru':
                self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
            case 'lstm':
                self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            case _:
                self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h


def configure_model(network_type, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
    model = CharRNN(vocab_size, embed_size, hidden_size, num_layers, network_type)
    return model
