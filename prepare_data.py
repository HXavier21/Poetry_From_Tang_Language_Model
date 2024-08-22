import torch
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = list(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        x = [self.char_to_idx[ch] for ch in self.text[idx:idx + self.seq_length]]
        y = self.char_to_idx[self.text[idx + self.seq_length]]
        return torch.tensor(x), torch.tensor(y)


def prepare_data(data_path, seq_length=100):
    with open(data_path, 'r') as f:
        text = f.read().lower()

    dataset = CharDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataset, dataloader
