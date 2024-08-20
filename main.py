import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from generate_text import generate_text


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


# 读取文本数据
with open('poetryFromTang.txt', 'r') as f:
    text = f.read().lower()

# 创建数据集和数据加载器
seq_length = 100
dataset = CharDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
last_loss = nn.CrossEntropyLoss()

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    h = None
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # 如果h存在且batch_size不匹配，则重置隐藏状态
        if h is not None and h.size(1) != x.size(0):
            h = None

        out, h = model(x, h)
        # out = out.view(-1, vocab_size)
        # y = y.view(-1)

        # Detach hidden state to prevent backpropagating through entire history
        h = h.detach()
        # Take the output only at the last time step
        out = out[:, -1, :]

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        # h = h.detach()

        pbar.set_postfix({'Loss': loss.item()})
        last_loss = loss

    pbar.close()

# 保存模型参数和优化器状态
torch.save({
    'epoch': num_epochs,  # 保存当前epoch数
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': last_loss,  # 保存当前损失值
}, 'char_rnn_checkpoint.pth')

torch.save(model, 'char_rnn_model_complete.pth')

# 加载模型和优化器状态
checkpoint = torch.load('char_rnn_checkpoint.pth')
model = CharRNN(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.to(device)
model.eval()  # 切换到评估模式

start_text = "君不见黄河之水天上来"  # 你想要补全的文本片段
generated_text = generate_text(model, start_text, dataset.char_to_idx, dataset.idx_to_char,
                               max_length=500,
                               temperature=1.0)
print(generated_text)

total_loss = 0
total_chars = 0

with torch.no_grad():
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x)
        # Take the output only at the last time step
        out = out[:, -1, :]
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        total_chars += len(y)

avg_loss = total_loss / total_chars
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f'Perplexity: {perplexity:.4f}')

model = torch.load('char_rnn_model_complete.pth')

model.to(device)
model.eval()  # 切换到评估模式

start_text = "人生得意须尽欢"  # 你想要补全的文本片段
generated_text = generate_text(model, start_text, dataset.char_to_idx, dataset.idx_to_char,
                               max_length=500,
                               temperature=1.0)
print(generated_text)

total_loss = 0
total_chars = 0

with torch.no_grad():
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x)
        # Take the output only at the last time step
        out = out[:, -1, :]
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        total_chars += len(y)

avg_loss = total_loss / total_chars
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f'Perplexity: {perplexity:.4f}')


