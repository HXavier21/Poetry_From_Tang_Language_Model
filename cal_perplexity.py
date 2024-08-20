from torch import nn
from prepare_data import dataloader
import torch
from configure_model import CharRNN
# from train_model import criterion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型的完整状态
model = torch.load('char_rnn_model_complete.pth').to(device)
model.eval()  # 切换到评估模式

criterion = nn.CrossEntropyLoss()

total_loss = 0
total_chars = 0

with torch.no_grad():
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x)
        out = out[:, -1, :]
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        total_chars += len(y)

avg_loss = total_loss / total_chars
perplexity = torch.exp(torch.tensor(avg_loss)).item()
print(f'Perplexity: {perplexity:.4f}')
