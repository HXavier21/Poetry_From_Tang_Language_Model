import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import csv


def train_model(
        model, dataloader, num_epochs, device=torch.device("cuda"),
        checkpoint_path='gru_char_rnn_checkpoint.pth',
        enable_logging=True, log_path='gru_log.csv', cal_perplexity=None
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 0

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            criterion = checkpoint['criterion']
            dataloader = checkpoint['dataloader']
            print(f'Loaded checkpoint from epoch {epoch} with loss {criterion}')
        except Exception as e:
            print(f"No checkpoint found: {e}")

    for epoch in range(epoch, num_epochs):
        model.train()
        h = None
        loss = 0
        pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if h is not None and h[0].size(1) != x.size(0):
                h = None
            out, h = model(x, h)
            if isinstance(h, tuple):
                h = (h[0].detach(), h[1].detach())  # 分别对隐藏状态和细胞状态调用 detach()
            else:
                h = h.detach()
            out = out[:, -1, :]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
        pbar.close()
        if enable_logging:
            if not os.path.exists(log_path):
                with open(log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'Loss', 'Perplexity'])
            perplexity = cal_perplexity(model, dataloader, criterion, device)
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, loss.item(), perplexity])

    torch.save({
        'epoch': num_epochs,  # 保存当前epoch数
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion,  # 保存当前损失函数
        'dataloader': dataloader
    }, model.network_type + '_char_rnn_checkpoint.pth')
