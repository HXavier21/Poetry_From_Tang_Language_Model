import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def train_model(
        model, dataloader, num_epochs, device=torch.device("cuda"),
        checkpoint_path='char_rnn_checkpoint.pth',
        first_training=False
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 0
    loss = 0

    if not first_training:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'Loaded checkpoint from epoch {epoch} with loss {loss}')
        except:
            print("No checkpoint found")

    model.to(device)

    for epoch in range(epoch, num_epochs):
        model.train()
        h = None
        pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if h is not None and h.size(1) != x.size(0):
                h = None
            out, h = model(x, h)
            h = h.detach()
            out = out[:, -1, :]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
        pbar.close()

    torch.save({
        'epoch': num_epochs + 1,  # 保存当前epoch数
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  # 保存当前损失值
    }, 'char_rnn_checkpoint.pth')
