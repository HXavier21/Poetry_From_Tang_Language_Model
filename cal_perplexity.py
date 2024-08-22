import torch


def cal_perplexity(model, dataloader, criterion, device=torch.device("cuda")):
    model.to(device)
    model.eval()  # 切换到评估模式
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
    return perplexity
