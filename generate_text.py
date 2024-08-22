import torch

# import configure_model
# from prepare_data import CharDataset


def generate_text(model, start_text, char_to_idx, idx_to_char, max_length, temperature=1.0,
                  device=torch.device("cuda")):
    """
    使用训练好的模型生成文本。
    :param model: 训练好的CharRNN模型
    :param start_text: 初始输入文本（字符串）
    :param char_to_idx: 字符到索引的映射
    :param idx_to_char: 索引到字符的映射
    :param max_length: 生成文本的最大长度
    :param temperature: 控制生成的多样性，值越高生成的文本越随机，值越低生成的文本越确定
    :param device: 运行设备（默认是"cuda"）
    :return: 生成的文本（字符串）
    """
    model = model.to(device)
    model.eval()  # 切换到评估模式
    h = None  # 初始化隐藏状态
    generated_text = start_text

    # 将输入文本转换为索引，并将其移动到设备上
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            # 前向传播，生成预测分布
            output, h = model(input_seq, h)

            # 取最后一个时间步的输出，并通过temperature调整
            output = output[:, -1, :] / temperature
            probabilities = torch.softmax(output, dim=-1).squeeze()

            # 根据概率分布采样下一个字符的索引
            next_char_idx = torch.multinomial(probabilities, 1).item()

            # 将生成的字符索引转回字符，并添加到生成文本中
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char

            # 准备下一步的输入
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    return generated_text


# if __name__ == '__main__':
#     # 加载模型和优化器状态
#     checkpoint = torch.load('char_rnn_checkpoint.pth')
#     dataset = checkpoint['dataset']
#     vocab_size = dataset.vocab_size
#     embed_size = 128
#     hidden_size = 256
#     num_layers = 2
#     model = configure_model.CharRNN(vocab_size, embed_size, hidden_size,
#                                     num_layers)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     # epoch = checkpoint['epoch']
#     # loss = checkpoint['loss']
#
#     model.to(torch.device("cuda"))
#     model.eval()  # 切换到评估模式
#
#     start_text = "王一鸣"  # 你想要补全的文本片段
#     for char in start_text:
#         generated_text = generate_text(model, char, dataset.char_to_idx, dataset.idx_to_char,
#                                        max_length=10,
#                                        temperature=1.0)
#         print(generated_text)
