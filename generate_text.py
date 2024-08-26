from enum import Enum

import torch


def generate_text(model, start_text: str, char_to_idx, idx_to_char, max_length: int, temperature=1.0,
                  device=torch.device("cuda")):
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


class PoetryType(Enum):
    five_character_quatrain = 24
    seven_character_quatrain = 32
    five_character_regulated_verse = 48
    seven_character_regulated_verse = 64
    unknown = None


def check_characters(text: str, characters: int):
    if characters not in [5, 7]:
        return False
    sentence = text[:characters * 2 + 2]
    if sentence.endswith('。'):
        sentences = sentence.split('，')
        if len(sentences[0]) == characters and len(sentences[1]) == characters + 1:
            return True
    return False


def check_poetry(text: str, poetry_type: PoetryType):
    match poetry_type:
        case PoetryType.five_character_quatrain:
            if check_characters(text[:12], 5) and check_characters(text[12:], 5):
                return True
        case PoetryType.seven_character_quatrain:
            if check_characters(text[:16], 7) and check_characters(text[16:], 7):
                return True
        case PoetryType.five_character_regulated_verse:
            if all(check_characters(text[i:i + 12], 5) for i in [0, 12, 24, 36]):
                return True
        case PoetryType.seven_character_regulated_verse:
            if all(check_characters(text[i:i + 16], 7) for i in [0, 16, 32, 48]):
                return True
        case PoetryType.unknown:
            return True
    return False


def generate_poetry(model, start_text: str, char_to_idx, idx_to_char, max_length: int, poetry_type: PoetryType,
                    random_generation=False, temperature=1.0, device=torch.device("cuda")):
    """
        使用训练好的模型生成文本。
        :param model: 训练好的CharRNN模型
        :param start_text: 初始输入文本（字符串）
        :param char_to_idx: 字符到索引的映射
        :param idx_to_char: 索引到字符的映射
        :param max_length: 生成文本的最大长度
        :param poetry_type: 诗歌类型
        :param random_generation: 是否随机生成（默认是False）
        :param temperature: 控制生成的多样性，值越高生成的文本越随机，值越低生成的文本越确定
        :param device: 运行设备（默认是"cuda"）
        :return: 生成的文本（字符串）
        """
    while True:
        if random_generation:
            start_text = idx_to_char[torch.randint(0, len(idx_to_char), (1,)).item()]
        generated_text = generate_text(model, start_text, char_to_idx, idx_to_char, max_length, temperature, device)
        if poetry_type == PoetryType.unknown:
            return generated_text
        generated_text = generated_text.replace('\n', '')[:poetry_type.value]
        if check_poetry(generated_text, poetry_type):
            return generated_text


if __name__ == '__main__':
    print(check_poetry('春眠不觉晓，处处闻啼鸟。\n夜来风雨声，花落知多少。', PoetryType.five_character_quatrain))
    print(check_poetry(
        '陆机二十作文赋，汝更小年能缀文。总角草书又神速，世上儿子徒纷纷。骅骝作驹已汗血，鸷鸟举翮连青云。词源倒流三峡水，笔阵独扫千人军。',
        PoetryType.seven_character_regulated_verse))
