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
    quatrain = 4
    regulated_verse = 8
    unknown = None


def check_characters(text: str, characters: int):
    if characters not in [5, 7]:
        return False
    sentence = text[:characters * 2 + 2]
    if sentence.endswith('。'):
        sentences = sentence.split('，')
        if len(sentences) > 1 and len(sentences[0]) == characters and len(sentences[1]) == characters + 1:
            return True
    return False


def check_poetry(text: str, characters: int, poetry_type: PoetryType):
    def check_lines(text, characters, num_lines):
        return all(check_characters(text[i * (characters * 2 + 2):(i + 1) * (characters * 2 + 2)], characters)
                   for i in range(num_lines))

    match poetry_type:
        case PoetryType.quatrain:
            return check_lines(text, characters, 2)
        case PoetryType.regulated_verse:
            return check_lines(text, characters, 4)
        case PoetryType.unknown:
            return True
    return False


def generate_poetry(model, start_text: str, char_to_idx, idx_to_char,
                    characters: int, poetry_type: PoetryType,
                    enable_single_sentence=False, enable_acrostic=False, enable_random_generation=False,
                    max_length=100, max_attempts=500, temperature=1.0, device=torch.device("cuda")):
    """
        使用训练好的模型生成文本。
        :param model: 训练好的CharRNN模型
        :param start_text: 初始输入文本（字符串）
        :param char_to_idx: 字符到索引的映射
        :param idx_to_char: 索引到字符的映射
        :param characters: 诗歌的字符数（5或7）
        :param poetry_type: 诗歌类型
        :param enable_single_sentence: 是否生成单句诗（默认是False）
        :param enable_acrostic: 是否启用藏头诗（默认是False,启用时输入文本即为藏头）
        :param enable_random_generation: 是否随机生成（默认是False,启用时输入文本失效）
        :param max_length: 生成文本的最大长度（默认是100）
        :param max_attempts: 最大尝试次数（默认是500）
        :param temperature: 控制生成的多样性，值越高生成的文本越随机，值越低生成的文本越确定
        :param device: 运行设备（默认是"cuda"）
        :return: 生成的文本（字符串）
        """
    if enable_acrostic:
        generated_text = ""
        for char in start_text:
            generated_text += generate_poetry(model, char, char_to_idx, idx_to_char, characters,
                                              poetry_type, enable_single_sentence=True)
        return generated_text.replace('。', '。\n')
    for _ in range(max_attempts):
        if enable_random_generation:
            start_text = idx_to_char[torch.randint(0, len(idx_to_char), (1,)).item()]
        generated_text = generate_text(model, start_text, char_to_idx, idx_to_char, max_length, temperature, device)
        if enable_single_sentence:
            generated_text = generated_text.replace('\n', '').split('。')[0] + '。'
            if poetry_type == PoetryType.unknown:
                return generated_text
            if check_characters(generated_text, characters):
                return generated_text
        else:
            if poetry_type == PoetryType.unknown:
                return generated_text
            generated_text = generated_text.replace('\n', '')[:(characters + 1) * poetry_type.value]
            if check_poetry(generated_text, characters, poetry_type):
                return generated_text.replace('。', '。\n')
    return generate_poetry(model, start_text, char_to_idx, idx_to_char, characters, poetry_type.unknown,
                           enable_single_sentence)


if __name__ == '__main__':
    print(check_poetry('春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。', 5, PoetryType.quatrain))
    print(check_poetry(
        '陆机二十作文赋，汝更小年能缀文。总角草书又神速，世上儿子徒纷纷。骅骝作驹已汗血，鸷鸟举翮连青云。词源倒流三峡水，笔阵独扫千人军。',
        7,
        PoetryType.regulated_verse))
