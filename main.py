import torch

import prepare_data
import configure_model
import train_model
import cal_perplexity
import generate_text
import draw_log_pic
from prepare_data import CharDataset

if __name__ == '__main__':
    mode = input('请选择功能（0-训练模型，1-计算困惑度，2-生成诗歌，3-绘制损失曲线）：')
    network_type = input('请选择RNN类型（gru/lstm/rnn）：')
    for case in mode:
        match case:
            case '0':
                dataset, dataloader = prepare_data.prepare_data('poetryFromTang.txt', 100)
                model = configure_model.configure_model(network_type, dataloader.dataset.vocab_size)
                num_epochs = int(input('请输入训练的目标轮数：'))
                train_model.train_model(model=model, dataloader=dataloader, num_epochs=num_epochs,
                                        checkpoint_path=network_type + '_char_rnn_checkpoint.pth',
                                        enable_logging=True, log_path=network_type + '_log.csv',
                                        cal_perplexity=cal_perplexity.cal_perplexity)
            case '1':
                checkpoint = torch.load(network_type + '_char_rnn_checkpoint.pth')
                dataloader = checkpoint['dataloader']
                model = configure_model.configure_model(network_type, dataloader.dataset.vocab_size)
                model.load_state_dict(checkpoint['model_state_dict'])
                criterion = checkpoint['criterion']
                perplexity = cal_perplexity.cal_perplexity(model, dataloader, criterion)
                print(f'Perplexity: {perplexity}')
            case '2':
                checkpoint = torch.load(network_type + '_char_rnn_checkpoint.pth')
                dataset = checkpoint['dataloader'].dataset
                model = configure_model.configure_model(network_type, dataset.vocab_size)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_text = '春'
                generated_text = generate_text.generate_poetry(
                    model=model, start_text=start_text,
                    char_to_idx=dataset.char_to_idx,
                    idx_to_char=dataset.idx_to_char,
                    characters=5,
                    poetry_type=generate_text.PoetryType.quatrain,
                    max_attempts=2000
                )
                print(generated_text)
            case '3':
                draw_log_pic.draw_log_chart('gru_log.csv', 'lstm_log.csv')
