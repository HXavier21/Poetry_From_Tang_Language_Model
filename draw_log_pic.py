from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt


class DrawMode(Enum):
    Loss = 0
    Both = 1
    Perplexity = 2


def draw_chart(plt_pointer, x, y, x_label, y_label, color='blue', label=None):
    plt_pointer.plot(x, y, label=label, color=color)
    plt_pointer.xlabel(x_label)
    plt_pointer.ylabel(y_label)


def draw_log_chart(*log_paths: str, draw_mode: DrawMode = DrawMode.Both):
    def plot_data(log_paths, y_label, title, file_name):
        plt.figure(figsize=(10, 5))
        for log_path in log_paths:
            log_type = log_path.split('_')[0]
            color = {'gru': 'red', 'lstm': 'blue'}.get(log_type, 'green')
            log_data = pd.read_csv(log_path)
            draw_chart(plt, log_data['Epoch'], log_data[y_label], 'Epoch', y_label, color,
                       f'{log_type.upper()} {y_label}')
        plt.title(title)
        plt.legend()
        plt.savefig(file_name)

    if draw_mode.value < 2:
        plot_data(log_paths, 'Loss', 'Training Loss over Epochs', 'training_loss.png')

    if draw_mode.value > 0:
        plot_data(log_paths, 'Perplexity', 'Perplexity over Epochs', 'perplexity.png')
