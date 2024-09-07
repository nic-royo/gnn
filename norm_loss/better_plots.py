import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})
color_palette = sns.color_palette("deep", 4)


model_colors = {
    'rgcn': color_palette[0],
    'resrgcn':color_palette[2],
    'pnrgcn': color_palette[1],
    'ggrgcn': color_palette[3]
}

dataset_styles = {
    'imdb': '-',
    'dblp': '--'
}

f1_styles = {
    'train': ':',
    'val': '^'
}

def plot_grad_norm(model_names, datasets, layers, max_value=1e10, exploding_models=['rgcn', 'resrgcn']):
    os.makedirs('plots', exist_ok=True)
    for layer in layers:
        plt.figure(figsize=(12, 7))
        for model in model_names:
            for dataset in datasets:
                filename = f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv'
                df = pd.read_csv(filename)

                if model in exploding_models:
                    df['grad_norm'] = df['grad_norm'].clip(upper=max_value, lower=0)

                label = f'{model}_{dataset}'
                plt.plot(df['grad_norm'], label=label, color=model_colors[model], linestyle=dataset_styles[dataset])

        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('Gradient Norm - L2', fontsize=16)
        plt.title(f'Gradient Norm - {", ".join(model_names)} - Layer {layer}', fontsize=18)
        plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.25, 1))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'plots/gradient_{"_".join(model_names)}_layer_{layer}.svg', bbox_inches='tight')
        plt.close()

def plot_f1_scores(model_names, datasets, layers):
    os.makedirs('plots', exist_ok=True)
    for layer in layers:
        plt.figure(figsize=(12, 7))
        for model in model_names:
            for dataset in datasets:
                filename = f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv'
                df = pd.read_csv(filename)
                plt.plot(df['epoch'], df['train_f1'],
                         label=f'{model}_{dataset}_train_f1',
                         color=model_colors[model],
                         linestyle=f1_styles['train'],
                         marker='o', markevery=5, markersize=4)
                plt.plot(df['epoch'], df['val_f1'],
                         label=f'{model}_{dataset}_val_f1',
                         color=model_colors[model],
                         linestyle=f1_styles['val'],
                         marker='s', markevery=5, markersize=4)
        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('F1 Score', fontsize=16)
        plt.title(f'F1 Scores - {", ".join(model_names)} - Layer {layer}', fontsize=18)
        plt.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'plots/f1_scores_{"_".join(model_names)}_layer_{layer}.svg', bbox_inches='tight')
        plt.close()

datasets = ['imdb', 'dblp']
layers = [4, 8, 16, 32]
plot_grad_norm(['pnrgcn', 'ggrgcn'], datasets, layers)
plot_f1_scores(['pnrgcn', 'ggrgcn'], datasets, layers)
plot_grad_norm(['rgcn', 'resrgcn'], datasets, layers, max_value=30000)
plot_f1_scores(['rgcn', 'resrgcn'], datasets, layers)