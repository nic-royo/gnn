import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

dataset_styles = {
    'imdb': '-',
    'dblp': '--'
}

f1_styles = {
    'train': 'o',
    'val': 's'
}

def plot_grad_norm(model_names, datasets, layers, color_palette, max_value=1e10, exploding_models=['rgcn', 'resrgcn']):
    os.makedirs('plots', exist_ok=True)
    model_colors = {model: color_palette[i] for i, model in enumerate(model_names)}
    for layer in layers:
        plt.figure(figsize=(8, 6))
        for model in model_names:
            for dataset in datasets:
                filename = f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv'
                df = pd.read_csv(filename)

                if model in exploding_models:
                    df['grad_norm'] = df['grad_norm'].clip(upper=max_value, lower=0)

                label = f'{model}_{dataset}'
                plt.plot(df['grad_norm'], label=label, color=model_colors[model], linestyle=dataset_styles[dataset])

        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('Gradient Norm - L2', fontsize=18)
        plt.title(f'Gradient Norm - {", ".join(model_names)} - Layer {layer}', fontsize=20)
        plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'plots/gradient_{"_".join(model_names)}_layer_{layer}.svg', bbox_inches='tight')
        plt.close()

def plot_f1_scores(model_names, datasets, layers, color_palette):
    os.makedirs('plots', exist_ok=True)
    model_colors = {model: color_palette[i] for i, model in enumerate(model_names)}
    for dataset in datasets:
        for layer in layers:
            plt.figure(figsize=(8, 6))
            for model in model_names:
                filename = f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv'
                df = pd.read_csv(filename)
                plt.plot(df['epoch'], df['train_f1'],
                         label=f'{model}_train_f1',
                         color=model_colors[model],
                         linestyle='-',
                         marker=f1_styles['train'], markevery=5, markersize=4)
                plt.plot(df['epoch'], df['val_f1'],
                         label=f'{model}_val_f1',
                         color=model_colors[model],
                         linestyle='--',
                         marker=f1_styles['val'], markevery=5, markersize=4)
            plt.xlabel('epochs', fontsize=18)
            plt.ylabel('F1 Score', fontsize=18)
            plt.title(f'F1 Scores - {", ".join(model_names)} - {dataset.upper()} - Layer {layer}', fontsize=20)
            plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(f'plots/f1_scores_{dataset}_{"_".join(model_names)}_layer_{layer}.svg', bbox_inches='tight')
            plt.close()


datasets = ['imdb', 'dblp']
layers = [4, 8, 16, 32]

color_palette_1 = sns.color_palette("husl", 2)
color_palette_2 = sns.color_palette("Set2", 2)


plot_grad_norm(['pnrgcn', 'ggrgcn'], datasets, layers, color_palette=color_palette_1)
plot_f1_scores(['pnrgcn', 'ggrgcn'], datasets, layers, color_palette=color_palette_1)

plot_grad_norm(['rgcn', 'resrgcn'], datasets, layers, color_palette=color_palette_2, max_value=30000)
plot_f1_scores(['rgcn', 'resrgcn'], datasets, layers, color_palette=color_palette_2)
