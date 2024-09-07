import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_grad_norm(model_names, datasets, layers, max_value=1e10):
    os.makedirs('plots', exist_ok=True)
    linestyles = ['-', '--', '-.', ':']
    for layer in layers:
        for dataset in datasets:
            plt.figure(figsize=(10, 6))
            for i, model in enumerate(model_names):
                if dataset == 'dblp':
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')
                else:
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')
                df['grad_norm'] = df['grad_norm'].clip(upper=50000, lower=0)
                label = f'{model}'
                plt.plot(df['grad_norm'], label=label, linestyle=linestyles[i % len(linestyles)])
            plt.xlabel('epochs')
            plt.ylabel('Gradient Norm - L2')
            plt.title(f'Gradient Norm - Layer {layer} - Dataset {dataset}')
            plt.legend(loc='upper right')
            plt.savefig(f'plots/gradient_layer_{layer}_{dataset}.svg')
            plt.close()

def plot_loss_and_f1(model_names, datasets, layers, max_value=1e10):
    os.makedirs('plots', exist_ok=True)
    linestyles = ['-', '--', '-.', ':']
    for layer in layers:
        for dataset in datasets:
            plt.figure(figsize=(10, 6))
            for i, model in enumerate(model_names):
                if dataset == 'dblp':
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')
                else:
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')

                df['train_loss'].replace([np.inf, -np.inf], max_value, inplace=True)
                df['val_loss'].replace([np.inf, -np.inf], max_value, inplace=True)
                plt.plot(df['epoch'], df['train_loss'], label=f'{model}_train_loss', linestyle=linestyles[i % len(linestyles)])
                plt.plot(df['epoch'], df['val_loss'], label=f'{model}_val_loss', linestyle=linestyles[i % len(linestyles)], alpha=0.7)
            plt.xlabel('epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss - Layer {layer} - Dataset {dataset}')
            plt.legend(loc='upper right')
            plt.savefig(f'plots/loss_layer_{layer}_{dataset}.svg')
            plt.close()

def plot_accuracy(model_names, datasets, layers):
    os.makedirs('plots', exist_ok=True)
    linestyles = ['-', '--', '-.', ':']
    for layer in layers:
        for dataset in datasets:
            plt.figure(figsize=(10, 6))
            for i, model in enumerate(model_names):
                if dataset == 'dblp':
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')
                else:
                    df = pd.read_csv(f'plot_data/training_metrics_layer_{layer}_{model}_{dataset}_short.csv')
                plt.plot(df['epoch'], df['train_acc'], label=f'{model}_train_acc', linestyle=linestyles[i % len(linestyles)])
                plt.plot(df['epoch'], df['val_acc'], label=f'{model}_val_acc', linestyle=linestyles[i % len(linestyles)], alpha=0.7)
            plt.xlabel('epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy - Layer {layer} - Dataset {dataset}')
            plt.legend(loc='lower right')
            plt.savefig(f'plots/accuracy_layer_{layer}_{dataset}.svg')
            plt.close()

plot_grad_norm(model_names=['rgcn', 'resrgcn'], datasets=['imdb', 'dblp'], layers=[4, 8, 16, 32])
plot_loss_and_f1(model_names=['rgcn', 'resrgcn'], datasets=['imdb', 'dblp'], layers=[4, 8, 16, 32])
plot_accuracy(model_names=['rgcn', 'resrgcn'], datasets=['imdb', 'dblp'], layers=[4, 8, 16, 32])