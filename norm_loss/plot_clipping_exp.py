import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

def plot_accuracy(model_names, datasets):
    os.makedirs('plots', exist_ok=True)

    linestyles = ['-', '--', '-.', ':']
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        df = pd.read_csv(f'plot_data/clipping_experiments/training_metrics_layer_64_{model}_dblp_gradnorm_1.0.csv')
        plt.plot(df['epoch'], df['train_f1'], label=f'{model}_train_acc', linestyle=linestyles[i % len(linestyles)])
        plt.plot(df['epoch'], df['val_f1'], label=f'{model}_val_acc', linestyle=linestyles[i % len(linestyles)], alpha=0.7)
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('F1 Score', fontsize=18)
    plt.title(f'F1 Scores - {", ".join(model_names)} - DBLP - Layer 64', fontsize=20)
    plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'plots/f1_layer_64_dblp_clipping.svg')
    plt.close()
plot_accuracy(model_names=['rgcn', 'resrgcn'], datasets=['dblp'])