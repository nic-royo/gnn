import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

color_palette = sns.color_palette("deep", 4)
line_styles = ['-', '--']

def plot_results(model_names, datasets, title, output_filename):
    plt.figure(figsize=(8, 6))
    os.makedirs('plots', exist_ok=True)

    for i, model in enumerate(model_names):
        for j, dataset in enumerate(datasets):
            if dataset == 'dblp' and model == 'ggrgcn':
                df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_64_{dataset}.csv')
            else:
                df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_128_{dataset}.csv')

            plt.plot((df['de']**2), label=f'{model}_{dataset}',
                     color=color_palette[i], linestyle=line_styles[j])

    plt.xlabel('Layer', fontsize=18)
    plt.ylabel('DE', fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{output_filename}.svg')
    plt.show()

def plot_results(model_names, datasets, title, output_filename):
    plt.figure(figsize=(8, 6))
    os.makedirs('plots', exist_ok=True)

    for i, model in enumerate(model_names):
        for j, dataset in enumerate(datasets):
            if dataset == 'dblp' and model == 'ggrgcn':
                df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_64_{dataset}.csv')
            else:
                df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_128_{dataset}.csv')

            plt.plot((df['de']**2), label=f'{model}_{dataset}',
                     color=color_palette[i], linestyle=line_styles[j])

    plt.xlabel('Layer', fontsize=18)
    plt.ylabel('DE', fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{output_filename}.svg')
    plt.show()

def plot_results_weight(model_names, datasets, title, output_filename):
    plt.figure(figsize=(8, 6))
    os.makedirs('plots', exist_ok=True)

    for i, model in enumerate(model_names):
        for j, dataset in enumerate(datasets):
            if dataset == 'dblp' and model == 'ggrgcn':
                df = pd.read_csv(f'results/weight_exp_results/oversmooth_adj_{model}_layers_64_{dataset}.csv')
            else:
                df = pd.read_csv(f'results/weight_exp_results/oversmooth_adj_{model}_layers_128_{dataset}.csv')

            plt.plot((df['de']**2), label=f'{model}_{dataset}',
                     color=color_palette[i], linestyle=line_styles[j])

    plt.xlabel('Layer', fontsize=18)
    plt.ylabel('DE', fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{output_filename}.svg')
    plt.show()

def plot_results_no_linear(model_names, datasets, title, output_filename):
    plt.figure(figsize=(8, 6))
    os.makedirs('plots', exist_ok=True)

    for i, model in enumerate(model_names):
        for j, dataset in enumerate(datasets):
            if dataset == 'dblp' and model == 'ggrgcn':
                df = pd.read_csv(f'results/no_linear/oversmooth_adj_{model}_layers_64_{dataset}.csv')
            else:
                df = pd.read_csv(f'results/no_linear/oversmooth_adj_{model}_layers_128_{dataset}.csv')

            plt.plot((df['de']**2), label=f'{model}_{dataset}',
                     color=color_palette[i], linestyle=line_styles[j])

    plt.xlabel('Layer', fontsize=18)
    plt.ylabel('DE', fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{output_filename}.svg')
    plt.show()

plot_results(
    model_names=['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'],
    datasets=['imdb', 'dblp'],
    title='Oversmooth Evaluation',
    output_filename='oversmooth_adj_comparison_1hop'
)

plot_results_no_linear(
    model_names=['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'],
    datasets=['imdb', 'dblp'],
    title='No linear projection',
    output_filename='oversmooth_adj_comparison_1hop_non_linear'
)

plot_results_weight(
    model_names=['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'],
    datasets=['imdb', 'dblp'],
    title='Uniform linear weight initialization',
    output_filename='oversmooth_adj_comparison_1hop_weights'
)