import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(model_names, datasets):
    plt.figure(figsize=(10, 6))

    os.makedirs('plots', exist_ok=True)

    for model in model_names:
        for dataset in datasets:
            if dataset == 'dblp' and model == 'ggrgcn':
                 df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_64_{dataset}.csv')
                 plt.plot(df['de'], label=f'{model}_{dataset}')
            else:
                df = pd.read_csv(f'results/oversmooth_adj_{model}_layers_128_{dataset}.csv')
                plt.plot(df['de'], label=f'{model}_{dataset}')

    plt.xlabel('Layer')
    plt.ylabel('DE')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Oversmooth Evaluation - RGCN base')
    plt.legend()
    plt.savefig('plots/oversmooth_adj_comparison_1hop.svg')
    plt.show()


def plot_results_non_lin(model_names, datasets):
    plt.figure(figsize=(10, 6))

    os.makedirs('plots', exist_ok=True)

    for model in model_names:
        for dataset in datasets:
            if dataset == 'dblp' and model == 'ggrgcn':
                 df = pd.read_csv(f'results/no_linear/oversmooth_adj_{model}_layers_64_{dataset}.csv')
                 plt.plot(df['de'], label=f'{model}_{dataset}')
            else:
                df = pd.read_csv(f'results/no_linear/oversmooth_adj_{model}_layers_128_{dataset}.csv')
                plt.plot(df['de'], label=f'{model}_{dataset}')

    plt.xlabel('Layer')
    plt.ylabel('DE')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Oversmooth Evaluation - No linear projection')
    plt.legend()
    plt.savefig('plots/oversmooth_adj_comparison_1hop_non_linear.svg')
    plt.show()

def plot_results_weights(model_names, datasets):
    plt.figure(figsize=(10, 6))

    os.makedirs('plots', exist_ok=True)

    for model in model_names:
        for dataset in datasets:
            if dataset == 'dblp' and model == 'ggrgcn':
                 df = pd.read_csv(f'results/weight_exp_results/oversmooth_adj_{model}_layers_64_{dataset}.csv')
                 plt.plot(df['de'], label=f'{model}_{dataset}')
            else:
                df = pd.read_csv(f'results/weight_exp_results/oversmooth_adj_{model}_layers_128_{dataset}.csv')
                plt.plot(df['de'], label=f'{model}_{dataset}')

    plt.xlabel('Layer')
    plt.ylabel('DE')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Oversmooth Evaluation - Uniform linear weight initialization')
    plt.legend()
    plt.savefig('plots/oversmooth_adj_comparison_1hop_weights.svg')
    plt.show()

plot_results(model_names= ['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'], datasets=['imdb', 'dblp'])
plot_results_non_lin(model_names= ['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'], datasets=['imdb', 'dblp'])
plot_results_weights(model_names= ['pnrgcn', 'rgcn', 'resrgcn', 'ggrgcn'], datasets=['imdb', 'dblp'])