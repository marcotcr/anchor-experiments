from __future__ import print_function
import argparse
import collections
import sys
import os
import pickle
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa
import seaborn  # noqa


def load_pickles(datasets, models, out_folder):
    pickles = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0.))
    for d in datasets:
        for m in models:
            path = os.path.join(out_folder, '%s-%s.pickle' % (d, m))
            pickles[d][m] = pickle.load(open(path))
    return pickles


def make_graph(pickles, dataset, model, save=None):
    lime_sub = pickles[dataset][model]['lime_pred_submodular'][2]
    anchor_sub = pickles[dataset][model]['anchor_submodular'][2]
    ks = range(1, len(lime_sub) + 1)
    fig = plt.figure()
    plt.ylabel('Coverage (%)')
    plt.xlabel('# explanations')
    plt.ylim(0, 100)
    seaborn.set(font_scale=2.9)
    seaborn.set_style('white')

    plt.plot(ks, np.array(anchor_sub) * 100, 'o-', lw=4,
             markersize=10, label='SP-Anchor')
    plt.plot(ks, np.array(lime_sub) * 100, 's--', lw=4,
             markersize=10, label='SP-LIME')
    legend_fontsize = 25

    if 'anchor_random' in pickles[dataset][model]:
        anchor_random = pickles[dataset][model]['anchor_random']
        plt.errorbar(
            ks, np.array(anchor_random[1]) * 100,
            yerr=np.array(anchor_random[3])*100, fmt='o-',
            lw=4, markersize=10, label='RP-Anchor')

    if 'lime_pred_random' in pickles[dataset][model]:
        lime_random = pickles[dataset][model]['lime_pred_random']
        plt.errorbar(
            ks, np.array(lime_random[1]) * 100,
            yerr=np.array(lime_random[3])*100, fmt='o-',
            lw=4, markersize=10, label='RP-LIME')

    lgd = plt.legend(loc='upper center', fontsize=legend_fontsize), # noqa
                    #  bbox_to_anchor=(1, 1))
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
    return fig


def make_table(pickles, datasets, models):
    tab =  '%Precision table\n'
    explanations = ['anchor', 'lime_naive']
    tab += '\\begin{table}[h!]\n\\small\n'
    tab += '\\begin{tabular}{|c|c|%s}\n' % ('c' * len(explanations))
    tab += ' & & %s \\\\\n' % ' & '.join(
        [x.replace('_', '\\_') for x in explanations])
    for d in datasets:
        tab += '\multirow{%d}{*}{\\rotatebox[origin=c]{90}{%s \\hspace{-\\normalbaselineskip}}} ' % (len(models), d) # noqa
        for m in models:
            tab += ' & %s &' % m
            tab += ' & '.join(['%.1f +- %.1f' % (pickles[d][m][e +'_1'][0] * 100, pickles[d][m][e +'_1'][2] * 100) for e in explanations]) # noqa
            tab += ' \\\\\n'
        tab += '\\hline \n'

    tab += '\n\\end{tabular}\n\\caption{Precision}\\end{table}'
    tab += '\n\n'
    tab += '%Coverage Table\n'
    explanations = ['anchor', 'lime_pred']
    tab += '\\begin{table}[h!]\n\\small\n'
    tab += '\\begin{tabular}{|c|c|%s}\n' % ('c' * len(explanations))
    tab += ' & & %s \\\\\n' % ' & '.join(
        [x.replace('_', '\\_') for x in explanations])
    for d in datasets:
        tab += '\multirow{%d}{*}{\\rotatebox[origin=c]{90}{%s}} ' % (
            len(models), d)
        for m in models:
            tab += ' & %s &' % m
            tab += ' & '.join(['%.1f +- %.1f' % (pickles[d][m][e +'_1'][1] * 100, pickles[d][m][e +'_1'][3] * 100) for e in explanations]) # noqa
            tab += ' \\\\\n'
        tab += '\\hline \n'

    tab += '\n\\end{tabular}\n\\caption{Coverage}\\end{table}'
    return tab

def main():
    parser = argparse.ArgumentParser(description='Graphs')
    parser.add_argument(
        '-r', dest='results_folder',
        default='./results') # noqa
    parser.add_argument(
        '-g', dest='graphs_folder',
        default='./graphs')

    args = parser.parse_args()
    datasets = ['adult', 'recidivism', 'lending']
    models = ['logistic', 'xgboost', 'nn']
    pickles = load_pickles(datasets, models, args.results_folder)
    print('')
    tab = make_table(pickles, datasets, models)
    print('Table:')
    print(tab)
    for dataset in datasets:
        for model in models:
            path = os.path.join(args.graphs_folder, '%s-%s.png' %
                                (dataset, model))
            make_graph(pickles, dataset, model, save=path)


if __name__ == '__main__':
    main()
