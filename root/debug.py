import sklearn.metrics
from matplotlib import pyplot as plt
from typing import Optional
import numpy as np

"""
Functions starting with _debug were only added for debugging during development and should not be executed in the
final submission.
"""


def _plot_tag_similarity_matrix(matrix: np.ndarray, tags: list, LIMIT:Optional[int]=20):
    if LIMIT and matrix.shape[0] > LIMIT:
        print(f'Exceeded size limit for printing: {matrix.shape[0]} > {LIMIT}')
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(tags.tolist())))
    ax.set_yticks(np.arange(len(tags.tolist())))
    ax.set_xticklabels(tags.tolist())
    ax.set_yticklabels(tags.tolist())
    plt.xticks(rotation=90)
    plt.show()

    print(matrix)


def _plot_scatter(matrix: np.ndarray, tags: list, LIMIT:Optional[int]=20):
    cmaplist = plt.cm.nipy_spectral(np.linspace(0, 1, LIMIT))
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for i in range(len(tags)):
        if i >= LIMIT:
            break
        plt.scatter(matrix[i,0], matrix[i, 1], alpha=1, label=tags[i], color=cmaplist[i])
    plt.legend(bbox_to_anchor=(1.00, 1.15), loc='upper left')
    plt.show()


def _filter_tag_pairs_by_similarity(pairwise_similarities: np.ndarray, all_tags: np.ndarray, min_similarity: float):
    x = np.argwhere(pairwise_similarities >= min_similarity)
    pairs = []

    for pair in x:
        if (all_tags[pair[0]], all_tags[pair[1]]) not in pairs and (all_tags[pair[1]], all_tags[pair[0]]) not in pairs:
            pairs.append((all_tags[pair[0]], all_tags[pair[1]]))

    for pair in pairs:
        print(f'{pair[0]} & {pair[1]} ')


def _similarity_max_histogram(pairwise_similarities: np.ndarray, all_tags):
    values = pairwise_similarities.max(0)


def _print_most_similar(pairwise_similarities: np.ndarray, all_tags, tag_name='Cute Girls Doing Cute Things'):
    id = np.where(all_tags == tag_name)
    similar_id = pairwise_similarities[id].argsort()[0][::-1]
    similar_name = all_tags[similar_id]
    print(similar_name)


def _tsne_parameter_search(sim, edges):
    from sklearn.manifold import TSNE
    from sklearn.utils.fixes import loguniform
    parameters = {'perplexity': np.random.uniform(1, 60, 5),
                  'early_exaggeration': np.random.uniform(1, 60, 5),
                  'learning_rate': np.random.uniform(10, 1000, 5),
                  'SIGMA':np.random.uniform(0, 1, 5),
                  'angle': np.random.uniform(0.00001, 1, 5)
                  }
    d = np.inf
    for i0 in parameters[list(parameters.keys())[0]]:
        for i1 in parameters[list(parameters.keys())[1]]:
            for i2 in parameters[list(parameters.keys())[2]]:
                for i3 in parameters[list(parameters.keys())[3]]:
                    for i4 in parameters[list(parameters.keys())[4]]:
                        tsne = TSNE(n_jobs=6, random_state=1, n_iter=15000, perplexity=i0, early_exaggeration=i1, learning_rate=i2, init="pca", angle=i4)
                        positions: np.ndarray = tsne.fit_transform(_logarithmic_scaling(sim, i3))
                        positions = positions + np.abs(positions.min())
                        positions = positions/positions.max()
                        _ = 0
                        for e in edges:
                            _ += sklearn.metrics.pairwise.euclidean_distances([positions[int(e[0])]], [positions[int(e[1])]])[0,0]**2

                        print('.', end='')
                        if _ < d:
                            print(f'\nnew min:{_} at (perplexity={i0}, early_exaggeration={i1}, learning_rate={i2}, SIGMA="{i3}", angle={i4})')
                            d = _



def _logarithmic_scaling(x, SIGMA):
    return -np.log((1 - SIGMA) * x + SIGMA)

def _print_stats(processed_data):
    pass