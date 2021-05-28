from matplotlib import pyplot as plt
import numpy as np

"""
Functions starting with _debug were only added for debugging during development and should not be executed in the
final submission.
"""


def _plot_tag_similarity_matrix(matrix: np.ndarray, tags: list, LIMIT:int or None =8):
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


def _filter_tag_pairs_by_similarity(pairwise_similarities: np.ndarray, all_tags: np.ndarray, min_similarity: float):
    x = np.argwhere(pairwise_similarities >= min_similarity)
    pairs = []

    for pair in x:
        if (all_tags[pair[0]], all_tags[pair[1]]) not in pairs and (all_tags[pair[1]], all_tags[pair[0]]) not in pairs:
            pairs.append((all_tags[pair[0]], all_tags[pair[1]]))

    for pair in pairs:
        print(f'{pair[0]} & {pair[1]} ')


def _print_stats(processed_data):
    pass