from matplotlib import pyplot as plt
import numpy as np

"""
Functions starting with _debug were only added for debugging during development and should not be executed in the
final submission.
"""


def _plot_matrix(matrix: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)

    plt.show()


def _filter_tag_pairs_by_similarity(pairwise_similarities: np.ndarray, all_tags: np.ndarray, min_similarity: float):
    x = np.argwhere(pairwise_similarities >= min_similarity)
    pairs = []

    for pair in x:
        if (all_tags[pair[0]], all_tags[pair[1]]) not in pairs and (all_tags[pair[1]], all_tags[pair[0]]) not in pairs:
            pairs.append((all_tags[pair[0]], all_tags[pair[1]]))

    for pair in pairs:
        print(f'{pair[0]} & {pair[1]} ')
