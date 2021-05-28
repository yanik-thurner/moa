import numpy as np
import pandas as pd

# TODO: remove for submission
from util import Task
from filter import FilterList
import debug

"""
The minimum ranking (ranging from 0 to 100) to be considered a valid tag for a show.
"""
MIN_TAG_RANKING = 20


"""
Used for logarithmic scaling when calculating the distances.

According to the paper:
[...] sigma is a small, positive, constant scaling value, currently set to
0.1, used to ensure a non-zero value inside the logarithm in the case
that two terms have a pairwise similarity of 0.
"""
SIGMA = 0.1


def preprocess(raw_data: pd.DataFrame) -> (pd.DataFrame, FilterList):
    filters = FilterList()

    # transpose so columns are attributes and rows are anime ids
    df = raw_data.transpose(copy=True)

    # remove id col since it is already the index
    df = df.drop('id', axis=1)

    # call preprocessing function of all filters
    filters.preprocess(df)

    # transform tag map to only list of names, remove low ranked and NSFW tags
    df.tags = df.tags.apply(lambda tags: sorted([tag['name'] for tag in tags if tag['rank'] >= MIN_TAG_RANKING and not tag['isAdult']]))

    # Remove Tagless
    df = df[df['tags'].str.len() != 0]

    # keep only relevant columns, drop everything else
    columns = ['tags', 'title'] + filters.column_name
    df = df.filter(columns)

    # initialize filtering dta
    filters.initialize_data(df)

    return df, filters


def process(preprocessed_data: pd.DataFrame, filters: FilterList):
    t = Task('Filter Data')
    filtered_data = preprocessed_data.copy()
    filters.filter(filtered_data)
    t.end()

    # Sort tags by number of occurrences emulate the papers sort by term weight
    all_occurrences = dict(filtered_data.tags.explode().value_counts())
    all_tags = np.array(sorted(set(filtered_data.tags.explode().drop_duplicates().dropna()),
                key=lambda x: all_occurrences[x],
                reverse=True))

    t = Task('Calculating Similarities')
    pairwise_similarities = _jaccard_matrix(all_tags, filtered_data.tags)
    print(pairwise_similarities)
    filtered_similarities = _filter_similarities(pairwise_similarities)
    print(filtered_similarities)
    debug._plot_tag_similarity_matrix(filtered_similarities, all_tags)
    #debug._filter_tag_pairs_by_similarity(pairwise_similarities, all_tags, 0.15)
    t.end()

    t = Task('Calculating Distances')
    distances = _calculate_distances(filtered_similarities)
    print(distances)
    debug._plot_tag_similarity_matrix(distances, all_tags)
    t.end()


def _jaccard_matrix(all_tags: np.ndarray, tags_column: pd.Series):
    m = np.zeros((len(all_tags), len(all_tags)))
    all_occurrences = dict(tags_column.explode().value_counts())
    co_occurrences = np.zeros((len(all_tags), len(all_tags)))
    tag_ids = {tag: i for i, tag in enumerate(all_tags)}

    # calculate co occurences
    for tags in tags_column:
        for i in range(len(tags)):
            for j in range(i, len(tags)):
                co_occurrences[tag_ids[tags[i]], tag_ids[tags[j]]] += 1
    co_occurrences = co_occurrences.T + co_occurrences
    #co_occurrences[np.diag_indices_from(co_occurrences)] /= 2
    np.fill_diagonal(co_occurrences, 0)

    # calculate jaccard matrix
    for i, tagI in enumerate(all_tags):
        for j, tagJ in enumerate(all_tags):
            union = (all_occurrences[tagI] + all_occurrences[tagJ]) - co_occurrences[i, j]
            m[i, j] = co_occurrences[i, j] / union

    return m


def _filter_similarities(pairwise_similarities: np.ndarray):
    # Filtering maybe isn't needed, since we only got relatively few, but high quality, terms compared to the original paper
    # TODO: Maybe implement implement filter top K terms or remove tags with less than x occurrences here
    return pairwise_similarities


def _logarithmic_scaling(x):
    global SIGMA
    return -np.log((1 - SIGMA) * x + SIGMA)


def _calculate_distances(filtered_similarities: np.ndarray):
    return _logarithmic_scaling(filtered_similarities)


def _cluster_tags():
    pass
