import numpy as np
import pandas as pd
import itertools

# TODO: remove for submission
from util import Task
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


class FilterableData:
    def __init__(self, preprocessed_data: pd.DataFrame = None):
        if preprocessed_data is not None:
            self.studios = sorted(set(itertools.chain.from_iterable(preprocessed_data.studios)))
            self.release_year_range = (preprocessed_data.seasonYear.min(), preprocessed_data.seasonYear.max())
            self.media_types = sorted(filter(None, set(preprocessed_data.media_type)))  # TODO: maybe handle None values instead of filter
        else:
            self.studios = None
            self.release_year_range = None
            self.media_types = None


def preprocess(raw_data: pd.DataFrame) -> (pd.DataFrame, FilterableData):
    # transpose so columns are attributes and rows are anime ids
    df = raw_data.transpose(copy=True)

    # remove id col since it is already the index
    df = df.drop('id', axis=1)

    # extract studio names from list
    df.studios = df.studios.apply(lambda studio: sorted([node['name'] for node in studio['nodes']
                                                  if node['isAnimationStudio']
                                                  or not node['isAnimationStudio']]))  # TODO: Maybe remove or

    # transform tag map to only list of names, remove low ranked and NSFW tags
    df.tags = df.tags.apply(lambda tags: sorted([tag['name'] for tag in tags if tag['rank'] >= MIN_TAG_RANKING and not tag['isAdult']]))

    # keep only relevant columns, drop everything else
    df = df.filter(['tags', 'title', 'studios', 'format', 'seasonYear'])  # TODO: Add new columns for filtering here

    # rename to be consistent lol
    df.rename(columns={'format': 'media_type'}, inplace=True)
    return df, FilterableData(df)


def process(preprocessed_data: pd.DataFrame, selected_filter: FilterableData):
    t = Task('Filter Data')
    filtered_data = _filter(preprocessed_data, selected_filter)
    t.end()

    all_tags = np.array(sorted(set(filtered_data.tags.explode().drop_duplicates().dropna())))

    t = Task('Calculating Similarities')
    pairwise_similarities = _jaccard_matrix(all_tags, filtered_data.tags)
    filtered_similarities = _filter_similarities(pairwise_similarities)
    debug._plot_tag_similarity_matrix(filtered_similarities, all_tags)
    debug._filter_tag_pairs_by_similarity(pairwise_similarities, all_tags, 0.15)
    t.end()

    t = Task('Calculating Distances')
    distances = _calculate_distances(filtered_similarities)
    debug._plot_tag_similarity_matrix(distances, all_tags)
    t.end()


def _filter(preprocessed_data: pd.DataFrame, selected_filter: FilterableData):
    filtered_data = preprocessed_data.copy()  # TODO: Maybe no copy?
    if "All" not in selected_filter.studios:
        mask = filtered_data.studios.apply(lambda x: any(studio for studio in selected_filter.studios if studio in x))
        filtered_data = filtered_data[mask]

    if "All" not in selected_filter.media_types:
        filtered_data = filtered_data[filtered_data['media_type'].isin(selected_filter.media_types)]

    return filtered_data


def _jaccard_matrix(all_tags: np.ndarray, tags_column: pd.Series):
    m = np.empty((len(all_tags), len(all_tags)))
    all_occurrences = dict(tags_column.explode().value_counts())
    co_occurrences = np.empty((len(all_tags), len(all_tags)))
    tag_ids = {tag: i for i, tag in enumerate(all_tags)}

    # calculate co occurences
    for tags in tags_column:
        for i in range(len(tags)):
            for j in range(i, len(tags)):
                co_occurrences[tag_ids[tags[i]], tag_ids[tags[j]]] += 1
    co_occurrences = co_occurrences.T + co_occurrences
    np.fill_diagonal(co_occurrences, 0)

    # calculate jaccard matrix
    for i, tagI in enumerate(all_tags):
        for j, tagJ in enumerate(all_tags):
            union = (all_occurrences[tagI] + all_occurrences[tagJ]) - co_occurrences[i, j]
            m[i, j] = co_occurrences[i, j] / union

    return m


def _filter_similarities(pairwise_similarities: np.ndarray):
    # Maybe not needed since we already have very few terms compared to the original paper
    return pairwise_similarities  # TODO: implement filter if needed


def _logarithmic_scaling(x):
    global SIGMA
    return -np.log((1 - SIGMA) * x + SIGMA)


def _calculate_distances(filtered_similarities: np.ndarray):
    return _logarithmic_scaling(filtered_similarities)


def _cluster_tags():
    pass
