import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt  # TODO: Remove
import time

MIN_TAG_RANKING = 20


class Task:
    def __init__(self, task_name):
        self.name = task_name
        self.start_time = time.time()
        print(f'--- STARTING: {task_name} ---')

    def end(self):
        print(f'--- FINISHING:  {self.name} in {"{:.3f}".format(time.time() - self.start_time)}s')


class FilterableData:
    def __init__(self):
        self.studios = None
        self.release_year_range = None
        self.media_types = None


def wrap(preprocessed_data: pd.DataFrame) -> FilterableData:
    # wrap it into AnilistData object
    data = FilterableData()
    data.studios = sorted(set(itertools.chain.from_iterable(preprocessed_data.studios)))
    data.release_year_range = (preprocessed_data.seasonYear.min(), preprocessed_data.seasonYear.max())
    data.media_types = sorted(filter(None, set(preprocessed_data.format)))  # TODO: maybe handle None values instead of filter
    return data


def preprocess(raw_data: pd.DataFrame) -> FilterableData:
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
    return df, wrap(preprocessed_data=df)


def process(preprocessed_data, available_filter, selected_filter):
    all_tags = np.array(sorted(set(preprocessed_data.tags.explode().drop_duplicates().dropna())))

    t = Task('Calculating Similarity Matrix')
    similarity_matrix = jaccard_matrix(all_tags, preprocessed_data.tags)
    t.end()


def jaccard_matrix(all_tags: np.ndarray, tags_column: pd.Series):
    m = np.empty((len(all_tags), len(all_tags)))
    all_occurences = dict(tags_column.explode().value_counts())
    co_occureces = np.empty((len(all_tags), len(all_tags)))
    tag_ids = {tag: i for i, tag in enumerate(all_tags)}

    # calculate co occurences
    for tags in tags_column:
        for i in range(len(tags)):
            for j in range(i, len(tags)):
                co_occureces[tag_ids[tags[i]], tag_ids[tags[j]]] += 1

    co_occureces = co_occureces.T + co_occureces
    np.fill_diagonal(co_occureces, 0)

    # calculate jaccard matrix
    for i, tagI in enumerate(all_tags):
        for j, tagJ in enumerate(all_tags):
            union = (all_occurences[tagI] + all_occurences[tagJ]) - co_occureces[i, j]
            m[i, j] = co_occureces[i, j] / union

    return m


def calculate_distances():
    pass


def cluster_tags():
    pass