import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from filter import FilterList
from util import Task, PointType

# TODO: remove for submission
import debug
from matplotlib import pyplot as plt

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
    filtered_similarities = _filter_similarities(pairwise_similarities)
    #debug._plot_tag_similarity_matrix(filtered_similarities, all_tags)
    #debug._filter_tag_pairs_by_similarity(pairwise_similarities, all_tags, 0.15)
    t.end()

    t = Task('Calculating Distances')
    distances = _calculate_distances(filtered_similarities)
    if distances.shape[0] != 1:
        tsne = TSNE(random_state=1, n_iter=15000)
        positions: np.ndarray = tsne.fit_transform(distances)
        #mds = MDS(metric=False, n_components=2, max_iter=3000, eps=1e-12, random_state=1)
        #positions = mds.fit_transform(distances)

    else:
        positions = np.array([[0, 0]])

    #debug._plot_scatter(positions, all_tags)
    #debug._plot_tag_similarity_matrix(distances, all_tags)
    t.end()
    #return new_distances.tolist()

    # add point type column
    positions = np.hstack((positions, np.full((positions.shape[0], 1), PointType.DATA.value)))
    positions = _add_random_border(positions)
    #positions = _add_boxes(positions)
    plt.scatter(positions[:,0], positions[:,1])
    plt.show()

    positions[:, :2] = (positions[:, :2] / np.max(np.abs(positions[:, :2])))
    return positions.tolist(), all_tags.tolist()


def _generate_box(x_center, y_center, width, height, point_distance):
    x1 = x_center - width/2
    x2 = x_center + width/2
    y1 = y_center - height/2
    y2 = y_center + height/2
    numx = int((x2 - x1) / point_distance) + 1
    numy = int((y2 - y1) / point_distance) + 1
    x = np.reshape(np.linspace(x1, x2, num=numx), (numx,1))
    y = np.reshape(np.linspace(y1, y2, num=numy), (numy,1))
    top = np.hstack((x, np.full((numx, 1), y1)))
    bottom = np.hstack((x, np.full((numx, 1), y2)))
    left = np.hstack((np.full((numy, 1), x1), y))
    right = np.hstack((np.full((numy, 1), x2), y))
    box = np.vstack((top, bottom, left, right))
    return box


def _add_boxes(positions: np.ndarray):
    _generate_box(0, 0, 1, 1, 0.1)
    for i, point in enumerate(positions):
        if point[2] == PointType.DATA.value:
            box = _generate_box(point[0], point[1], 1, 0.5, 0.05)
            box = np.hstack((box, np.full((box.shape[0], 1), i)))
            positions = np.vstack((positions, box))

    return positions


def _add_random_border(positions: np.ndarray):
    NUM_RANDOM_SAMPLES_PER_CELL = 10
    BORDER_SIZE_PERCENT = 0.15
    BORDER_DISTANCE_PERCENT = 0.10
    CELL_SICE_PERCENT = 0.04
    grid_corners = [positions[:, 0].min(),
                   positions[:, 1].min(),
                   positions[:, 0].max(),
                   positions[:, 1].max()]

    # Extend grid corners by border distance and border size
    dx_data = abs(grid_corners[2] - grid_corners[0])
    dy_data = abs(grid_corners[3] - grid_corners[1])
    dx_data = dx_data if dx_data != 0 else dy_data
    dy_data = dy_data if dy_data != 0 else dx_data
    if dx_data == 0 and dy_data == 0:
        dx_data, dy_data = 1, 1

    border_distance = max(dx_data, dy_data) * BORDER_DISTANCE_PERCENT
    grid_corners[0] -= dx_data * BORDER_DISTANCE_PERCENT + dx_data * BORDER_SIZE_PERCENT
    grid_corners[1] -= dy_data * BORDER_DISTANCE_PERCENT + dy_data * BORDER_SIZE_PERCENT
    grid_corners[2] += dx_data * BORDER_DISTANCE_PERCENT + dx_data * BORDER_SIZE_PERCENT
    grid_corners[3] += dy_data * BORDER_DISTANCE_PERCENT + dy_data * BORDER_SIZE_PERCENT

    dx_grid = abs(grid_corners[2] - grid_corners[0])
    dy_grid = abs(grid_corners[3] - grid_corners[1])

    num_cells_x = int(dx_grid / (dx_grid * CELL_SICE_PERCENT))
    num_cells_y = int(dy_grid / (dy_grid * CELL_SICE_PERCENT))

    cells_borders_x = np.linspace(grid_corners[0], grid_corners[2], num_cells_x + 1)
    cells_borders_y = np.linspace(grid_corners[1], grid_corners[3], num_cells_y + 1)

    random_regular_grid_sample = np.empty((0, 3))

    for i in range(0, num_cells_x):
        for j in range(0, num_cells_y):
            x = np.random.uniform(cells_borders_x[i], cells_borders_x[i+1], (NUM_RANDOM_SAMPLES_PER_CELL, 1))
            y = np.random.uniform(cells_borders_y[j], cells_borders_y[j+1], (NUM_RANDOM_SAMPLES_PER_CELL, 1))
            new_points = np.hstack((x, y, np.full((NUM_RANDOM_SAMPLES_PER_CELL, 1), PointType.BORDER.value)))
            random_regular_grid_sample = np.vstack((random_regular_grid_sample,new_points))

    # filter points that are too close to the actual data
    distances_to_data = sklearn.metrics.pairwise.euclidean_distances(random_regular_grid_sample[:,:2], positions[:,:2])
    mask = distances_to_data.min(1) > border_distance
    positions = np.vstack((positions, random_regular_grid_sample[mask,]))

    return positions


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
