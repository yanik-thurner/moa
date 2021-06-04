import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from root.filter import FilterList
from root.util import Task, PointType
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import Delaunay
import sknetwork as skn
from sknetwork.clustering import Louvain


"""
The minimum ranking (ranging from 0 to 100) to be considered a valid tag for a show.
"""
MIN_TAG_RANKING = 20
MIN_TAG_APPEARANCES = 5

"""
Used for logarithmic scaling when calculating the distances.

According to the paper:
[...] sigma is a small, positive, constant scaling value, currently set to
0.1, used to ensure a non-zero value inside the logarithm in the case
that two terms have a pairwise similarity of 0.
"""
SIGMA = 0.1


def preprocess(raw_data: pd.DataFrame) -> (pd.DataFrame, FilterList):
    """
    Preprocessing pipeline, taking the raw data from the json file as a pandas DataFrame and clean it up, e.g. filter
    invalid values. Also runs all of the preprocessing steps required for each filter.
    @param raw_data: raw data from json file as pandas DataFrame
    @return: preprocessed dataframe and initialized filters.
    """
    # initialize filters
    filters = FilterList()

    # transpose so columns are attributes and rows are anime ids
    df = raw_data.transpose(copy=True)

    # remove id col since it is already the index
    df = df.drop('id', axis=1)

    # call preprocessing function of all filters
    filters.preprocess(df)

    # transform tag map to only list of names, remove low ranked and NSFW tags
    df.tags = df.tags.apply(lambda tags: sorted([tag['name'] for tag in tags if tag['rank'] >= MIN_TAG_RANKING and not tag['isAdult']]))

    # Remove uncommon tags
    tag_count = df.tags.explode().value_counts()
    uncommon = tag_count[tag_count < MIN_TAG_APPEARANCES].index.tolist()
    df.tags = df.tags.apply(lambda tags: [tag for tag in tags if tag not in uncommon])

    # Remove Tagless
    df = df[df['tags'].str.len() != 0]

    # keep only relevant columns, drop everything else
    columns = ['tags', 'title'] + filters.column_name
    df = df.filter(columns)

    # initialize filtering dta
    filters.initialize_data(df)

    return df, filters


def process(preprocessed_data: pd.DataFrame, filters_base: FilterList, filters_heat: FilterList):
    """
    Main processing pipeline of this application. The main tasks are:
        - Filtering the Data based on the selected filter values
        - Calculate the pairwise similarities of the filtered tags
        - Scale the similarities to better represent distances.
        - Generate the edges for the basemap
        - Space out the map points so that the text wont overlap less
        - Generate the countries for the basemap
        - Generate support points i.e. random points around the data for smoother edges outside and boxes points around
          the data for smoother edges inside
        - Calculate the frequencies for the heatmap
        - normalize the data points so that they can be scaled in the frontend
    @param preprocessed_data: data resulting from the preprocess method as a Pandas DataFrame
    @param filters_base: FilterList for Basemap filters
    @param filters_heat: FilterList for Heatmap filters
    @return: returns a tuple of:
        - tags for the basemap
        - position of data points including type, county, and number of occurrences
        - list of edges edges referencing the index of the disappoints
        - number of occurrences for heatmap
    """
    t = Task('Filter Data')
    filtered_data_base = preprocessed_data.copy()
    filtered_data_heat = preprocessed_data.copy()
    filters_base.filter(filtered_data_base)
    filters_heat.filter(filtered_data_heat)
    t.end()

    # Sort tags by number of occurrences emulate the papers sort by term weight
    all_occurrences = dict(filtered_data_base.tags.explode().value_counts())
    all_tags = np.array(sorted(set(filtered_data_base.tags.explode().drop_duplicates().dropna()),
                key=lambda x: (all_occurrences[x], x),
                reverse=True))

    t = Task('Calculating Similarities')
    pairwise_similarities = _jaccard_matrix(all_tags, filtered_data_base.tags)
    filtered_similarities, filtered_tags = _filter_similarities(pairwise_similarities, all_tags)
    t.end()

    t = Task('Calculating Distances')
    distances = _calculate_distances(filtered_similarities)
    if distances.shape[0] != 1:
        tsne = TSNE(random_state=1, n_iter=15000, init='pca')
        positions: np.ndarray = tsne.fit_transform(distances)
    else:
        positions = np.array([[0, 0]])
    t.end()

    t = Task('Generating Edges')
    edges = _generate_edges(filtered_similarities, positions, 2, 0.2)
    t.end()

    t = Task('Spacing out Mappoints')
    G = nx.Graph()
    tri = Delaunay(positions)
    for path in tri.simplices:
        nx.add_path(G, path)
    G.add_edges_from(edges)
    graph_positions = nx.kamada_kawai_layout(G)
    for k in sorted(graph_positions.keys()):
        positions[k, 0] = graph_positions[k][0]
        positions[k, 1] = graph_positions[k][1]
    t.end()

    t = Task('Generating Countries')
    adjacency = np.zeros((len(filtered_tags), len(filtered_tags)))
    for edge in _generate_edges(filtered_similarities, positions, 6, 0.3):
        adjacency[edge[0], edge[1]] += 1
    adjacency += adjacency.T
    louvain = Louvain()
    basemap_countries = louvain.fit_transform(adjacency)
    t.end()

    t = Task('Add support points')
    # generate support
    random_points = _generate_random_border(positions)
    boxes = _generate_boxes(positions)
    # attach them
    positions = np.vstack((positions, random_points, boxes))
    # add adtiontal columns
    num_box_points_per_box = int(len(boxes)/len(filtered_tags))
    num_support = len(positions) - len(filtered_tags)
    point_types = np.array([[PointType.DATA.value]*len(filtered_tags) +
                            [PointType.BORDER.value]*len(random_points) +
                            sorted([x for x in range(len(all_tags))] * num_box_points_per_box)]).T
    point_countries = np.array([basemap_countries.tolist() + [-1] * num_support]).T
    point_occurrences = np.array([[all_occurrences[x] for x in filtered_tags] + [0] * num_support]).T
    positions = np.hstack((positions, point_types, point_countries, point_occurrences))
    t.end()

    t = Task('Calculate Heatmap occurrences')
    heat_occurrences_dict = dict(filtered_data_heat.tags.explode().value_counts())
    heat_tags = np.array(sorted(set(filtered_data_heat.tags.explode().drop_duplicates().dropna()),
                               key=lambda x: (heat_occurrences_dict[x], x),
                               reverse=True))

    heat_tags = [x for x in heat_tags]
    heat_occurrences_sorted = [heat_occurrences_dict[x] if x in heat_tags else 0 for x in filtered_tags]
    t.end()

    # normalize point data
    positions[:, :2] = (positions[:, :2] / np.max(np.abs(positions[:, :2])))
    return filtered_tags.tolist(), positions.tolist(), edges.tolist(), heat_occurrences_sorted


def _generate_edges(pairwise_similarities: np.ndarray, positions: np.ndarray, max_edges=2, sample_percent=0.05):
    """
    Generate the edges based on similarities. Similarities are sorted descending, self loops are filtered and then a
    certain number of points are samples. From these samples the edges are drawn to the nearest points.
    @param pairwise_similarities: NxN similiarity matrix as np.array
    @param positions: positions resulting from dimensionality reduction
    @param max_edges: soft maximum of edges as described in the paper
    @param sample_percent: percent of overall points of how many best matches should be sampled
    @return: list of lists containing pairs of points which are connected by edges.
    """
    num_samples = int(len(pairwise_similarities) * sample_percent)
    edges = np.ndarray((0,2), dtype=int)
    for i in range(pairwise_similarities.shape[0]):

        # get indexes of most similar
        best_matches_index = pairwise_similarities[i].argsort()[::-1]
        # filter loops
        best_matches_index = best_matches_index[best_matches_index != i]
        # sample points
        best_matches_index = best_matches_index.flatten()[:num_samples]
        # calculate distances
        best_matches_distances = euclidean_distances(np.array([positions[i]]), positions[best_matches_index])
        # get nearest
        best_matches_index = best_matches_index[np.argsort(best_matches_distances)[0][:max_edges]]

        best_matches_index = np.array([best_matches_index]).T
        current_edges = np.hstack((np.full((len(best_matches_index), 1), i), best_matches_index))

        # filter existing
        if len(edges) > 0 and len(current_edges) > 0:
            current_edges = current_edges[euclidean_distances(current_edges[:, ::-1], edges).min(1) != 0]

        edges = np.vstack((edges, current_edges))

    return edges


def _generate_box(x_center, y_center, width, height, point_distance):
    """
    Generate points in the form of a rectangle around a certain point with provided width, height, and distance between
    those points.
    @param x_center: X coordinate of box center
    @param y_center: Y coordinate of box center
    @param width: width of the box
    @param height: height of the box
    @param point_distance: distance between the points that form the box
    @return: points that form the box
    """
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
    return np.unique(box, axis=0)


def _generate_boxes(positions: np.ndarray):
    """
    generates a box and moves a copy of it to all data point positions. Returns the list of box points
    @param positions: np array of data points
    @return: np array of box points
    """
    blueprint = _generate_box(0, 0, 0.08, 0.06, 0.006)
    boxes = np.ndarray((0, 2))
    for i, point in enumerate(positions):
        b = blueprint.copy()
        b[:, 0] += point[0]
        b[:, 1] += point[1]
        boxes = np.vstack((boxes, b))
    return boxes


def _generate_random_border(positions: np.ndarray):
    """
    Generates random points around the cluster of data points. The points are sampled randomly in each cell of a
    regular grid.
    @param positions: positions of data points
    @return: np array of random points around the cluster
    """
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

    # extend the corners of the sample grid so that there is enough space around the cluster
    border_distance = max(dx_data, dy_data) * BORDER_DISTANCE_PERCENT
    grid_corners[0] -= dx_data * BORDER_DISTANCE_PERCENT + dx_data * BORDER_SIZE_PERCENT
    grid_corners[1] -= dy_data * BORDER_DISTANCE_PERCENT + dy_data * BORDER_SIZE_PERCENT
    grid_corners[2] += dx_data * BORDER_DISTANCE_PERCENT + dx_data * BORDER_SIZE_PERCENT
    grid_corners[3] += dy_data * BORDER_DISTANCE_PERCENT + dy_data * BORDER_SIZE_PERCENT

    # calculate the borders of cells of the regular grid.
    dx_grid = abs(grid_corners[2] - grid_corners[0])
    dy_grid = abs(grid_corners[3] - grid_corners[1])
    num_cells_x = int(dx_grid / (dx_grid * CELL_SICE_PERCENT))
    num_cells_y = int(dy_grid / (dy_grid * CELL_SICE_PERCENT))
    cells_borders_x = np.linspace(grid_corners[0], grid_corners[2], num_cells_x + 1)
    cells_borders_y = np.linspace(grid_corners[1], grid_corners[3], num_cells_y + 1)

    # sample in the grid
    random_regular_grid_sample = np.empty((0, 2))
    for i in range(0, num_cells_x):
        for j in range(0, num_cells_y):
            x = np.random.uniform(cells_borders_x[i], cells_borders_x[i+1], (NUM_RANDOM_SAMPLES_PER_CELL, 1))
            y = np.random.uniform(cells_borders_y[j], cells_borders_y[j+1], (NUM_RANDOM_SAMPLES_PER_CELL, 1))
            new_points = np.hstack((x, y))
            random_regular_grid_sample = np.vstack((random_regular_grid_sample, new_points))

    # filter points that are too close to the actual data
    distances_to_data = euclidean_distances(random_regular_grid_sample[:,:2], positions[:,:2])
    mask = distances_to_data.min(1) > border_distance
    return random_regular_grid_sample[mask, ]


def _jaccard_matrix(all_tags: np.ndarray, tags_column: pd.Series):
    """
    Calculate the Jaccard coefficient for each pair of tags which is defined as:
    J(t1, t2) = Number of common occurrences(t1, t2) / Number of individual occurrences(t1, t2)
    @param all_tags: list of all tags
    @param tags_column: tags column of the filtered data
    @return: NxN matrix of jaccard coefficients where N is the number of all tags
    """
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
    co_occurrences[np.diag_indices_from(co_occurrences)] /= 2

    # calculate jaccard matrix
    for i, tagI in enumerate(all_tags):
        for j, tagJ in enumerate(all_tags):
            union = (all_occurrences[tagI] + all_occurrences[tagJ]) - co_occurrences[i, j]
            m[i, j] = co_occurrences[i, j] / union

    return m


def _filter_similarities(pairwise_similarities: np.ndarray, all_tags):
    """
    Placeholder method, since this is a pipeline step in the original implementation but, to this point, was not needed
    in our case.
    @param pairwise_similarities: NxN matrix of similarities
    @param all_tags: list of all tags
    @return: filtered similarities and tags
    """
    # Filtering isn't needed, since we only got relatively few, but high quality, terms compared to the original paper
    return pairwise_similarities, all_tags


def _logarithmic_scaling(x: float) -> float:
    """"
    Method to scale a similarity logarithmically to represent a distance, as described in the paper
    """
    global SIGMA
    return -np.log((1 - SIGMA) * x + SIGMA)


def _calculate_distances(filtered_similarities: np.ndarray):
    """
    Scales each value of the similarity matrix logarithmically
    @param filtered_similarities: NxN similarity matrix
    @return: NxN distance matrix
    """
    return _logarithmic_scaling(filtered_similarities)
