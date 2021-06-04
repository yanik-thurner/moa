import json

from flask import Flask, render_template, request
import pandas as pd
import pipeline
from filter import FilterList, Filter
import os
import copy

app = Flask(__name__)
filters_base: FilterList = None
filters_heat: FilterList = None
CACHE_LOCATION = '../cached_data.pkl'
f = 'false'


@app.route("/", methods=["GET", "POST"])
def index():
    global filters

    filters.refresh_selection()

    t = pipeline.Task('Processing Pipeline')
    tags, basemap_points, basemap_edges, heat_tags = pipeline.process(preprocessed_data, filters)
    t.end()

    return render_template("index.html", filters_base=filters_base,
                           filters_heat=filters_heat,
                           basemap_points=basemap_points,
                           basemap_edges=basemap_edges,
                           heat_tags=heat_tags,
                           tags=tags, circle=f)


if __name__ == "__main__":
    t = pipeline.Task('Reading JSON file')
    if os.path.exists(CACHE_LOCATION):
        raw_data = pd.read_pickle(CACHE_LOCATION)
    else:
        raw_data = pd.read_json('../anilist.json')
        print('Caching json file in pkl for faster loading speed')
        raw_data.to_pickle(CACHE_LOCATION)

    t.end()

    t = pipeline.Task('Preprocessing Data')
    preprocessed_data, filters = pipeline.preprocess(raw_data)
    filters_base = filters
    filters_heat = copy.deepcopy(filters)
    filters_heat.set_type(Filter.FilterType.HEATMAP)
    t.end()

    # TODO: disable debug so program doesn't run twice on initialization
    app.run(debug=True)