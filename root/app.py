import json

from flask import Flask, render_template, request
import pandas as pd
import pipeline
from filter import FilterList
import os

app = Flask(__name__)
filters: FilterList = None
CACHE_LOCATION = '../cached_data.pkl'
f = 'false'

@app.route("/", methods=["GET", "POST"])
def index():
    global filters

    filters.refresh_selection()

    t = pipeline.Task('Processing Pipeline')
    basemap_points, basemap_edges, tags = pipeline.process(preprocessed_data, filters)
    t.end()

    return render_template("index.html", filters=filters, basemap_points=basemap_points, basemap_edges=basemap_edges, tags=tags, circle=f)


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
    t.end()

    # TODO: disable debug so program doesn't run twice on initialization
    app.run(debug=False, port=5000)