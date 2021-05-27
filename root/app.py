import json

from flask import Flask, render_template, request
import pandas as pd
import pipeline
from pipeline import FilterableData
import time

app = Flask(__name__)
available_filter = None
selected_filter = None


def _clean_filters(selected_filter):
    if "All" in selected_filter.studios or not len(selected_filter.studios):
        selected_filter.studios = ["All"]

    if "All" in selected_filter.media_types or not len(selected_filter.media_types):
        selected_filter.media_types = ["All"]


@app.route("/", methods=["GET", "POST"])
def index():
    global selected_filter
    global available_filter

    selected_filter = FilterableData()
    selected_filter.studios = x if (x := request.form.getlist('studios[]', type=str)) else ["All"]
    selected_filter.media_types = x if (x := request.form.getlist('media_types[]', type=str)) else ["All"]
    selected_filter.release_year_range = (request.form.get('release_year_min', type=int, default=available_filter.release_year_range[0]),
                                 request.form.get('release_year_max', type=int, default=available_filter.release_year_range[1]))

    _clean_filters(selected_filter)
    t = pipeline.Task('Processing Pipeline')
    map_data = pipeline.process(preprocessed_data, selected_filter)
    t.end()

    return render_template("index.html", available_filter=available_filter, selected_filter=selected_filter, map_data=map_data)


if __name__ == "__main__":

    t = pipeline.Task('Reading JSON file')
    raw_data = pd.read_json('../anilist.json')
    t.end()

    t = pipeline.Task('Preprocessing Data')
    preprocessed_data, available_filter = pipeline.preprocess(raw_data)
    t.end()

    # TODO: disable debug so program doesn't run twice on initialization
    app.run(debug=True)