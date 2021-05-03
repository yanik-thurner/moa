import json

from flask import Flask, render_template, request
import pandas as pd
import pipeline
from pipeline import FilterableData
import time

app = Flask(__name__)
available_filter = None
selected_filter = None

@app.route("/", methods=["GET", "POST"])
def index():
    global selected_filter
    global available_filter

    selected_filter = FilterableData()
    selected_filter.studios = x if (x := request.form.getlist('studios[]', type=int)) else [-1]
    selected_filter.media_types = x if (x := request.form.getlist('media_types[]', type=int)) else [-1]
    selected_filter.release_year_range = (request.form.get('release_year_min', type=int, default=available_filter.release_year_range[0]),
                                 request.form.get('release_year_max', type=int, default=available_filter.release_year_range[1]))

    return render_template("index.html", available_filter=available_filter, selected_filter=selected_filter, map_data=None)


if __name__ == "__main__":

    t = pipeline.Task('Reading JSON file')
    raw_data = pd.read_json('../anilist.json')
    t.end()

    t = pipeline.Task('Preprocessing Data')
    preprocessed_data, available_filter = pipeline.preprocess(raw_data)
    t.end()

    t = pipeline.Task('Processing Pipeline')
    map_data = pipeline.process(preprocessed_data, available_filter, selected_filter)
    t.end()

    app.run(debug=True)