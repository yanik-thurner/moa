import json

from flask import Flask, render_template, request
import pandas as pd
import pipeline
from filter import FilterList

app = Flask(__name__)
filters: FilterList = None


@app.route("/", methods=["GET", "POST"])
def index():
    global filters

    filters.refresh_selection()

    t = pipeline.Task('Processing Pipeline')
    map_data = pipeline.process(preprocessed_data, filters)
    t.end()

    return render_template("index.html", filters=filters, map_data=map_data)


if __name__ == "__main__":

    t = pipeline.Task('Reading JSON file')
    raw_data = pd.read_json('../anilist.json')
    t.end()

    t = pipeline.Task('Preprocessing Data')
    preprocessed_data, filters = pipeline.preprocess(raw_data)
    t.end()

    # TODO: disable debug so program doesn't run twice on initialization
    app.run(debug=True)