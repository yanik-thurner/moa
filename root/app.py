import json

from flask import Flask, render_template
import pandas as pd
import itertools


class FilterableData:
    def __init__(self):
        self.studios = None
        self.season_years_range = None
        self.media_types = None


def wrap(preprocessed_data: pd.DataFrame) -> FilterableData:
    # wrap it into AnilistData object
    data = FilterableData()
    data.studios = sorted(set(itertools.chain.from_iterable(preprocessed_data.studios)))
    data.season_years_range = (preprocessed_data.seasonYear.min(), preprocessed_data.seasonYear.max())
    data.media_types = set(preprocessed_data.format)
    return data


def preprocess(raw_data: pd.DataFrame) -> FilterableData:
    # transpose so columns are attributes and rows are anime ids
    df = raw_data.transpose(copy=True)

    # remove id col since it is already the index
    df = df.drop('id', axis=1)

    # extract studio names from list
    df.studios = df.studios.apply(lambda studio: [node['name'] for node in studio['nodes']
                                                  if node['isAnimationStudio']
                                                  or not node['isAnimationStudio']])  # TODO: Maybe remove or

    return wrap(preprocessed_data=df)


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", data=filterable_data)


if __name__ == "__main__":

    raw_data = pd.read_json('../anilist.json')
    filterable_data = preprocess(raw_data)

    app.run(debug=True)