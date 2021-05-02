import json

from flask import Flask, render_template, request
import itertools
import pandas as pd


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
    data.media_types = sorted(filter(None, set(preprocessed_data.format)))
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


@app.route("/", methods=["GET", "POST"])
def index():
    filter = FilterableData()
    filter.studios = x if (x := request.form.getlist('studios[]', type=int)) else [-1]
    filter.media_types = x if (x := request.form.getlist('media_types[]', type=int)) else [-1]
    filter.release_year_range = (request.form.get('release_year_min', type=int, default=filterable_data.release_year_range[0]),
                                 request.form.get('release_year_max', type=int, default=filterable_data.release_year_range[1]))
    return render_template("index.html", data=filterable_data, filter=filter)



if __name__ == "__main__":

    raw_data = pd.read_json('../anilist.json')
    filterable_data = preprocess(raw_data)

    app.run(debug=True)