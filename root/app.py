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
    """
    Main root method, gets called when a GET or POST at the main page occurs
    """
    global filters_base
    global filters_heat
    global first
    filters_base.refresh_selection()
    filters_base.clean()
    filters_heat.refresh_selection()
    filters_heat.clean()

    if len(request.form) == 0:
        show_edges = True
        show_heatmap = True
    else:
        show_edges = request.form.get('show_edges', type=bool, default=False)
        show_heatmap = request.form.get('show_heatmap', type=bool, default=False)

    t = pipeline.Task('Processing Pipeline')
    tags, basemap_points, basemap_edges, heat_tags = pipeline.process(preprocessed_data, filters_base, filters_heat)
    t.end()

    return render_template("index.html", filters_base=filters_base,
                           filters_heat=filters_heat,
                           basemap_points=basemap_points,
                           basemap_edges=basemap_edges,
                           heat_tags=heat_tags,
                           tags=tags, circle=f,
                           show_edges=show_edges, show_heatmap=show_heatmap)


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

    app.run(debug=True)