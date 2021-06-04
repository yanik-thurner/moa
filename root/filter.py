import itertools
from flask import request
import pandas as pd
from enum import Enum


class Filter:
    """
    Prototype class for Filters, so that all Filter specific code only occurs in one place
    """
    class FilterType(Enum):
        """
        Type of the filter, either BASEMAP or HEATMAP
        """
        BASEMAP = '_b'
        HEATMAP = '_h'

    def __init__(self, type: FilterType):
        self.available_values: list = []
        self.selected_values: list = []
        self.type = type

    def initialize_data(self, data: pd.DataFrame) -> None:
        """
        Initialize available values here.
        """
        raise NotImplementedError

    def clean(self) -> None:
        """
        Clean the selected values, for example select all if the list is empty
        """
        raise NotImplementedError

    def preprocess(self, data: pd.DataFrame):
        """
        Needed steps for preprocessing the data, for example renaming columns or merging entries. Should be done inplace.
        @param data: the pandas DataFrame of the raw data
        """
        raise NotImplementedError

    def filter(self, data: pd.DataFrame):
        """
        Filter the DataFrame with the selected filters. Should be done inplace.
        @param data:
        """
        raise NotImplementedError

    @property
    def column_name(self):
        """
        Return the name of the column of the filtered attribute
        @return: name as string
        """
        raise NotImplementedError

    def refresh_selection(self):
        """
        Fetch the transmitted data via the requests module. Set the selected values accordingly.
        @return:
        """
        raise NotImplementedError


class FilterList(Filter):
    """
    Class that manages all filters. New filters should be added in the __init__ method. Implements Filter since it
    should have all filter methods to call for each available filter.
    """
    def __init__(self, type: Filter.FilterType = Filter.FilterType.BASEMAP):
        self._map = {}
        self._index = 0
        self._map.update({(x := StudioFilter(type)).column_name: x})
        self._map.update({(x := ReleaseYearFilter(type)).column_name: x})
        self._map.update({(x := MediaTypeFilter(type)).column_name: x})

    def set_type(self, type: Filter.FilterType):
        """
        Set the type of the filter
        @param type: filter type
        """
        for filter_key in self._map.keys():
            self._map[filter_key].type = type

    def __iter__(self):
        return self

    def __getitem__(self, item) -> Filter:
        return self._map[item]

    def __next__(self) -> Filter:
        self._index += 1
        try:
            return self._map[list(self._map.keys())[self._index - 1]]
        except IndexError:
            self._index = 0
            raise StopIteration

    def initialize_data(self, data):
        for f in self:
            f.initialize_data(data)

    def clean(self):
        for f in self:
            f.clean()

    def preprocess(self, data):
        for f in self:
            f.preprocess(data)

    def filter(self, data):
        for f in self:
            f.filter(data)

    @property
    def column_name(self):
        """
        return lists of all implemented filter column names
        @return: list of str
        """
        cns = []
        for f in self:
            cns.append(f.column_name)
        return cns

    def refresh_selection(self):
        for f in self:
            f.refresh_selection()


class StudioFilter(Filter):
    def initialize_data(self, data):
        self.available_values = sorted(set(itertools.chain.from_iterable(data[self.column_name])))

    def clean(self):
        pass

    def preprocess(self, data):
        # extract studio names from list
        data[self.column_name] = data[self.column_name].apply(
            lambda studio: sorted([node['name'] for node in studio['nodes']
                                   if node['isAnimationStudio']
                                   or not node['isAnimationStudio']]))  # TODO: Maybe remove or

    def filter(self, data):
        if "All" not in self.selected_values:
            mask = data.studios.apply(
                lambda x: any(studio for studio in self.selected_values if studio in x))
            data.drop(data[~mask].index, inplace=True)

    @property
    def column_name(self):
        return 'studios'

    def refresh_selection(self):
        self.selected_values = x if (x := request.form.getlist(f'{self.column_name}{self.type.value}[]', type=str)) else ["All"]

        if "All" in self.selected_values or not len(self.selected_values):
            self.selected_values = ["All"]


class ReleaseYearFilter(Filter):
    def refresh_selection(self):
        self.selected_values = [x for x in range(
            request.form.get(f'{self.column_name}_min{self.type.value}', type=int, default=self.available_values[0]),
            request.form.get(f'{self.column_name}_max{self.type.value}', type=int, default=self.available_values[-1]) + 1)]

    def initialize_data(self, data):
        self.available_values = [x for x in range(data[self.column_name].min(), data[self.column_name].max())]

    def clean(self):
        pass

    def preprocess(self, data):
        pass

    def filter(self, data):
        pass

    @property
    def column_name(self):
        return 'seasonYear'


class MediaTypeFilter(Filter):
    def initialize_data(self, data):
        self.available_values = sorted(filter(None, set(data[self.column_name])))

    def clean(self):
        pass

    def preprocess(self, data):
        pass

    def filter(self, data):
        if "All" not in self.selected_values:
            mask = data[self.column_name].isin(self.selected_values)
            data.drop(data[~mask].index, inplace=True)

    @property
    def column_name(self):
        return 'format'

    def refresh_selection(self):
        self.selected_values = x if (x := request.form.getlist(f'{self.column_name}{self.type.value}[]', type=str)) else ["All"]
        if "All" in self.selected_values or not len(self.selected_values):
            self.selected_values = ["All"]
