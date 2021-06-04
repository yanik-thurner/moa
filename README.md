Maps of Anime
===
## Idea
<img width="200" align="right" src="https://i.imgur.com/nNG1JAt.png">

This application is based on the paper by [Fried et al](https://arxiv.org/abs/1304.2681). We use this technique to visualize the relationship of tag data for different animation series with data from [anilist](https://anilist.co). Our idea is to show how close different tags are related to each other and visualize what kind of shows different animation sutdios produced over a certain time period. 
Similar to the original paper we can filter differently for the basemap and the heatmap to explore the data. Instead of journal or year, in our application one can filter by studio, release year and media type (like movie, music video, etc.). The filtering interface can be seen on the right.


The map implements all features one would expect like zooming and panning.
Below are two example results (cropped for easier readability). One the left one a strong relation between the tags mafia, drugs, delinquents.. etc can be seen shown by the background color and the lines connecting them which means that these tags usually appear together or in similar shows.
The heatmap in the example on the right represents the produced series of the animation studio Kyoto Animation in the years 2000-2010, the image shows that the studio mainly produced shows in a school setting with a female cast.

<img height="250" src="https://i.imgur.com/zszBsDD.png"> &nbsp; &nbsp; &nbsp; <img height="250" src="https://i.imgur.com/XSd3spw.png">

## Usage
To use this application several packages have to be installed, preferably in an virtual environment to reduce version issues. These are listed in the provided requirements.txt and can be installed in the following way:

```bash
pip install -r requirements.txt
```

To execute the program open the folder as a project in an IDE like PyCharm or execute the following commands:

```bash
cd root
python app.py
```




## Implementation
The python backend is responsible for scraping the data from the anilist api and processing it up to the part where the similarity based distances are calculated. The following packeges are needed:
- flask
- pandas
- numpy
- networkx
- scikit-learn
- scikit-network


The frontend uses D3 for the data visualization which includes the generation of the map and heatmap. As well as Jquery for the sliders and Dropdown menus.



