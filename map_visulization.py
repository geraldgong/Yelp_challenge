import plotly.graph_objects as go
import pandas as pd

coord = pd.read_csv('./yelp_dataset/coord.csv')
# coord['is_open'] = coord['is_open'].values * 255
scl = [0,"rgb(255,0,0)"], [1,"rgb(0, 0, 255)"]

mapbox_access_token = 'pk.eyJ1IjoieXV4aWFuZ2dvbmciLCJhIjoiY2p6M2I1ejJnMDFsNjNjcXRpdmw4cjAweCJ9.ecfVq0S-fu56dHr50J2YwQ'

fig = go.Figure(go.Scattermapbox(
        lat=coord['latitude'],
        lon=coord['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9,
            color = coord['is_open'],
            colorscale = scl
        ),

    ))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=40.436447,
            lon=-79.922706,
        ),
        pitch=0,
        zoom=10
    ),
)

fig.show()