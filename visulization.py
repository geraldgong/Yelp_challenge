import plotly.graph_objects as go
import pandas as pd

coords = pd.read_csv('/home/gong/Documents/Yelp_challenge/yelp_dataset/coords.csv')
lat = coords.latitude.values.tolist()
lag = coords.longitude.values.tolist()

mapbox_access_token = 'pk.eyJ1IjoieXV4aWFuZ2dvbmciLCJhIjoiY2p6M2I1ejJnMDFsNjNjcXRpdmw4cjAweCJ9.ecfVq0S-fu56dHr50J2YwQ'

fig = go.Figure(go.Scattermapbox(
        lat=lat,
        lon=lag,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),

    ))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=36.128561,
            lon=-115.17113,
        ),
        pitch=0,
        zoom=10
    ),
)

fig.show()