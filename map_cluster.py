import plotly.express as px
import pandas as pd
px.set_mapbox_access_token("pk.eyJ1IjoieXV4aWFuZ2dvbmciLCJhIjoiY2p6M2I1ejJnMDFsNjNjcXRpdmw4cjAweCJ9.ecfVq0S-fu56dHr50J2YwQ")
cluster_neighbor = pd.read_csv('center_neighbor.csv')
fig = px.scatter_mapbox(cluster_neighbor, lat="latitude", lon="longitude", 
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1)
fig.show()