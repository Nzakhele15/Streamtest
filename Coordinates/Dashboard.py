import streamlit as st
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import Point

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Title and Project Introduction
st.title("Interactive EV Charging Station Placement Analysis in South Africa")
st.header("Project Overview")
st.write("""
This dashboard provides a comprehensive analysis for optimal EV charging station placement in South Africa,
using three main strategies: Grid-based, Geospatial, and K-means clustering. The goal is to identify ideal 
locations for fast and slow chargers based on factors such as population density, power grid accessibility, 
and road networks. Use the interactive maps below to explore each strategy.
""")

# Load Data (replace paths with your actual data file paths)
power_grid = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/power_grid.geojson")
roads = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/roads.geojson")
buildings = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/buildings.geojson")
current_charging_stations = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/current_charging_stations.geojson")
suggested_fast_chargers = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/fast_charger.shp")
suggested_slow_chargers = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/slow_charger.shp")

# Adding a dummy score column for demonstration (use actual score column when available)
suggested_fast_chargers['dummy_score'] = np.random.randint(1, 100, size=len(suggested_fast_chargers))

# Section: Grid-Based Analysis
st.header("1. Grid-Based Analysis")
st.write("""
In this approach, the area is divided into a grid of cells, each scored for charger suitability based on 
population density, proximity to roads, and grid power availability. Higher scores indicate higher suitability 
for EV chargers.
""")

fig_grid = px.choropleth_mapbox(
    suggested_fast_chargers,
    geojson=suggested_fast_chargers.geometry,
    color="dummy_score",  # Use actual suitability score column when available
    title="Grid-Based Analysis: Charger Suitability Scores",
    mapbox_style="open-street-map",
    zoom=5,
    center={"lat": -30, "lon": 25},
    opacity=0.5
)
st.plotly_chart(fig_grid, use_container_width=True)

# Section: Geospatial Analysis with Layered Data
st.header("2. Geospatial Analysis")
st.write("""
This analysis layers multiple datasets, including road networks, power grid proximity, and population density, 
to assess suitability for charging stations. Layers are added progressively for a clear visualization of how 
overlapping factors contribute to the charger placement strategy.
""")

# Base Map with OpenStreetMap style
fig_geo = go.Figure()

# Add Road Network
fig_geo.add_trace(go.Scattermapbox(
    lat=roads.geometry.y,
    lon=roads.geometry.x,
    mode="lines",
    line=dict(width=1, color="blue"),
    name="Road Network"
))

# Add Power Grid
fig_geo.add_trace(go.Scattermapbox(
    lat=power_grid.geometry.y,
    lon=power_grid.geometry.x,
    mode="lines",
    line=dict(width=1, color="orange"),
    name="Power Grid"
))

# Add Population Density (if applicable) and Buildings as Points
fig_geo.add_trace(go.Scattermapbox(
    lat=buildings.geometry.y,
    lon=buildings.geometry.x,
    mode="markers",
    marker=go.scattermapbox.Marker(size=5, color="green"),
    name="Building Density"
))

# Configure Map Layout
fig_geo.update_layout(
    mapbox=dict(style="open-street-map", zoom=5, center={"lat": -30, "lon": 25}),
    title="Geospatial Analysis with Road Network and Power Grid Layers",
    height=600
)
st.plotly_chart(fig_geo, use_container_width=True)

# Section: K-means Clustering
st.header("3. K-means Clustering Analysis")
st.write("""
K-means clustering groups regions with similar characteristics, helping identify areas where fast or slow chargers 
are most suitable. High-density commercial clusters are prioritized for fast chargers, while residential clusters 
are suited for slow chargers.
""")

# Replace this with actual cluster data if available
fig_kmeans = px.scatter_mapbox(
    buildings,  # Replace with actual dataset containing cluster information
    lat=buildings.geometry.y,
    lon=buildings.geometry.x,
    color="cluster_label",  # Use the correct cluster label column
    title="K-means Clustering: Charger Type Suitability",
    mapbox_style="open-street-map",
    zoom=5,
    center={"lat": -30, "lon": 25}
)
st.plotly_chart(fig_kmeans, use_container_width=True)

# Final Recommendations Section
st.header("Final Recommendations for New Charging Stations")
st.write("""
The map below shows proposed sites for new EV charging stations, with separate recommendations for fast and slow chargers.
These recommendations are based on the combined insights from the grid-based, geospatial, and K-means clustering analyses.
""")

fig_recommend = go.Figure()

# Suggested Slow Chargers
fig_recommend.add_trace(go.Scattermapbox(
    lat=suggested_slow_chargers.geometry.y,
    lon=suggested_slow_chargers.geometry.x,
    mode="markers",
    marker=go.scattermapbox.Marker(size=10, color="blue"),
    name="Suggested Slow Chargers"
))

# Suggested Fast Chargers
fig_recommend.add_trace(go.Scattermapbox(
    lat=suggested_fast_chargers.geometry.y,
    lon=suggested_fast_chargers.geometry.x,
    mode="markers",
    marker=go.scattermapbox.Marker(size=10, color="red"),
    name="Suggested Fast Chargers"
))

# Configure Map Layout
fig_recommend.update_layout(
    mapbox=dict(style="open-street-map", zoom=5, center={"lat": -30, "lon": 25}),
    title="Proposed Locations for New EV Charging Stations",
    height=600
)
st.plotly_chart(fig_recommend, use_container_width=True)
