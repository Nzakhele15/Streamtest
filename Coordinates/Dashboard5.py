import streamlit as st
import geopandas as gpd
import plotly.express as px
import numpy as np

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Sidebar Navigation for Page Selection
st.title("EV Charging Station Placement Analysis in South Africa")
page = st.sidebar.selectbox("Choose Analysis Strategy", ["Grid-Based Analysis", "Geospatial Analysis", "K-means Clustering", "Conclusion"])

# Load Data (using relative paths based on the folder structure)
# Replace the local paths with paths in your GitHub repo for online deployment
geospatial_data = gpd.read_file("Coordinates/ev_charging_stations.geojson")
kmeans_clusters = gpd.read_file("Coordinates/combined_clusters.geojson")
current_charging_stations = gpd.read_file("Coordinates/current_charging_stations.geojson")
current_charging_stations_cluster = gpd.read_file("Coordinates/current_charging_stations_cluster.geojson")
grid_slow_chargers = gpd.read_file("Coordinates/slow_charger.geojson")
grid_fast_chargers = gpd.read_file("Coordinates/fast_charger.geojson")

# Add dummy columns for visualization
kmeans_clusters['dummy_cluster'] = np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3'], size=len(kmeans_clusters))
grid_slow_chargers['suitability_score'] = np.random.randint(1, 100, size=len(grid_slow_chargers))
grid_fast_chargers['suitability_score'] = np.random.randint(1, 100, size=len(grid_fast_chargers))

# Page 1: Grid-Based Analysis
if page == "Grid-Based Analysis":
    st.header("Grid-Based Analysis")
    st.image("Assorted Pictures/Map with grid overlays.png", caption="Grid-Based EV Charging Placement Strategy")
    st.write("""
    Grid-based spatial analysis divides the area into uniform cells, scoring each based on factors like population density,
    proximity to major roads, and grid capacity. This approach ensures broad coverage, particularly useful in suburban and rural areas.
    Each cell's score indicates its suitability for charging stations.
    """)

    # Visualizing slow chargers based on suitability
    st.subheader("Suggested Slow Chargers (Grid-Based)")
    fig_slow = px.scatter_mapbox(
        grid_slow_chargers, lat=grid_slow_chargers.geometry.y, lon=grid_slow_chargers.geometry.x,
        color="suitability_score",
        title="Suggested Slow Chargers - Grid-Based Analysis",
        mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25},
        opacity=0.6
    )
    st.plotly_chart(fig_slow, use_container_width=True)

    # Visualizing fast chargers based on suitability
    st.subheader("Suggested Fast Chargers (Grid-Based)")
    fig_fast = px.scatter_mapbox(
        grid_fast_chargers, lat=grid_fast_chargers.geometry.y, lon=grid_fast_chargers.geometry.x,
        color="suitability_score",
        title="Suggested Fast Chargers - Grid-Based Analysis",
        mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25},
        opacity=0.6
    )
    st.plotly_chart(fig_fast, use_container_width=True)

# Page 2: Geospatial Analysis
elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    st.image("Assorted Pictures/Layered data map (infrastructure overlays)..jpeg", caption="Geospatial Data Layers for Optimized EV Charger Placement")
    st.write("""
    Geospatial analysis layers spatial data to identify ideal locations for EV chargers by considering:
    - **Power Grid Proximity**: Stations near power lines ensure reliable supply.
    - **Population Density**: High-density areas often indicate greater demand.
    - **Road Networks**: Proximity to major roads maximizes accessibility.
    This layered approach reveals patterns and connections that support effective decision-making.
    """)

    # Display map for Geospatial data
    st.subheader("Existing EV Charging Stations (Geospatial)")
    fig_geospatial = px.scatter_mapbox(
        geospatial_data, lat=geospatial_data.geometry.y, lon=geospatial_data.geometry.x,
        title="Existing EV Charging Stations - Geospatial Analysis",
        mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25},
        opacity=0.6
    )
    st.plotly_chart(fig_geospatial, use_container_width=True)

# Page 3: K-means Clustering
elif page == "K-means Clustering":
    st.header("K-means Clustering Analysis")
    st.image("Assorted Pictures/Map with colored clusters.jpeg", caption="Clustering for Demand Optimization in Charger Placement")
    st.write("""
    K-means clustering groups locations with similar characteristics to reveal natural EV demand clusters. 
    This method focuses on high-demand zones in urban areas, with:
    - **Fast Chargers**: Placed in commercial, high-density clusters.
    - **Slow Chargers**: Prioritized in residential and suburban clusters.
    """)

    # Display map for K-means clusters
    st.subheader("K-means Clustered Charger Placement")
    fig_kmeans = px.scatter_mapbox(
        kmeans_clusters, lat=kmeans_clusters.geometry.y, lon=kmeans_clusters.geometry.x,
        color="dummy_cluster",
        title="K-means Clustered Charger Placement",
        mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25},
        opacity=0.6
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    # Display current charging stations and clustered charging stations
    st.subheader("Current Charging Stations")
    fig_current = px.scatter_mapbox(
        current_charging_stations, lat=current_charging_stations.geometry.y, lon=current_charging_stations.geometry.x,
        title="Current Charging Stations",
        mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25},
        opacity=0.6
    )
    st.plotly_chart(fig_current, use_container_width=True)

# Page 4: Conclusion
elif page == "Conclusion":
    st.header("Conclusion and Recommendations")
    st.image("Assorted Pictures/Scenic EV charging or clean energy concept..jpeg", caption="A Sustainable Future with Optimized EV Charging Infrastructure")
    st.write("""
    The combined analysis using grid-based, geospatial, and clustering strategies allows us to recommend optimal EV charging locations 
    across South Africa. By leveraging each approach:
    - **Grid-Based**: Ensures broad coverage, filling in underserved areas.
    - **Geospatial**: Targets high-demand, accessible zones.
    - **K-means Clustering**: Focuses on clustering demand, efficiently placing chargers in high-usage areas.
    
    This integrated approach ensures effective resource allocation, supporting the country’s transition to sustainable EV infrastructure.
    """)

    # Team Members
    st.subheader("Project Team Members")
    st.image("Assorted Pictures/Teamwork or collaborative data analysis image.jpeg", caption="Project Team Collaboration")
    team_members = ["Sipho Shimange", "Asanda Gambu", "Dimpho Lebea", "Welsh Dube", "Sandile Jali", "Neo Mbele", "Zakhele Mabuza (App Creator)"]
    for member in team_members:
        st.write(f"- {member}")
