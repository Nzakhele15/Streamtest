import streamlit as st
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Sidebar Navigation for Page Selection
st.title("EV Charging Station Placement Analysis in South Africa")
page = st.sidebar.selectbox("Choose Analysis Strategy", ["Grid-Based Analysis", "Geospatial Analysis", "K-means Clustering", "Conclusion"])

# Load Data (replace paths with your actual data file paths)
power_grid = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/power_grid.geojson")
roads = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/roads.geojson")
buildings = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/buildings.geojson")
population_density = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/med_density.geojson")  # Use appropriate population file
current_charging_stations = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/current_charging_stations.geojson")
suggested_fast_chargers = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/fast_charger.shp")
suggested_slow_chargers = gpd.read_file("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Coordinates/slow_charger.shp")

# Add dummy columns for testing (replace with actual data columns if available)
suggested_fast_chargers['dummy_score'] = np.random.randint(1, 100, size=len(suggested_fast_chargers))
buildings['dummy_cluster'] = np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3'], size=len(buildings))

# Data overview for all maps
datasets = [
    ("Buildings", buildings),
    ("Population Density", population_density),
    ("Road Network", roads),
    ("Electricity Grid", power_grid),
    ("Current Charging Stations", current_charging_stations),
    ("Suggested Fast Chargers", suggested_fast_chargers),
    ("Suggested Slow Chargers", suggested_slow_chargers)
]

# Page 1: Grid-Based Analysis
if page == "Grid-Based Analysis":
    st.header("Grid-Based Analysis")
    st.image("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Assorted Pictures/Map with grid overlays.png", caption="Grid-Based EV Charging Placement Strategy")
    st.write("""
    Grid-based spatial analysis divides the area into uniform cells, scoring each based on factors like population density,
    proximity to major roads, and grid capacity. This approach ensures broad coverage, particularly useful in suburban and rural areas.
    Each cell's score indicates its suitability for charging stations.
    """)

    # Display maps for each dataset with grid-based context
    for name, data in datasets:
        st.subheader(f"{name} Map")
        fig = px.scatter_mapbox(
            data, lat=data.geometry.y, lon=data.geometry.x,
            color="dummy_score" if name in ["Suggested Fast Chargers", "Suggested Slow Chargers"] else None,
            title=f"{name} - Grid-Based Analysis",
            mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25}
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 2: Geospatial Analysis
elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    st.image("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Assorted Pictures/Layered data map (infrastructure overlays)..jpeg", caption="Geospatial Data Layers for Optimized EV Charger Placement")
    st.write("""
    Geospatial analysis layers spatial data to identify ideal locations for EV chargers by considering:
    - **Power Grid Proximity**: Stations near power lines ensure reliable supply.
    - **Population Density**: High-density areas often indicate greater demand.
    - **Road Networks**: Proximity to major roads maximizes accessibility.
    This layered approach reveals patterns and connections that support effective decision-making.
    """)

    # Display maps for each dataset with geospatial context
    for name, data in datasets:
        st.subheader(f"{name} Map")
        fig = px.scatter_mapbox(
            data, lat=data.geometry.y, lon=data.geometry.x,
            color="dummy_cluster" if name == "Buildings" else None,
            title=f"{name} - Geospatial Analysis",
            mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25}
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 3: K-means Clustering
elif page == "K-means Clustering":
    st.header("K-means Clustering Analysis")
    st.image("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Assorted Pictures/Map with colored clusters.jpeg", caption="Clustering for Demand Optimization in Charger Placement")
    st.write("""
    K-means clustering groups locations with similar characteristics to reveal natural EV demand clusters. 
    This method focuses on high-demand zones in urban areas, with:
    - **Fast Chargers**: Placed in commercial, high-density clusters.
    - **Slow Chargers**: Prioritized in residential and suburban clusters.
    """)
    
    # Display maps for each dataset with clustering context
    for name, data in datasets:
        st.subheader(f"{name} Map")
        fig = px.scatter_mapbox(
            data, lat=data.geometry.y, lon=data.geometry.x,
            color="dummy_cluster" if name == "Buildings" else None,
            title=f"{name} - K-means Clustering",
            mapbox_style="open-street-map", zoom=5, center={"lat": -30, "lon": 25}
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 4: Conclusion
elif page == "Conclusion":
    st.header("Conclusion and Recommendations")
    st.image("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Assorted Pictures/Scenic EV charging or clean energy concept..jpeg", caption="A Sustainable Future with Optimized EV Charging Infrastructure")
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
    st.image("C:/Users/Zakhele/Downloads/Compressed/pickled_dfs/Assorted Pictures/Teamwork or collaborative data analysis image.jpeg", caption="Project Team Collaboration")
    team_members = ["Sipho Shimange", "Asanda Gambu", "Dimpho Lebea", "Welsh Dube", "Sandile Jali", "Neo Mbele", "Zakhele Mabuza (App Creator)"]
    for member in team_members:
        st.write(f"- {member}")
