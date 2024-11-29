import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from data_utils import get_mean_probability, get_data_bounds

def get_color(probability):
    """Get color based on probability value using a colormap"""
    cmap = plt.get_cmap('inferno')
    rgba = cmap(probability)
    return tuple(int(255 * c) for c in rgba[:3])

def get_size(probability):
    """Get constant size for all markers"""
    return 5  # Constant size for all markers

@st.cache_data
def create_base_map(center, zoom_start=3):
    """Create and cache the base map"""
    return folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='Cartodb dark_matter'
    )

def update_selected_marker(m, lat, lon, popup_content):
    """Update selected marker without recreating the map"""
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='red', icon='info-sign'),
        popup=popup_content,
        tooltip="Selected Airport"
    ).add_to(m)

@st.cache_data
def get_map_points(filtered_data):
    """Generate and cache map points"""
    points = []
    for _, row in filtered_data.iterrows():
        color = get_color(row['mean_probability'])
        size = get_size(row['mean_probability'])
        point = {
            'location': [row['ourairports_latitude'], row['ourairports_longitude']],
            'radius': size,
            'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            'popup': f"<b><u>{row['airport']}</b></u>\n Probability: <b>{row['mean_probability']:.2f}</b>",
        }
        points.append(point)
    return points

@st.cache_data
def get_map_view(merged_data, date, probability_threshold):
    """Generate a folium map view with probability markers"""
    try:
        # Convert date strings to datetime for comparison
        if 'date' not in merged_data.columns:
            st.error("Missing 'date' column in data")
            return None
            
        merged_data = merged_data.copy()
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        date = pd.to_datetime(date)
        
        # Filter by date
        filtered_data = merged_data[merged_data['date'] == date].copy()
        if filtered_data.empty:
            st.warning(f"No data available for date: {date.strftime('%Y-%m-%d')}")
            return None
        
        # Calculate mean probability across all horizons
        filtered_data['mean_probability'] = filtered_data.apply(
            lambda row: get_mean_probability(row), axis=1
        )
        
        # Sort the filtered data once
        filtered_data = filtered_data.sort_values('mean_probability', ascending=False)
        
        # Get bounds for the map
        lat_min, lat_max, lon_min, lon_max = get_data_bounds(merged_data)
        center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
        
        # Create base map
        m = create_base_map(center)
        
        # Add points directly to map
        for _, row in filtered_data.iterrows():
            color = get_color(row['mean_probability'])
            size = get_size(row['mean_probability'])
            folium.CircleMarker(
                location=[row['ourairports_latitude'], row['ourairports_longitude']],
                radius=size,
                stroke=True,
                color='white',
                weight=0.5,
                fill=True,
                fill_color=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
                fill_opacity=.8,
                popup=f"<b><u>{row['airport']}</b></u>\n Probability: <b>{row['mean_probability']:.2f}</b>",
                tooltip="Probability Marker"
            ).add_to(m)

        return m
        
    except Exception as e:
        st.error(f"Error creating map view: {str(e)}")
        return None

def zoom_to_location(m, lat, lon, zoom_level=8):
    """Zoom the map to the specified location"""
    m.location = [lat, lon]
    m.zoom_start = zoom_level
    # Force the map to recenter
    m.options['center'] = [lat, lon]
    m.options['zoom'] = zoom_level
    return m