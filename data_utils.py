import pandas as pd
import numpy as np
import streamlit as st
from data_loader import load_dashboard_data

PROB_COLUMN_MAP = {
    '1-5 days': ('pred_prob_1_to_5d', 'pred_prob_1_5d', 'probability_1_5d'),
    '6-10 days': ('pred_prob_6_to_10d', 'pred_prob_6_10d', 'probability_6_10d'),
    '11-15 days': ('pred_prob_11_to_15d', 'pred_prob_11_15d', 'probability_11_15d')
}

@st.cache_data
def get_cached_data():
    """Load and cache the dashboard data"""
    global PROB_COLUMN_MAP

    try:
        with st.spinner('Loading data...'):
            data = load_dashboard_data()
            if data is None or data.empty:
                st.error("No data loaded")
                return None

            # Validate and find actual probability columns
            actual_prob_columns = {}
            for horizon, possible_cols in PROB_COLUMN_MAP.items():
                found_col = next((col for col in possible_cols if col in data.columns), None)
                if found_col:
                    actual_prob_columns[horizon] = found_col
                    data[found_col] = data[found_col].astype(float)
                else:
                    st.warning(f"No probability column found for horizon {horizon}. Checked: {', '.join(possible_cols)}")
            
            if not actual_prob_columns:
                st.error("No probability columns found in the data")
                return None
                
            # Update global mapping with found columns
            PROB_COLUMN_MAP = actual_prob_columns
            
            # Convert date column to datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            else:
                st.error("Missing 'date' column in data")
                return None

            # Calculate mean probability across all horizons
            data['mean_probability'] = data.apply(lambda row: get_mean_probability(row), axis=1)

            return data
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_data_bounds(data):
    if 'ourairports_latitude' not in data.columns or 'ourairports_longitude' not in data.columns:
        st.error("Missing latitude or longitude columns in data")
        return [0, 0, 0, 0]
    return [
        data['ourairports_latitude'].min(),
        data['ourairports_latitude'].max(),
        data['ourairports_longitude'].min(),
        data['ourairports_longitude'].max()
    ]

def get_max_probability(row, selected_horizons):
    """Calculate maximum probability across selected time horizons"""
    probs = []
    for h in selected_horizons:
        col = PROB_COLUMN_MAP.get(h)
        if col and col in row:
            probs.append(row[col])
    return max(probs) if probs else 0

def get_mean_probability(row):
    """Calculate mean probability across all time horizons"""
    probs = []
    for cols in PROB_COLUMN_MAP.values():
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            if col in row:
                probs.append(row[col])
    return np.mean(probs) if probs else 0

def get_high_risk_locations(merged_data, date, probability_threshold, map_bounds):
    if 'date' not in merged_data.columns:
        st.error("Missing 'date' column in data")
        return f"No locations with probability ≥ {probability_threshold}"

    # Create a copy to avoid warnings
    filtered = merged_data[merged_data['date'] == date].copy()
    
    # Calculate mean probability first
    filtered['mean_probability'] = filtered.apply(
        lambda row: get_mean_probability(row), axis=1
    )
    
    # Filter based on mean probability threshold
    filtered = filtered[filtered['mean_probability'] >= probability_threshold]
    
    # Filter by map bounds
    lat_min, lat_max, lon_min, lon_max = map_bounds
    filtered = filtered[
        (filtered['ourairports_latitude'] >= lat_min) & 
        (filtered['ourairports_latitude'] <= lat_max) & 
        (filtered['ourairports_longitude'] >= lon_min) & 
        (filtered['ourairports_longitude'] <= lon_max)
    ]
    
    if filtered.empty:
        return f"No locations with probability ≥ {probability_threshold}"

    # Flatten PROB_COLUMN_MAP values and filter existing columns
    prob_columns = [col for cols in PROB_COLUMN_MAP.values() for col in (cols if isinstance(cols, tuple) else [cols])]
    prob_columns = [col for col in prob_columns if col in filtered.columns]

    # Keep only required columns plus coordinates for mapping
    locations = filtered[['airport', 'mean_probability', 'ourairports_latitude', 'ourairports_longitude', 'true_1_to_5d', 'true_6_to_10d', 'true_11_to_15d'] + prob_columns].copy()
    locations = locations.sort_values('mean_probability', ascending=False)
    
    return locations

def prepare_display_data(locations):
    """Prepare display data for the dataframe - show all rows"""
    if isinstance(locations, str):
        return None
        
    locations = locations.copy()
    locations['views_history'] = locations.apply(
        lambda row: get_ordered_probabilities(row), axis=1
    )
    
    # Sort but don't limit rows
    display_df = locations.sort_values('mean_probability', ascending=False)[
        ['airport', 'mean_probability', 'views_history']
    ]
    return display_df

def get_ordered_probabilities(row):
    """Get probabilities in order: 1-5d, 6-10d, 11-15d"""
    probs = []
    time_windows = ['1-5 days', '6-10 days', '11-15 days']
    
    for window in time_windows:
        cols = PROB_COLUMN_MAP.get(window, [])
        if isinstance(cols, str):
            cols = [cols]
        
        prob = next((row[col] for col in cols if col in row), 0)
        probs.append(prob)
        
    return probs