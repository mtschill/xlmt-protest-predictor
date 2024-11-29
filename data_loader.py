import pandas as pd
import numpy as np
import os

def get_data_file_path(filename):
    """Get absolute path for data file relative to script location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')  # Create a 'data' subdirectory path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Create the directory if it doesn't exist
    return os.path.join(data_dir, filename)

def process_data(pred_file, airports_file, acled_file):
    """Process raw data files and return a consolidated DataFrame"""
    # Load prediction data
    data = pd.read_pickle(pred_file)
    data['date'] = pd.to_datetime(data['timestamp']).dt.date
    
    # Load airports data
    airports_data = pd.read_csv(airports_file)
    
    # Transform place_names from Installation_ to Airport_
    data['place_name'] = data['place_name'].str.replace('Installation_', 'Airport_')
    airports_data['place_name'] = airports_data['place_name'].str.replace('Installation_', 'Airport_')
    
    # Clean and prepare data
    required_cols = ['place_name', 'latitude', 'longitude']
    if not all(col in airports_data.columns for col in required_cols):
        raise ValueError(f"Airports file missing required columns. Required: {required_cols}")
    
    # Merge predictions with airport data
    merged_data = pd.merge(
        data,
        airports_data[['place_name', 'latitude', 'longitude']],
        on='place_name',
        how='inner'
    )
    
    # Group by airport and date
    grouped_data = merged_data.groupby(['place_name', 'date']).agg({
        'pred_prob_1_to_5d': 'mean',
        'pred_prob_6_to_10d': 'mean',
        'pred_prob_11_to_15d': 'mean',
        'true_1_to_5d': 'first',
        'true_6_to_10d': 'first',
        'true_11_to_15d': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Rename columns for consistency
    grouped_data = grouped_data.rename(columns={
        'place_name': 'airport_id',
        'latitude': 'ourairports_latitude',
        'longitude': 'ourairports_longitude'
    })
    
    grouped_data['airport'] = grouped_data['airport_id']  # Use airport_id as the display name
    
    # Compute combined probability
    grouped_data['pred_prob_1_to_15d'] = grouped_data[
        ['pred_prob_1_to_5d', 'pred_prob_6_to_10d', 'pred_prob_11_to_15d']
    ].max(axis=1)
    
    # Load and process ACLED data
    acled_data = pd.read_csv(acled_file)
    acled_data['event_date'] = pd.to_datetime(acled_data['event_date'])
    
    # Add protest indicators - only use ACLED for past protests
    grouped_data['had_recent_protest'] = grouped_data.apply(
        lambda row: check_recent_protests(row, acled_data), axis=1
    )
    
    # Use true labels for future protests instead of ACLED data
    grouped_data['had_protest_in_5d'] = grouped_data['true_1_to_5d']
    grouped_data['had_protest_in_10d'] = grouped_data['true_6_to_10d']
    grouped_data['had_protest_in_15d'] = grouped_data['true_11_to_15d']
    
    grouped_data['had_protest_in_window'] = 0  # Initialize default column
    
    if grouped_data.empty:
        raise ValueError("No data after merging.")
    
    # After processing, save the reduced dataset
    reduced_data_path = get_data_file_path('reduced_dashboard_data.parquet')
    try:
        grouped_data.to_parquet(reduced_data_path, compression='snappy')
    except Exception as e:
        print(f"Warning: Could not save reduced dataset: {str(e)}")
    
    return grouped_data

def check_recent_protests(row, acled_data):
    """Check for protests or riots in the past 5 days near the location"""
    date = pd.to_datetime(row['date'])
    location_lat = row['ourairports_latitude']
    location_lon = row['ourairports_longitude']
    
    mask = (
        (acled_data['event_date'] >= date - pd.Timedelta(days=5)) &
        (acled_data['event_date'] < date) &
        (acled_data['event_type'].isin(['Protests', 'Riots'])) &
        (np.abs(acled_data['latitude'] - location_lat) < 0.45) &
        (np.abs(acled_data['longitude'] - location_lon) < 0.45)
    )
    return int(mask.any())

def load_dashboard_data():
    """Load data for the dashboard, using cached version if available"""
    reduced_data_path = get_data_file_path('reduced_dashboard_data.parquet')
    
    # Try to load the reduced dataset first
    if os.path.exists(reduced_data_path):
        try:
            return pd.read_parquet(reduced_data_path)
        except Exception as e:
            print(f"Warning: Could not load reduced dataset: {str(e)}")
    
    # If reduced dataset is not available or couldn't be loaded, process raw data
    try:
        return process_data(
            pred_file=get_data_file_path('predictions_0.pkl'),
            airports_file=get_data_file_path('airports_oa.csv'),
            acled_file=get_data_file_path('ACLED_2016-01-01-2016-12-31_filtered.csv')
        )
    except Exception as e:
        raise Exception(f"Error loading dashboard data: {str(e)}")