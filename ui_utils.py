import streamlit as st
import pandas as pd
import numpy as np
from data_utils import PROB_COLUMN_MAP, prepare_display_data  # Add this import
from map_utils import get_color
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
from map_utils import update_selected_marker  # Add this import

def calculate_optimal_threshold(data, selected_date):
    """Calculate threshold that maximizes F1 score for the selected date"""
    # Filter data for selected date
    date_data = data[data['date'] == selected_date]
    thresholds = np.arange(0, 1.01, 0.01)
    max_f1 = 0
    optimal_threshold = 0.5
    
    true_values = date_data[['true_1_to_5d', 'true_6_to_10d', 'true_11_to_15d']].max(axis=1)
    
    for threshold in thresholds:
        predicted = (date_data['mean_probability'] >= threshold).astype(int)
        f1 = f1_score(true_values, predicted)
        
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold
    
    return optimal_threshold

def create_prediction_distribution_chart(data, threshold):
    """Create pie charts showing prediction distributions"""
    true_values = data[['true_1_to_5d', 'true_6_to_10d', 'true_11_to_15d']].max(axis=1)
    predicted_positive = data['mean_probability'] >= threshold
    
    # Get counts for both pie charts
    true_positive = sum((predicted_positive) & (true_values == 1))
    false_positive = sum((predicted_positive) & (true_values == 0))
    false_negative = sum((~predicted_positive) & (true_values == 1))
    true_negative = sum((~predicted_positive) & (true_values == 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 6))
    fig.patch.set_facecolor('#0e1117')
    
    # First pie chart (Positive Predictions)
    total_positive = true_positive + false_positive
    if total_positive > 0:
        ax1.set_facecolor('#0e1117')
        sizes1 = [true_positive, false_positive]
        labels1 = ['Correct', 'Incorrect']
        colors1 = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes1, labels=labels1, colors=colors1,
                autopct='%1.1f%%', textprops={'color': 'white'})
        ax1.set_title('Distribution of Positive Predictions', 
                     color='white', pad=10)
    
    # Second pie chart (Actual Protests)
    total_actual = true_positive + false_negative
    if total_actual > 0:
        ax2.set_facecolor('#0e1117')
        sizes2 = [true_positive, false_negative]
        labels2 = ['Detected', 'Missed']
        colors2 = ['#2ecc71', '#f1c40f']
        ax2.pie(sizes2, labels=labels2, colors=colors2,
                autopct='%1.1f%%', textprops={'color': 'white'})
        ax2.set_title('Distribution of Actual Protests', 
                     color='white', pad=10)
    
    plt.tight_layout()
    return fig if (total_positive > 0 or total_actual > 0) else None

def setup_page():
    st.set_page_config(
        page_title="Protest Prediction Dashboard",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Protest Prediction Dashboard")
    st.markdown("""
    This dashboard shows protest probability predictions across different locations and time horizons.
    Use the sidebar controls to filter the data.
    """)

def render_sidebar_controls(merged_data):
    st.sidebar.title("Dashboard Controls")
    
    date = st.sidebar.date_input(
        "📅 Date",
        value=pd.to_datetime(sorted(merged_data['date'].unique())[0]),
        min_value=pd.to_datetime(merged_data['date'].min()),
        max_value=pd.to_datetime(merged_data['date'].max())
    )
    
    # Filter data for the selected date
    date_str = date.strftime('%Y-%m-%d')
    filtered_data = merged_data[merged_data['date'] == date_str].copy()
    
    # Initialize session state variables if not present
    if 'threshold' not in st.session_state:
        st.session_state.threshold = calculate_optimal_threshold(merged_data, date_str)
    if 'last_date' not in st.session_state:
        st.session_state.last_date = date_str
    
    # Recalculate optimal threshold if date changed
    if st.session_state.last_date != date_str:
        st.session_state.threshold = calculate_optimal_threshold(merged_data, date_str)
        st.session_state.last_date = date_str
    
    optimal_threshold = st.session_state.threshold
    
    # Create columns for threshold slider and reset button
    col1, col2 = st.sidebar.columns([4, 2], vertical_alignment='bottom')
    
    with col1:
        probability_threshold = st.slider(
            "🚩 Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold, 
            help=f"Threshold optimized for {date.strftime('%Y-%m-%d')}; adjust as needed"
        )
    
    with col2:
        if st.button("Reset", help="Reset to optimal threshold for selected date"):
            st.session_state.threshold = optimal_threshold
            probability_threshold = optimal_threshold
    
    # Update session state with current threshold
    st.session_state.threshold = probability_threshold
        
    # Separate expanders for different visualization types
    with st.sidebar.expander("📊 Distribution Analysis", expanded=False):
        pie_charts = create_prediction_distribution_chart(filtered_data, probability_threshold)
        if pie_charts:
            st.pyplot(pie_charts)
            
    with st.sidebar.expander("🧮 Confusion Matrix", expanded=False):
        render_confusion_matrix(filtered_data, date_str, probability_threshold)
    
    return date_str, probability_threshold

def handle_airport_click(map_view, airport, locations):
    """Handle click event on airport column to update map view"""
    airport_data = locations[locations['airport'] == airport]
    if airport_data.empty:
        st.warning(f"Airport {airport} not found in current data")
        return
        
    location = airport_data.iloc[0]
    lat, lon = location['ourairports_latitude'], location['ourairports_longitude']
    
    # Update marker without recreating map
    popup_content = f"<b><u>{airport}</b></u>\n Probability: <b>{location['mean_probability']:.2f}</b>"
    update_selected_marker(map_view, lat, lon, popup_content)
    
    # Update view bounds
    map_view.fit_bounds(
        [[lat - 3, lon - 3], [lat + 3, lon + 3]]
    )

def render_high_risk_locations(locations, probability_threshold, map_view):
    """Remove max_rows parameter to show all locations above threshold"""
    if isinstance(locations, str):
        st.info(locations)
        return None

    # Initialize high_risk_locations in session state
    st.session_state.high_risk_locations = locations.copy()
    
    # Use prepare_display_data without row limit
    display_df = prepare_display_data(locations)
    if display_df is None:
        return None

    # Handle dataframe display and selection
    event = st.dataframe(
        display_df,
        column_config={
            'airport': 'Airport',
            'mean_probability': st.column_config.NumberColumn(
                'Probability',
                format="%.2f"
            ),
            'views_history': st.column_config.AreaChartColumn(
                'Next 15 Days',
                y_min=probability_threshold,
                y_max=1,
            )
        },
        hide_index=True,
        height=250,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Create empty container for callout box
    details_container = st.empty()

    # Show callout box with details when row is selected
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_airport = display_df.iloc[selected_idx]['airport']
        selected_data = st.session_state.high_risk_locations[
            st.session_state.high_risk_locations['airport'] == selected_airport
        ].iloc[0]

        with details_container.container():
            st.markdown("### 📍 Location Details")
            st.write(f"**{selected_data['airport']}** (*lon: {selected_data['ourairports_latitude']:.4f}, lat: {selected_data['ourairports_longitude']:.4f}*)")
            
            col1, col2, col3 = st.columns([1.8,2,2.1], gap="small")
            
            with col1:
                st.markdown("###### <u>Timeframe</u>", unsafe_allow_html = True)
                st.write("1-5 days: ")
                st.write("6-10 days: ")
                st.write("11-15 days: ")

            with col2:
                st.markdown("###### <u>Probabilities</u>", unsafe_allow_html = True)
                st.write(f"{selected_data['pred_prob_1_to_5d']:.2f}")
                st.write(f"{selected_data['pred_prob_6_to_10d']:.2f}")
                st.write(f"{selected_data['pred_prob_11_to_15d']:.2f}")
            
            with col3:
                st.markdown("###### <u>Ground Truth</u>", unsafe_allow_html = True)
                st.write(bool(selected_data['true_1_to_5d']))
                st.write(bool(selected_data['true_6_to_10d']))
                st.write(bool(selected_data['true_11_to_15d']))

        # Update map marker
        handle_airport_click(map_view, selected_airport, st.session_state.high_risk_locations)

    return None

def render_legend():
    """Render a vertical color gradient legend"""
    # Create gradient steps
    steps = 100
    probabilities = np.linspace(0, 1, steps)
    
    # Create legend with custom HTML and CSS
    map_height = 400  # Adjust this value to match your map's height
    color_stops = [f"rgb{get_color(prob)} {(i/steps)*100}%" for i, prob in enumerate(probabilities)]
    
    legend_html = f"""
        <div style="text-align: center; width: 10px; margin-left: 0px;">
            <div>1.0</div>
        <div style="
            height: {map_height}px;
            width: 10px;
            background: linear-gradient(to top, {','.join(color_stops)});
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-left: 3px;
        "></div>
        <div style="text-align: center; width: 10px; margin-left: 0px;">
            <div>0.0</div>
    """
    
    st.markdown(legend_html, unsafe_allow_html=True)

def render_confusion_matrix(merged_data, date, probability_threshold):
    # Filter data by the selected date
    filtered_data = merged_data[merged_data['date'] == date]
    
    # Extract true values from the filtered dataset
    true_values = filtered_data[['true_1_to_5d', 'true_6_to_10d', 'true_11_to_15d']].max(axis=1)
    
    # Calculate predicted values based on the probability threshold
    predicted_values = (filtered_data['mean_probability'] >= probability_threshold).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(true_values, predicted_values, labels=[0, 1])

    # Create and plot confusion matrix with dark theme
    disp = ConfusionMatrixDisplay.from_predictions(
        true_values,
        predicted_values,
        labels=[0, 1],
        display_labels=['No Protest', 'Protest'],
        cmap='Blues',
        normalize=None,
        colorbar=False,
        include_values=True
    )
    
    # Customize plot appearance
    fig = disp.figure_
    fig.patch.set_facecolor('#0e1117')
    ax = disp.ax_
    ax.set_facecolor('#0e1117')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_title('Confusion Matrix', color='white')
   
    st.pyplot(fig)