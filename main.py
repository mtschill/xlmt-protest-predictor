import streamlit as st
from streamlit_folium import st_folium
from data_utils import get_cached_data, get_high_risk_locations, get_data_bounds
from map_utils import get_map_view
from ui_utils import setup_page, render_sidebar_controls, render_high_risk_locations, render_legend

def main():
    setup_page()

    # Load data with caching
    merged_data = get_cached_data()
    if merged_data is None:
        st.stop()
    st.session_state.merged_data = merged_data

    # Sidebar controls - moved back to main
    date, probability_threshold = render_sidebar_controls(merged_data)
    st.session_state.date = date
    st.session_state.probability_threshold = probability_threshold

    # Call the fragment function with control values
    render_map_content()

    # Add footer
    st.markdown("---")
    st.markdown("M. Trent Schill | NPS Thesis Project | 2024")

@st.fragment
def render_map_content():
    merged_data = st.session_state.merged_data
    date = st.session_state.date
    probability_threshold = st.session_state.probability_threshold
    map_bounds = get_data_bounds(merged_data)

    # Remove spinner wrapper, keep core functionality
    high_risk_locations = get_high_risk_locations(merged_data, date, probability_threshold, map_bounds)

    # Generate map view
    map_view = get_map_view(merged_data, date, probability_threshold)
    if map_view is None:
        st.error("Unable to display map. Please check if there is data available for the selected filters.")
        return

    col1, col2, col3 = st.columns([1.25, 3, 0.25])

    with col1:
        st.subheader("⚠️ Risk of Protest or Riot", help="Based on probability threshold, risk of protest or riot in the next 15 days")
        render_high_risk_locations(high_risk_locations, probability_threshold, map_view)
        # Removed render_confusion_matrix call from here

    with col2:
        st.subheader("🪧 Protest Probability Map")
        try:
            map_key = f"map_{date}_{probability_threshold}"
            with st.form(key='map_form'):
                map_state = st_folium(
                    map_view,
                    height=600,
                    width='100%',
                    key=map_key
                )
                # Form submit button (hidden but required)
                st.form_submit_button(label='', type='secondary', use_container_width=False)
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")

    with col3:
        st.markdown("<br>" * 4, unsafe_allow_html=True)  # Add some spacing
        render_legend()

if __name__ == "__main__":
    main()
