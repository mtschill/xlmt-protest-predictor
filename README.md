# XLM-T Protest Predictor Dashboard

Interactive dashboard to visualize protest/riot predictions from fine-tuned XLM-T model outputs near geospatial points of interest.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xlmt-protest-predictor.git
cd xlmt-protest-predictor
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- streamlit
- folium
- streamlit-folium
- pandas
- numpy
- matplotlib
- scikit-learn
- pyarrow

## Usage

1. Ensure you have the required data files in the `data/` directory:
   - `reduced_dashboard_data.parquet` (cached dataset)
   - OR the raw input files:
     - `predictions_0.pkl` (model predictions)
     - `airports_oa.csv` (airport locations)
     - `ACLED_2016-01-01-2016-12-31_filtered.csv` (historical events)

2. Place `logo.png` in the project root directory for the dashboard header.

3. Run the Streamlit app:
```bash
streamlit run main.py
```

4. Open your browser to http://localhost:8501

## Data Sources

The dashboard can work with either:

1. **Cached Dataset** (Recommended)
   - Uses `reduced_dashboard_data.parquet`
   - Pre-processed and optimized for the dashboard
   - Much faster loading times

2. **Raw Data Processing**
   - Uses original prediction and reference files
   - Automatically creates cached version on first run
   - Slower initial load time
   - Useful for updating with new model predictions

## Screenshot

![Dashboard Screenshot](demo_dashboard_streamlit.png)

## Notes

- The dashboard will automatically detect and use the cached parquet file if available
- If no cached file exists, it will process the raw data files and create one
- Updates to raw data files will require manual deletion of the parquet file to regenerate