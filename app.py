import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import geocoder
import requests
from datetime import datetime, timedelta
import io
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from streamlit_option_menu import option_menu
import base64
import plotly.express as px
from geopy.geocoders import Nominatim
import time
import random

# FIRMS API Key
MAP_KEY = '3c739594579d3445040b35fafe8d0720'  # Replace with your actual FIRMS API Key

# Set page config and apply custom CSS
st.set_page_config(page_title="Wildfire Prediction App", layout="wide")

# Custom CSS with updated colors and centered title and logo
st.markdown("""
<style>
    .stApp {
        background-color: #2F3C7E;
    }
    body, p, div, span, li, td, th, caption, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stCode {
        color: #FBEAEB !important;
        font-size: 30px !important;
    }
    .stButton>button {
        color: #FBEAEB !important;
        background-color: #3C7E2F !important;
        width: 100%;
        font-size: 30px !important;
    }
    .stButton>button:not(.sidebar .stButton>button) {
        background-color: #3C7E2F !important;
        color: #FBEAEB !important;
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        color: #FBEAEB !important;
        font-size: 30px !important;
    }
    .stSelectbox>div>div>div, .stNumberInput>div>div>input, .stTextInput>div>div>input {
        color: #2F3C7E !important;
        font-size: 30px !important;
    }
    .stDataFrame {
        color: #FBEAEB !important;
        font-size: 30px !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #2F3C7E !important;
        font-size: 30px !important;
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FBEAEB !important;
        font-size: 30px !important;
    }
    .stTextArea textarea {
        background-color: #FBEAEB;
        color: #2F3C7E !important;
        font-size: 30px !important;
    }
    .sidebar .sidebar-content {
        background-color: #505FA1 !important;
    }
    .sidebar .sidebar-content * {
        color: #FBEAEB !important;
        font-size: 30px !important;
    }
    .sidebar .stButton>button {
        background-color: #FBEAEB;
        color: #2F3C7E !important;
        width: 100%;
        font-size: 30px !important;
        margin-bottom: 10px;
        border: 2px solid #daa520 !important;
    }
    .centered-title {
        text-align: center;
        font-size: 40px !important;
        font-weight: bold;
        margin-top: 20px;
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;  # Adjust as needed
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Define StackingClassifier class
class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        for model in self.base_models_:
            model.fit(X, y)
        
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        self.meta_model_.fit(meta_features, y)
        
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('advanced_wildfire_model.joblib')
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model()

# Check if model is loaded successfully
if model is None:
    st.error("Failed to load the model. Please check the model file and try again.")
    st.stop()

# Function to get current location
def get_current_location():
    g = geocoder.ip('me')
    return g.latlng

# Function to generate mock wildfire data
def generate_mock_wildfire_data(latitude, longitude, radius_km=100):
    num_incidents = np.random.randint(5, 15)
    mock_data = []
    
    for _ in range(num_incidents):
        lat_offset = np.random.uniform(-radius_km/111, radius_km/111)
        lon_offset = np.random.uniform(-radius_km/111, radius_km/111)
        
        incident = {
            'latitude': latitude + lat_offset,
            'longitude': longitude + lon_offset,
            'incident_name': f"Mock Fire {_+1}",
            'incident_type': np.random.choice(['Wildfire', 'Prescribed Fire']),
            'final_acres': np.random.randint(10, 1000),
            'acq_date': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'acq_time': f"{np.random.randint(0, 24):02d}{np.random.randint(0, 60):02d}",
            'confidence': np.random.choice(['low', 'nominal', 'high']),
            'frp': np.random.uniform(1, 100)
        }
        mock_data.append(incident)
    
    return pd.DataFrame(mock_data)

# Function to fetch live wildfire data from NASA FIRMS
def fetch_live_wildfire_data():
    min_lat, max_lat = -80, 80
    min_lon, max_lon = -180, 180
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{min_lon},{min_lat},{max_lon},{max_lat}/{yesterday}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        
        if data.empty:
            st.info("No wildfire incidents found globally in the last 24 hours.")
            return None
        
        if 'latitude' not in data.columns and 'lat' in data.columns:
            data = data.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        
        if 'acq_date' in data.columns and 'acq_time' in data.columns:
            data['datetime'] = pd.to_datetime(data['acq_date'] + ' ' + data['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
        
        return data
    
    except requests.RequestException as e:
        st.error(f"Error fetching live data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Function to check FIRMS API transaction count
def get_transaction_count():
    url = f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}'
    try:
        df = pd.read_json(url, typ='series')
        return df['current_transactions']
    except:
        st.error("Error checking transaction count.")
        return None

# New: Data Simulation for Predictive Modeling
def simulate_wildfire_data(days=30):
    date_range = pd.date_range(end=datetime.now(), periods=days)
    data = []
    for date in date_range:
        incidents = random.randint(5, 50)
        for _ in range(incidents):
            data.append({
                'date': date,
                'latitude': random.uniform(25, 49),
                'longitude': random.uniform(-125, -65),
                'intensity': random.choice(['low', 'medium', 'high']),
                'cause': random.choice(['lightning', 'human', 'unknown']),
                'area_affected': random.uniform(0.1, 1000)
            })
    return pd.DataFrame(data)

# New: Interactive Risk Assessment Quiz
def risk_assessment_quiz():
    st.subheader("Wildfire Risk Assessment Quiz")
    score = 0
    questions = [
        {
            "question": "How often do you clear dry vegetation around your property?",
            "options": ["Weekly", "Monthly", "Yearly", "Never"],
            "scores": [3, 2, 1, 0]
        },
        {
            "question": "Do you have a fire evacuation plan?",
            "options": ["Yes, and we practice it", "Yes, but we don't practice", "No"],
            "scores": [2, 1, 0]
        },
        {
            "question": "What type of roofing material does your home have?",
            "options": ["Fire-resistant (e.g., metal, tile)", "Wood shingles", "I don't know"],
            "scores": [2, 0, 1]
        }
    ]
    
    for q in questions:
        answer = st.radio(q["question"], q["options"])
        score += q["scores"][q["options"].index(answer)]
    
    st.write(f"Your risk score: {score}/7")
    if score <= 2:
        st.error("High risk. Please take immediate actions to improve fire safety.")
    elif score <= 5:
        st.warning("Moderate risk. There's room for improvement in your fire safety measures.")
    else:
        st.success("Low risk. Great job on fire safety, but stay vigilant!")

# New: Custom Alert System
def custom_alert_system():
    st.subheader("Custom Alert System")
    alert_types = st.multiselect(
        "Select the types of alerts you want to receive:",
        ["High Temperature", "Low Humidity", "Strong Winds", "Lightning Activity"]
    )
    location = st.text_input("Enter your location (city, state):")
    if st.button("Set Alert Preferences"):
        st.session_state.alert_prefs = {
            "types": alert_types,
            "location": location
        }
        st.success("Alert preferences saved!")

# New: Wildfire Prevention Tips Generator
def prevention_tips_generator():
    tips = [
        "Create a defensible space around your home by clearing vegetation.",
        "Use fire-resistant materials for home construction and renovations.",
        "Keep gutters, roof, and outdoor areas clear of flammable debris.",
        "Develop and practice a family evacuation plan.",
        "Properly store flammable materials and chemicals.",
        "Install smoke detectors and keep fire extinguishers accessible.",
        "Be cautious with outdoor burning and follow local regulations.",
        "Teach children about fire safety and the dangers of playing with fire.",
        "Keep important documents in a fire-safe box or digital cloud storage.",
        "Stay informed about local fire conditions and warnings."
    ]
    st.subheader("Wildfire Prevention Tips")
    num_tips = st.slider("How many tips would you like?", 1, len(tips))
    selected_tips = random.sample(tips, num_tips)
    for i, tip in enumerate(selected_tips, 1):
        st.write(f"{i}. {tip}")

# New: Offline Mode with Cached Data
@st.cache_data
def get_cached_data():
    # This function would normally fetch and cache real data
    # For this example, we'll generate some mock data
    return simulate_wildfire_data(days=30)

def offline_mode():
    st.subheader("Offline Mode")
    if st.button("Load Cached Data"):
        data = get_cached_data()
        st.write("Showing cached wildfire data:")
        st.dataframe(data)
        
        # Create a map with the cached data
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        for _, row in data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Date: {row['date']}<br>Intensity: {row['intensity']}<br>Cause: {row['cause']}",
                color="red",
                fill=True,
            ).add_to(m)
        folium_static(m)

# Modify the navigation function to include new options
def navigation():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Input Features", "Wildfire Data", "Prediction Results", 
                     "Feature Importance", "Real-time Monitoring", "Risk Assessment", 
                     "Custom Alerts", "Prevention Tips", "Offline Mode", "About"],
            icons=["house", "pencil-square", "fire", "graph-up", "bar-chart", 
                   "clock-history", "question-circle", "bell", "lightbulb", "cloud-offline", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "1rem", "background-color": "#2F3C7E"},
                "icon": {"color": "#FBEAEB", "font-size": "30px"},
                "nav-link": {"font-size": "30px", "text-align": "left", "margin": "0.5rem 0", "color": "#FBEAEB", "border-radius": "5px"},
                "nav-link-selected": {"background-color": "#1E2A5E", "color": "#F2A541", "font-weight": "bold"},
            }
        )
    return selected


# Home page
def home():
    st.markdown("""
    <div class="centered-content">
        <img src="data:image/png;base64,{}" class="centered-image">
        <h1 class="centered-title">Wildfire Prediction Application</h1>
    </div>
    """.format(get_image_as_base64("wildfire_logo.png")), unsafe_allow_html=True)
    
    st.write("""
    Welcome to the Wildfire Prediction Application. This powerful tool combines advanced machine learning techniques with real-time data to predict and analyze wildfire risks across the globe.
    
    ### Key Features:
    
    1. **Predictive Analysis**: Our state-of-the-art machine learning model assesses the likelihood of high-confidence fires based on various environmental factors.
    
    2. **Real-time Data**: Access up-to-date wildfire information from NASA's FIRMS (Fire Information for Resource Management System) API.
    
    3. **Interactive Map**: Visualize current wildfire hotspots on a global map.
    
    4. **Custom Predictions**: Input your own environmental data to get personalized wildfire risk assessments.
    
    5. **Feature Importance**: Understand which factors contribute most significantly to wildfire risks.
    
    6. **Real-time Monitoring**: Keep track of evolving wildfire situations with live updates.
    
    ### How to Use:
    
    - Navigate through the app using the sidebar menu.
    - Start by entering custom input features or explore current wildfire data.
    - View prediction results and analyze feature importance to gain insights into wildfire patterns.
    - Use the real-time monitoring feature for up-to-the-minute wildfire information.
    
    ### Why It Matters:
    
    Wildfires pose significant threats to ecosystems, communities, and economies worldwide. By leveraging data-driven predictions, we aim to contribute to better preparedness and response strategies for wildfire management.
    
    Remember, while this tool provides valuable insights, always consult with local authorities and official sources for critical decision-making related to wildfires.
    
    Let's explore the world of wildfire prediction together!
    """)
    
    st.info("""
    **Did You Know?** 
    According to the National Interagency Fire Center, an average of 70,000 wildfires burn about 7 million acres of land in the United States each year.
    """)

    st.warning("""
    **Safety First**: If you're in a wildfire-prone area, always have an emergency plan ready and stay informed about local fire conditions and evacuation procedures.
    """)

# Helper function to convert image to base64
def get_image_as_base64(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Input Features page
def input_features():
    st.header("Enter Feature Values:")
    current_lat, current_lon = get_current_location()

    col1, col2, col3 = st.columns(3)

    with col1:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=current_lat)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=current_lon)
        brightness = st.number_input("Brightness", min_value=0.0, value=300.0)
        scan = st.number_input("Scan", min_value=0.0, value=1.0)

    with col2:
        track = st.number_input("Track", min_value=0.0, value=1.0)
        bright_t31 = st.number_input("Bright T31", min_value=0.0, value=290.0)
        frp = st.number_input("FRP", min_value=0.0, value=10.0)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)

    with col3:
        hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
        daynight = st.selectbox("Day/Night", options=["Day", "Night"])
    
    # Calculate derived features
    latitude_abs = abs(latitude)
    brightness_frp_ratio = brightness / (frp + 1)
    scan_track_ratio = scan / (track + 1)
    daynight_numeric = 1 if daynight == "Day" else 0

    # Create a dataframe with user inputs
    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'brightness': [brightness],
        'scan': [scan],
        'track': [track],
        'bright_t31': [bright_t31],
        'frp': [frp],
        'month': [month],
        'hour': [hour],
        'daynight_numeric': [daynight_numeric],
        'latitude_abs': [latitude_abs],
        'brightness_frp_ratio': [brightness_frp_ratio],
        'scan_track_ratio': [scan_track_ratio]
    })

    if st.button("Save Input"):
        st.session_state.input_data = input_data
        st.success("Input data saved successfully!")

# New function for time series analysis
def time_series_analysis(data):
    if 'acq_date' in data.columns:
        data['acq_date'] = pd.to_datetime(data['acq_date'])
        daily_counts = data.groupby('acq_date').size().reset_index(name='count')
        fig = px.line(daily_counts, x='acq_date', y='count', title='Daily Wildfire Incidents')
        st.plotly_chart(fig)

# New function for heatmap generation
def generate_heatmap(data):
    if 'latitude' in data.columns and 'longitude' in data.columns:
        # Create a map centered on the mean coordinates
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)

        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for index, row in data.iterrows()]
        folium.plugins.HeatMap(heat_data).add_to(m)

        # Add markers for each fire incident
        for index, row in data.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"Fire incident at {row['latitude']:.2f}, {row['longitude']:.2f}",
                icon=folium.Icon(color='red', icon='fire', prefix='fa')
            ).add_to(m)

        # Display the map
        folium_static(m)

        # Display additional plotly chart for time series
        fig = px.line(data.groupby('acq_date').size().reset_index(name='count'), 
                      x='acq_date', y='count', title='Daily Wildfire Incidents')
        st.plotly_chart(fig)
    else:
        st.error("Required columns 'latitude' and 'longitude' not found in the data.")

# New function for reverse geocoding
def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="wildfire_app")
    try:
        location = geolocator.reverse(f"{lat}, {lon}")
        return location.address
    except:
        return "Location not found"

# Enhanced wildfire data function
def enhanced_wildfire_data():
    st.header("Enhanced Wildfire Data Analysis:")
    data_source = st.radio("Choose data source:", ("Mock Data", "Live Global Data (NASA FIRMS)"))

    if st.button("Generate/Fetch Wildfire Data"):
        if data_source == "Mock Data":
            with st.spinner("Generating mock wildfire data..."):
                wildfire_data = generate_mock_wildfire_data(0, 0, radius_km=10000)
            st.success(f"Generated {len(wildfire_data)} mock wildfire incidents globally.")
        else:
            tcount = get_transaction_count()
            if tcount is not None:
                st.info(f"Current transaction count: {tcount}")
                if tcount >= 1000:
                    st.warning("Transaction limit reached. Please try again later.")
                    return
            
            with st.spinner("Fetching live global wildfire data..."):
                wildfire_data = fetch_live_wildfire_data()
            if wildfire_data is not None:
                st.success(f"Fetched {len(wildfire_data)} live wildfire incidents globally from the last 24 hours.")
            else:
                st.warning("Unable to fetch live data. Please try again later.")
                wildfire_data = generate_mock_wildfire_data(0, 0, radius_km=10000)  # Generate global mock data as fallback
                st.info(f"Generated {len(wildfire_data)} mock wildfire incidents globally as a fallback.")
        
        if wildfire_data is not None and not wildfire_data.empty:
            st.subheader("Interactive Map")
            generate_heatmap(wildfire_data)
            
            st.subheader("Time Series Analysis")
            time_series_analysis(wildfire_data)
            
            st.subheader("Top 5 Most Affected Areas")
            top_areas = wildfire_data.groupby(['latitude', 'longitude']).size().nlargest(5).reset_index(name='count')
            for _, row in top_areas.iterrows():
                address = reverse_geocode(row['latitude'], row['longitude'])
                st.write(f"Location: {address}")
                st.write(f"Number of incidents: {row['count']}")
                st.write("---")

            st.subheader("Raw Data")
            st.dataframe(wildfire_data)

# New function for real-time monitoring
def real_time_monitoring():
    st.header("Real-time Wildfire Monitoring")
    
    if st.button("Start Monitoring"):
        placeholder = st.empty()
        for i in range(100):
            with placeholder.container():
                data = fetch_live_wildfire_data()
                if data is not None:
                    st.write(f"Total active wildfires: {len(data)}")
                    generate_heatmap(data)
                    time_series_analysis(data)
                time.sleep(60)  # Update every minute
            if st.button("Stop Monitoring"):
                break

# Enhanced prediction results function
def enhanced_prediction_results():
    st.header("Enhanced Prediction Results:")
    if 'input_data' not in st.session_state:
        st.warning("Please enter input features first!")
        return

    input_data = st.session_state.input_data

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.error(f"High confidence fire predicted! (Probability: {probability:.2f})")
    else:
        st.success(f"No high confidence fire predicted. (Probability: {probability:.2f})")
    
    st.subheader("Risk Factors")
    feature_importance = pd.DataFrame({
        'feature': model.named_steps['feature_selection'].get_feature_names_out(),
        'importance': model.named_steps['classifier'].base_models_[0].feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance.head(5), x='importance', y='feature', orientation='h',
                 title='Top 5 Risk Factors')
    st.plotly_chart(fig)
    
    st.subheader("Recommended Actions")
    if prediction[0] == 1:
        st.markdown("""
        1. Alert local authorities immediately
        2. Prepare for potential evacuation
        3. Clear combustible materials around structures
        4. Stay informed about wind conditions
        5. Have an emergency kit ready
        """)
    else:
        st.markdown("""
        1. Maintain regular fire safety practices
        2. Stay informed about local fire conditions
        3. Review your emergency plan periodically
        4. Participate in community fire prevention programs
        5. Consider fire-resistant landscaping
        """)

# Feature Importance page
def feature_importance():
    st.header("Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': model.named_steps['feature_selection'].get_feature_names_out(),
        'importance': model.named_steps['classifier'].base_models_[0].feature_importances_
    }).sort_values('importance', ascending=False)

    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title='Feature Importance')
    st.plotly_chart(fig)

# About page
def about():
    st.header("About the App:")
    st.write("""
    This wildfire prediction app combines a machine learning model with both mock and live wildfire data.

    1. Prediction Model: The app uses a stacked ensemble of machine learning algorithms to predict the likelihood of a high-confidence fire based on input features.

    2. Mock Data: The app can generate mock wildfire incident data for demonstration purposes.

    3. Live Data: The app can fetch real-time fire data from the NASA FIRMS API, providing up-to-date information on active fire incidents.

    4. Real-time Monitoring: Users can monitor wildfire incidents in real-time with automatic updates.

    5. Enhanced Visualizations: The app uses interactive charts and maps for better data representation.

    Please note that while the prediction model provides an estimate based on historical data, always refer to official sources for critical decision-making related to wildfires.
    """)

    st.warning("""
    Disclaimer: This application is for educational and demonstration purposes only. 
    It should not be used as a substitute for professional wildfire detection and management systems.
    Always refer to official sources and expert opinions for real-world wildfire information and decision-making.
    """)

# Modify the main app function to include new pages
def main():
    page = navigation()

    if page == "Home":
        home()
    elif page == "Input Features":
        input_features()
    elif page == "Wildfire Data":
        enhanced_wildfire_data()
    elif page == "Prediction Results":
        enhanced_prediction_results()
    elif page == "Feature Importance":
        feature_importance()
    elif page == "Real-time Monitoring":
        real_time_monitoring()
    elif page == "Risk Assessment":
        risk_assessment_quiz()
    elif page == "Custom Alerts":
        custom_alert_system()
    elif page == "Prevention Tips":
        prevention_tips_generator()
    elif page == "Offline Mode":
        offline_mode()
    elif page == "About":
        about()

if __name__ == "__main__":
    main()