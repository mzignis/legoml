import streamlit as st
import os
import glob
import time
import json
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go

# Signal file location
UPDATE_SIGNAL_FILE = Path("/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1/.dashboard_update_signal")

st.set_page_config(
    page_title="Image Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS styling
dark_theme_css = """
<style>
    /* Main app styling */
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    /* Card styling */
    .dashboard-card {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #404040;
        margin-bottom: 20px;
        height: 100%;
    }

    .dashboard-card h3 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .dashboard-card h4 {
        color: #b3b3b3;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #3d3d3d;
        border: 1px solid #505050;
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
    }

    [data-testid="metric-container"] > div {
        color: #ffffff;
    }

    /* Text styling */
    .stMarkdown {
        color: #ffffff;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2d2d2d;
    }

    /* Data editor styling */
    .stDataFrame {
        background-color: #2d2d2d;
    }

    /* Chart container */
    .chart-container {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 10px;
    }

    /* Header styling */
    .main-header {
        background-color: #2d2d2d;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #404040;
    }

    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
    }

    .main-header p {
        color: #b3b3b3;
        margin: 5px 0 0 0;
        font-size: 0.9rem;
    }

    /* Image container styling */
    .image-container {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

def check_for_signal_file():
    """Simple check for signal file - returns (has_update, signal_info)"""
    if UPDATE_SIGNAL_FILE.exists():
        try:
            mtime = UPDATE_SIGNAL_FILE.stat().st_mtime
            time_str = time.strftime('%H:%M:%S', time.localtime(mtime))
            UPDATE_SIGNAL_FILE.unlink()  # Delete signal file
            return True, {'timestamp': mtime, 'time_str': time_str}
        except Exception as e:
            st.sidebar.error(f"Signal file error: {e}")
            return False, None
    return False, None

def load_real_time_data():
    """Load real-time prediction data from JSON file"""
    data_file = Path("/home/candfpi4b/fresh_repo/dashboard_data.json")
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            return None
    return None

def initialize_class_counters():
    """Initialize counters for all brick classes"""
    class_names = [
        'white_1x3_good', 'white_2x2_good', 'white_2x4_good',
        'blue_2x2_good', 'blue_2x6_good', 'blue_1x6_good',
        'white_1x3_damaged', 'white_2x2_damaged', 'white_2x4_damaged',
        'blue_2x2_damaged', 'blue_2x6_damaged', 'blue_1x6_damaged'
    ]
    
    if 'class_counters' not in st.session_state:
        st.session_state.class_counters = {class_name: 0 for class_name in class_names}
    
    return st.session_state.class_counters

def update_class_counters(real_time_data):
    """Update class counters with new prediction data"""
    if not real_time_data or 'classes' not in real_time_data or 'confidences' not in real_time_data:
        return False, None
    
    classes = real_time_data['classes']
    confidences = real_time_data['confidences']
    
    if len(classes) != len(confidences) or len(classes) == 0:
        return False, None
    
    max_confidence_index = confidences.index(max(confidences))
    predicted_class = classes[max_confidence_index]
    max_confidence = confidences[max_confidence_index]
    
    current_timestamp = real_time_data.get('timestamp', '')
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = ''
    
    if current_timestamp != st.session_state.last_processed_timestamp:
        counters = initialize_class_counters()
        
        if predicted_class in counters:
            st.session_state.class_counters[predicted_class] += 1
            st.session_state.last_processed_timestamp = current_timestamp
            return True, {'class': predicted_class, 'confidence': max_confidence}
    
    return False, None

def get_metric_values():
    """Get current counter values organized by category"""
    counters = initialize_class_counters()
    
    undamaged_counts = [
        counters['white_1x3_good'], counters['white_2x2_good'], counters['white_2x4_good'],
        counters['blue_2x2_good'], counters['blue_1x6_good'], counters['blue_2x6_good']
    ]
    
    damaged_counts = [
        counters['white_1x3_damaged'], counters['white_2x2_damaged'], counters['white_2x4_damaged'],
        counters['blue_2x2_damaged'], counters['blue_1x6_damaged'], counters['blue_2x6_damaged']
    ]
    
    return undamaged_counts, damaged_counts

def load_latest_snapshots(limit=5):
    """Load the most recent snapshot images"""
    snapshots_dir = Path("../../snapshots")
    if snapshots_dir.exists():
        image_files = sorted(
            glob.glob(str(snapshots_dir / "*.JPG")),
            key=lambda x: Path(x).stat().st_mtime,
            reverse=True
        )
        return image_files[:limit]
    return []

def create_predictions_data(real_time_data):
    """Create dictionary for predictions chart from real-time data"""
    if real_time_data is None:
        return {
            'categories': ['No Data', 'Available', 'Please Check', 'JSON File', 'Connection'],
            'confidences': [20, 20, 20, 20, 20]
        }
    
    try:
        classes = real_time_data.get('classes', [])
        confidences = real_time_data.get('confidences', [])
        
        if len(classes) != 4 or len(confidences) != 4:
            raise ValueError("Expected 4 classes and 4 confidences")
        
        if all(c <= 1.0 for c in confidences):
            confidences_pct = [c * 100 for c in confidences]
        else:
            confidences_pct = confidences
        
        total_confidence = sum(confidences_pct)
        other_confidence = max(0, 100 - total_confidence)
        
        categories = classes + ['Other']
        confidence_values = confidences_pct + [other_confidence]
        
        return {
            'categories': categories,
            'confidences': confidence_values
        }
    
    except Exception as e:
        st.error(f"Error processing real-time data: {e}")
        return {
            'categories': ['Error', 'Loading', 'Data', 'Please', 'Retry'],
            'confidences': [20, 20, 20, 20, 20]
        }


def create_card_header(title, subtitle=None):
    """Create a styled card header"""
    if subtitle:
        return f"""
        <div class="dashboard-card">
            <h3>{title}</h3>
            <h4>{subtitle}</h4>
        """
    else:
        return f"""
        <div class="dashboard-card">
            <h3>{title}</h3>
        """

def close_card():
    """Close the card div"""
    return "</div>"



def amount_metric(state, num):
    """Display brick count metrics in a grid"""
    st.markdown(create_card_header(state), unsafe_allow_html=True)

    # Create 2 rows of 3 columns each
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="W 1x3", value=num[0], border=True)
    with col2:
        st.metric(label="W 2x2", value=num[1], border=True)
    with col3:
        st.metric(label="W 2x4", value=num[2], border=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(label="B 2x2", value=num[3], border=True)
    with col5:
        st.metric(label="B 1x6", value=num[4], border=True)
    with col6:
        st.metric(label="B 2x6", value=num[5], border=True)

    st.markdown(close_card(), unsafe_allow_html=True)

def parameter_metrics(parameters, label): 
    """Display parameter metrics in a card"""
    st.markdown(create_card_header(label), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Metric 1", value=parameters['A'], border=True)
    with col2:
        st.metric(label="Metric 2", value=parameters['B'], border=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric(label="Metric 3", value=parameters['C'], border=True)
    with col4:
        st.metric(label="Metric 4", value=parameters['D'], border=True)
    with col5:
        st.metric(label="Metric 5", value=parameters['E'], border=True)

    st.markdown(close_card(), unsafe_allow_html=True)

def predictions(data_dict):
    """Create a horizontal bar chart with dark theme"""
    st.markdown(create_card_header("Top Categories", "Confidence Distribution"), unsafe_allow_html=True)

    with st.container():
        categories = data_dict['categories']
        confidences = data_dict['confidences']
        
        fig = go.Figure(data=[
            go.Bar(
                y=categories,
                x=confidences,
                orientation='h',
                marker_color='#4a9eff',
                text=[round(conf, 1) for conf in confidences],
                texttemplate='%{text}%',
                textposition='inside',
                textfont=dict(color='blue', size=12)
            )
        ])
        
        fig.update_layout(
            height=300,
            xaxis_title="Confidence (%)",
            yaxis_title="Top Categories",
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=1, range=[0, 100]),
            yaxis=dict(showgrid=False,color='white',  autorange='reversed')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(close_card(), unsafe_allow_html=True)

def extract_confidence_from_filename(filepath):
    """Extract confidence value from filename ending with conf{X.XXX}.jpg"""
    try:
        filename = Path(filepath).name
        if "conf" in filename and filename.endswith(".jpg"):
            conf_part = filename.split("conf")[-1].replace(".jpg", "")
            confidence = float(conf_part)
            return confidence * 100 if confidence <= 1.0 else confidence
    except (ValueError, IndexError):
        pass
    return None

def display_image_vertical_with_metrics(image_paths):    
    """Display images vertically with confidence metrics"""
    st.markdown(create_card_header("Recent Snapshots", "Latest images"), unsafe_allow_html=True)
    placeholder_path = "placeholder.jpg"
    if not image_paths:
        st.warning("No JPG images found in the snapshots folder.")
        return

    confidence_scores = [
        extract_confidence_from_filename(path) or 50.0 for path in image_paths
    ]

    # --- Show most recent image with confidence ---
    if image_paths:
        try:
            current_image = Image.open(image_paths[0])
            resized_image = current_image.resize((400, 200), Image.Resampling.LANCZOS)

            with st.container():
                metric_col, img_col = st.columns([2, 5])

                with metric_col:
                    conf = confidence_scores[0]
                    delta = conf - confidence_scores[1] if len(confidence_scores) > 1 else None
                    st.metric("Confidence", f"{conf:.1f}%", f"{delta:.1f}%" if delta else None)

                with img_col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(resized_image, use_container_width=True, caption="Most Recent")
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading current image: {e}")
    else:
        # No images → placeholder as most recent
        try:
            placeholder_image = Image.open(placeholder_path)
            resized_placeholder = placeholder_image.resize((400, 200), Image.Resampling.LANCZOS)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(resized_placeholder, use_container_width=True, caption="Most Recent")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"No images found and placeholder not available: {e}")

    # --- Show grid of up to 4 previous images ---
    st.markdown("**Previous Images**")
    col1, col2 = st.columns(2)
    previous_images = image_paths[1:5]

    for i in range(4):  # always render 4 slots
        target_col = col1 if i % 2 == 0 else col2

        if i < len(previous_images):  # real image available
            try:
                img = Image.open(previous_images[i])
                resized_img = img.resize((200, 100), Image.Resampling.LANCZOS)

                with target_col:
                    metric_col, img_col = st.columns([2, 4])

                    with metric_col:
                        conf_index = i + 1
                        if conf_index < len(confidence_scores):
                            conf = confidence_scores[conf_index]
                            delta = conf - confidence_scores[conf_index + 1] if (conf_index + 1) < len(
                                confidence_scores) else None
                            st.metric("Confidence", f"{conf:.1f}%", f"{delta:.1f}%" if delta else None)
                        else:
                            st.metric("Confidence", "N/A")

                    with img_col:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(resized_img, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading image {i + 2}: {e}")
        else:
            # Always show placeholder with metric
            try:
                placeholder_img = Image.open(placeholder_path)
                resized_placeholder = placeholder_img.resize((200, 100), Image.Resampling.LANCZOS)

                with target_col:
                    metric_col, img_col = st.columns([2, 4])

                    with metric_col:
                        st.metric("Confidence", "N/A")

                    with img_col:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(resized_placeholder, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                with target_col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.write("No image available")
                    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(close_card(), unsafe_allow_html=True)

def render_dashboard_content():
    """Render the main dashboard content once"""
    # Initialize and load data
    initialize_class_counters()
    real_time_data = load_real_time_data()
    latest_images = load_latest_snapshots()
    
    if real_time_data:
        update_class_counters(real_time_data)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Image Analysis Dashboard</h1>
        <p>Real-time monitoring and analytics</p>
    </div>
    """, unsafe_allow_html=True)


    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        if UPDATE_SIGNAL_FILE.exists():
            st.warning("Signal file detected!")
        else:
            st.success("Monitoring active")
        
        st.info(f"Watching: {UPDATE_SIGNAL_FILE.name}")
        
        if real_time_data:
            st.success("Real-time data loaded")
            if 'timestamp' in real_time_data:
                st.text(f"Data timestamp: {real_time_data['timestamp']}")
        else:
            st.error("No real-time data")
        
        # Interactive buttons that work during monitoring
        if st.button("Test Signal"):
            UPDATE_SIGNAL_FILE.touch()
            st.success("Test signal created")
        
        if st.button("Reset Counters"):
            for key in st.session_state.class_counters:
                st.session_state.class_counters[key] = 0
            st.success("Counters reset!")
            st.rerun()
        
        if st.button("Manual Refresh"):
            st.rerun()

        if st.button("Stop Monitoring"):
            st.session_state.monitoring_active = False
            st.success("Monitoring stopped")
        
        st.text(f"Last updated: {time.strftime('%H:%M:%S')}")
        st.info("Auto-monitoring: Every 2s")
    
    # Main content
    if not Path("../../snapshots").exists():
        st.error("'snapshots' folder does not exist.")
        return
    
    if not latest_images:
        st.warning("No images found in snapshots folder.")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.header("Recent Snapshots")
        st.markdown("*Latest captured images with confidence scores*")
        display_image_vertical_with_metrics(latest_images)
    
    with col2:
        st.header("Analytics & Metrics")
        st.markdown("*Real-time analysis and performance data*")
        
        predictions_data = create_predictions_data(real_time_data)
        predictions(predictions_data)

        parameters = {
            'A': '25.5%', 'B': '$18.2K', 'C': '33.1°C',
            'D': '12.7 mins', 'E': '10.5 kg'
        }
        parameter_metrics(parameters, "Parameter Metrics")

    with col3:
        st.header("Statistics")
        st.markdown("Collected data")

        undamaged_counts, damaged_counts = get_metric_values()
        amount_metric("Damaged", damaged_counts)
        amount_metric("Undamaged", undamaged_counts)


st.set_page_config(
    page_title="Image Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    st.title("Image Analysis Dashboard")
    
    # Render dashboard content once
    render_dashboard_content()

    content_container = st.empty()

    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = True

    while st.session_state.monitoring_active:
        # Check for updates
        has_update, signal_info = check_for_signal_file()

        # Always render content (either updated or same)
        with content_container.container():
            render_dashboard_content()

            # Show update status if there was a signal
            if has_update and signal_info:
                st.success(f"✅ Updated at: {signal_info['time_str']}")

        # Wait before next check
        time.sleep(2)

if __name__ == "__main__":
    main()