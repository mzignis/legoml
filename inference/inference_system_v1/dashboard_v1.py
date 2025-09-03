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
    snapshots_dir = Path("/home/candfpi4b/fresh_repo/legoml/snapshots")
    if snapshots_dir.exists():
        image_files = sorted(
            glob.glob(str(snapshots_dir / "*.jpg")) + glob.glob(str(snapshots_dir / "*.JPG")),
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

def amount_metric(state, num):
    """Display brick count metrics in a grid"""
    with st.container():
        st.markdown(f"### {state}")
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

def parameter_metrics(parameters, label): 
    """Display additional parameter metrics"""
    with st.container():
        st.markdown(f"### {label}")
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

def predictions(data_dict):
    """Create a Plotly horizontal bar chart for predictions"""
    with st.container():
        categories = data_dict['categories']
        confidences = data_dict['confidences']
        
        fig = go.Figure(data=[
            go.Bar(
                y=categories,
                x=confidences,
                orientation='h',
                marker_color='#B34949',
                text=[round(conf, 1) for conf in confidences],
                texttemplate='%{text}%',
                textposition='inside',
                textfont=dict(color='white', size=12)
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
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, range=[0, 100]),
            yaxis=dict(showgrid=False, autorange='reversed')
        )
        
        st.plotly_chart(fig, use_container_width=True)

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
    if not image_paths:
        st.warning("No JPG images found in the snapshots folder.")
        return
    
    image_sizes = [(400, 200), (360, 180), (300, 150), (240, 120), (200, 100)]
    
    confidence_scores = []
    for image_path in image_paths:
        conf = extract_confidence_from_filename(image_path)
        confidence_scores.append(conf if conf is not None else 50.0)
    
    st.markdown("""
    <style>
    .stContainer > div { margin-bottom: 24px; }
    </style>
    """, unsafe_allow_html=True)
    
    for i, image_path in enumerate(image_paths[:5]):
        try:
            current_image = Image.open(image_path)
            target_width, target_height = image_sizes[i]
            resized_image = current_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            with st.container():
                metric_col, img_col = st.columns([1, 4])
                
                with metric_col:
                    confidence = confidence_scores[i]
                    delta = None
                    if i+1 < len(confidence_scores):
                        delta = confidence - confidence_scores[i+1]
                    
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%",
                        delta=f"{delta:.1f}%" if delta is not None else None
                    )
                
                with img_col:
                    st.image(resized_image, width=target_width)
                
                st.write("")
                
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")

def render_dashboard_content():
    """Render the main dashboard content once"""
    # Initialize and load data
    initialize_class_counters()
    real_time_data = load_real_time_data()
    latest_images = load_latest_snapshots()
    
    if real_time_data:
        update_class_counters(real_time_data)
    
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
        
        st.text(f"Last updated: {time.strftime('%H:%M:%S')}")
        st.info("Auto-monitoring: Every 2s")
    
    # Main content
    if not Path("/home/candfpi4b/fresh_repo/legoml/snapshots").exists():
        st.error("'snapshots' folder does not exist.")
        return
    
    if not latest_images:
        st.warning("No images found in snapshots folder.")
        return
    
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.header("Recent Snapshots")
        st.markdown("*Latest captured images with confidence scores*")
        display_image_vertical_with_metrics(latest_images)
    
    with right_col:
        st.header("Analytics & Metrics")
        st.markdown("*Real-time analysis and performance data*")
        
        predictions_data = create_predictions_data(real_time_data)
        predictions(predictions_data)
        
        inner_left_col, inner_right_col = st.columns(2)
        undamaged_counts, damaged_counts = get_metric_values()
        
        with inner_left_col:
            amount_metric("Undamaged", undamaged_counts)
        
        with inner_right_col:
            amount_metric("Damaged", damaged_counts)
        
        parameters = {
            'A': '25.5%', 'B': '$18.2K', 'C': '33.1°C', 
            'D': '12.7 mins', 'E': '10.5 kg'
        }
        parameter_metrics(parameters, "Parameter Metrics")

st.set_page_config(
    page_title="Image Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    st.title("Image Analysis Dashboard")
    
    # Render dashboard content once
    render_dashboard_content()
    
    # Efficient monitoring loop - only rerun when signal detected
    while True:
        has_update, signal_info = check_for_signal_file()
        
        if has_update:
            # Show brief update notification
            st.success("Real-time update detected! Refreshing...")
            if signal_info:
                st.sidebar.success(f"Signal at: {signal_info['time_str']}")
            
            time.sleep(0.5)  # Brief pause to show notification
            st.rerun()  # Refresh with new data
            break  # Exit loop since st.rerun() restarts everything
        
        time.sleep(2)  # Wait 2 seconds before next check

if __name__ == "__main__":
    main()