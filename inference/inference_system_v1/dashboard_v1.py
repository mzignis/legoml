import streamlit as st
import os
import glob
import queue
import time
import json
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go

from signal_listener import start_signal_listener


# Signal handling functions
def setup_signal_listener():
    """Initialize the signal listener (only once per session)"""
    if 'signal_listener' not in st.session_state:
        # Create a queue for thread-safe communication
        st.session_state.update_queue = queue.Queue()
        
        def on_signal_received(signal: bool):
            """Callback when signal is detected"""
            if signal:
                # Put update notification in queue with timestamp
                st.session_state.update_queue.put({
                    'type': 'dashboard_update',
                    'timestamp': time.time(),
                    'source': 'ml_inference'
                })
        
        # Start the signal listener
        st.session_state.signal_listener = start_signal_listener(
            signal_path="/home/candfpi4b/fresh_repo/legoml/inference/inference_system_v1/",  # Adjust path as needed
            callback=on_signal_received,
            check_interval=0.2  # Check every 200ms for responsiveness
        )
        
        st.sidebar.success("🎯 Signal listener started!")


def check_for_updates():
    """Check if any updates are pending in the queue"""
    if hasattr(st.session_state, 'update_queue'):
        try:
            # Non-blocking check for updates
            update = st.session_state.update_queue.get_nowait()
            return True, update
        except queue.Empty:
            return False, None
    return False, None


def cleanup_listener():
    """Clean up the signal listener (called on app shutdown)"""
    if hasattr(st.session_state, 'signal_listener'):
        st.session_state.signal_listener.stop()


# Original dashboard functions (unchanged)
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
    
    # Find the class with highest confidence
    max_confidence_index = confidences.index(max(confidences))
    predicted_class = classes[max_confidence_index]
    max_confidence = confidences[max_confidence_index]
    
    # Check if this is a new prediction (avoid double counting)
    current_timestamp = real_time_data.get('timestamp', '')
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = ''
    
    if current_timestamp != st.session_state.last_processed_timestamp:
        # Initialize counters if needed
        counters = initialize_class_counters()
        
        # Increment counter for the predicted class (if it's one of our tracked classes)
        if predicted_class in counters:
            st.session_state.class_counters[predicted_class] += 1
            st.session_state.last_processed_timestamp = current_timestamp
            return True, {'class': predicted_class, 'confidence': max_confidence}
    
    return False, None


def get_metric_values():
    """Get current counter values organized by category"""
    counters = initialize_class_counters()
    
    # Map counters to metric positions
    undamaged_counts = [
        counters['white_1x3_good'],    # W 1x3
        counters['white_2x2_good'],    # W 2x2  
        counters['white_2x4_good'],    # W 2x4
        counters['blue_2x2_good'],     # B 2x2
        counters['blue_1x6_good'],     # B 1x6
        counters['blue_2x6_good']      # B 2x6
    ]
    
    damaged_counts = [
        counters['white_1x3_damaged'],  # W 1x3
        counters['white_2x2_damaged'],  # W 2x2
        counters['white_2x4_damaged'],  # W 2x4  
        counters['blue_2x2_damaged'],   # B 2x2
        counters['blue_1x6_damaged'],   # B 1x6
        counters['blue_2x6_damaged']    # B 2x6
    ]
    
    return undamaged_counts, damaged_counts


def load_latest_snapshots(limit=5):
    """Load the most recent snapshot images"""
    snapshots_dir = Path("/home/candfpi4b/fresh_repo/snapshots")
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
        
        # Ensure we have 4 classes and confidences
        if len(classes) != 4 or len(confidences) != 4:
            raise ValueError("Expected 4 classes and 4 confidences")
        
        # Convert confidences to percentages if they're in 0-1 range
        if all(c <= 1.0 for c in confidences):
            confidences_pct = [c * 100 for c in confidences]
        else:
            confidences_pct = confidences
        
        # Calculate "other" confidence
        total_confidence = sum(confidences_pct)
        other_confidence = max(0, 100 - total_confidence)
        
        # Create the data dictionary
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


st.set_page_config(
    page_title="Image Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def amount_metric(state, num):
    """Display brick count metrics in a grid"""
    with st.container():
        st.markdown(f"### {state}")
        # First row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="W 1x3", value=num[0], border=True)

        with col2:
            st.metric(label="W 2x2", value=num[1], border=True)

        with col3:
            st.metric(label="W 2x4", value=num[2], border=True)

        # Second row
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
        
        # Create Plotly horizontal bar chart
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
        
        # Update layout
        fig.update_layout(
            height=300,
            xaxis_title="Confidence (%)",
            yaxis_title="Top Categories",
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                range=[0, 100]
            ),
            yaxis=dict(
                showgrid=False,
                autorange='reversed'
            )
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
    
    # Define fixed rectangular sizes for each image
    image_sizes = [
        (400, 200),   # Most recent image
        (360, 180),   # Second image  
        (300, 150),   # Third image
        (240, 120),   # Fourth image
        (200, 100)    # Fifth image
    ]
    
    # Extract confidence scores from filenames
    confidence_scores = []
    for image_path in image_paths:
        conf = extract_confidence_from_filename(image_path)
        if conf is not None:
            confidence_scores.append(conf)
        else:
            confidence_scores.append(50.0)  # Default 50%
    
    # Add custom CSS for the image-metric containers
    st.markdown("""
    <style>
    .image-metric-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 32px;
        gap: 12px;
    }
    
    .image-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    .metric-container {
        display: flex;
        align-items: flex-start;
        min-width: 120px;
    }
    
    .stContainer > div {
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display all images vertically with their metrics
    for i, image_path in enumerate(image_paths[:5]):
        try:
            # Open the image
            current_image = Image.open(image_path)
            
            # Get the fixed size for this image
            target_width, target_height = image_sizes[i]
            
            # Resize to fixed rectangular dimensions
            resized_image = current_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Create a container for this image-metric pair
            with st.container():
                # Create columns: metric on left, image on right
                metric_col, img_col = st.columns([1, 4])
                
                with metric_col:
                    # Display the metric
                    confidence = confidence_scores[i] if i < len(confidence_scores) else 50
                    
                    # Calculate delta as change to next confidence
                    if i+1 >= len(confidence_scores):
                        delta = None
                    else:
                        next_confidence = confidence_scores[i+1]
                        delta = confidence - next_confidence
                    
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%",
                        delta=f"{delta:.1f}%" if delta is not None else None
                    )
                
                with img_col:
                    st.image(resized_image, width=target_width)
                
                st.write("")  # Extra spacing between images
                
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")


def main():
    st.title("Image Analysis Dashboard")
    
    # Setup signal listener (only once)
    setup_signal_listener()
    
    # Check for pending updates from signal listener
    has_update, update_data = check_for_updates()
    
    if has_update:
        st.success("⚡ Real-time update detected! Refreshing...")
        # Optional: Show update details
        if update_data:
            st.sidebar.info(f"Update from: {update_data.get('source', 'unknown')}")
        
        # Trigger immediate refresh
        st.rerun()
    
    # Initialize class counters
    initialize_class_counters()
    
    # Load real-time data and latest snapshots
    real_time_data = load_real_time_data()
    latest_images = load_latest_snapshots()
    
    # Update class counters if new data detected
    if real_time_data:
        counter_updated, detection_info = update_class_counters(real_time_data)
    
    # Configuration section in the sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Signal listener status
        if hasattr(st.session_state, 'signal_listener'):
            st.success("🎯 Real-time monitoring active")
            # Show queue status
            if hasattr(st.session_state, 'update_queue'):
                queue_size = st.session_state.update_queue.qsize()
                if queue_size > 0:
                    st.warning(f"⏳ {queue_size} updates pending")
                else:
                    st.info("📡 Monitoring for updates...")
        else:
            st.error("❌ Signal listener not started")
        
        # Data status
        if real_time_data:
            st.success("✅ Real-time data loaded")
            if 'timestamp' in real_time_data:
                st.text(f"Data timestamp: {real_time_data['timestamp']}")
        else:
            st.error("❌ No real-time data")
            st.text("Check dashboard_data.json")
        
        # Reset counters button
        if st.button("Reset All Counters"):
            for key in st.session_state.class_counters:
                st.session_state.class_counters[key] = 0
            st.success("All counters reset to 0!")
            st.rerun()
        
        # Manual refresh button
        if st.button("Manual Refresh"):
            st.rerun()
        
        # Display last update time
        st.text(f"Last updated: {time.strftime('%H:%M:%S')}")
        
        # Cleanup button (for development)
        if st.button("Stop Listener"):
            cleanup_listener()
            st.success("Signal listener stopped")
    
    # Check if snapshots directory exists
    if not Path("/home/candfpi4b/fresh_repo/snapshots").exists():
        st.error("'snapshots' folder does not exist. Please create it and add JPG images.")
        return
    
    if not latest_images:
        st.warning("No images found in the 'snapshots' folder. Please add some JPG images.")
        return
    
    # Display the dashboard
    left_col, right_col = st.columns([1, 1])  
    
    with left_col:
        st.header("Recent Snapshots")
        st.markdown("*Latest captured images with confidence scores*")
        display_image_vertical_with_metrics(latest_images)
    
    with right_col:
        st.header("Analytics & Metrics")
        st.markdown("*Real-time analysis and performance data*")
        
        # Create predictions chart from real-time data
        predictions_data = create_predictions_data(real_time_data)
        predictions(predictions_data)
        
        inner_left_col, inner_right_col = st.columns(2)

        # Get current counter values
        undamaged_counts, damaged_counts = get_metric_values()

        with inner_left_col:
            amount_metric("Undamaged", undamaged_counts)

        with inner_right_col:
            amount_metric("Damaged", damaged_counts)
            
        parameters = {
            'A': '25.5%',
            'B': '$18.2K',
            'C': '33.1°C',
            'D': '12.7 mins',
            'E': '10.5 kg'
        }
        parameter_metrics(parameters, "Parameter Metrics")


if __name__ == "__main__":
    main()