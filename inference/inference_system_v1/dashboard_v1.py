import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image
import time
import json
from pathlib import Path


def load_real_time_data():
    data_file = Path("dashboard_data.json")
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return None


def initialize_class_counters():
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
        
        # Increment counter for the predicted class (if it's one of our 12 tracked classes)
        if predicted_class in counters:
            st.session_state.class_counters[predicted_class] += 1
            st.session_state.last_processed_timestamp = current_timestamp
            return True, {'class': predicted_class, 'confidence': max_confidence}
    
    return False, None


def get_file_modification_time(filepath):
    try:
        return Path(filepath).stat().st_mtime
    except:
        return None


def get_metric_values():
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


def check_for_updates():
    json_file = Path("dashboard_data.json")
    snapshots_dir = Path("snapshots")
    
    # Initialize session state for tracking modification times
    if 'last_json_mtime' not in st.session_state:
        st.session_state.last_json_mtime = None
    if 'last_snapshots_mtime' not in st.session_state:
        st.session_state.last_snapshots_mtime = None
    
    updates_detected = False
    
    # Check JSON file
    if json_file.exists():
        current_json_mtime = get_file_modification_time(json_file)
        if st.session_state.last_json_mtime != current_json_mtime:
            st.session_state.last_json_mtime = current_json_mtime
            updates_detected = True
    
    # Check snapshots directory
    if snapshots_dir.exists():
        try:
            # Get the most recent image modification time
            image_files = sorted(
                glob.glob(str(snapshots_dir / "*.jpg")) + glob.glob(str(snapshots_dir / "*.JPG")),
                key=lambda x: Path(x).stat().st_mtime,
                reverse=True
            )
            if image_files:
                current_snapshots_mtime = Path(image_files[0]).stat().st_mtime
                if st.session_state.last_snapshots_mtime != current_snapshots_mtime:
                    st.session_state.last_snapshots_mtime = current_snapshots_mtime
                    updates_detected = True
        except:
            pass
    
    return updates_detected


def load_latest_snapshots(limit=5):
    snapshots_dir = Path("snapshots")
    if snapshots_dir.exists():
        image_files = sorted(
            glob.glob(str(snapshots_dir / "*.jpg")) + glob.glob(str(snapshots_dir / "*.JPG")),
            key=lambda x: Path(x).stat().st_mtime,
            reverse=True
        )
        return image_files[:limit]
    return []


def create_predictions_dataframe(real_time_data):
    """Create DataFrame for predictions chart from real-time data"""
    if real_time_data is None:
        # Fallback data if no real-time data available
        return pd.DataFrame({
            'Top Categories': ['No Data', 'Available', 'Please Check', 'JSON File', 'Connection'],
            'Confidence (%)': [20, 20, 20, 20, 20]
        })
    
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
        other_confidence = max(0, 100 - total_confidence)  # Ensure non-negative
        
        # Create the DataFrame
        categories = classes + ['Other']
        confidence_values = confidences_pct + [other_confidence]
        
        return pd.DataFrame({
            'Top Categories': categories,
            'Confidence (%)': confidence_values
        })
    
    except Exception as e:
        st.error(f"Error processing real-time data: {e}")
        # Return fallback data
        return pd.DataFrame({
            'Top Categories': ['Error', 'Loading', 'Data', 'Please', 'Retry'],
            'Confidence (%)': [20, 20, 20, 20, 20]
        })


st.set_page_config(
    page_title="Image Dashboard",
    layout="wide",  # This makes the page use the full width
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed but accessible
)

def amount_metric(state, num):
    with st.container():
        st.markdown(f"### {state}")
        # First row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="W 1x3",
                value=num[0],
                border=True
            )

        with col2:
            st.metric(
                label="W 2x2", 
                value=num[1],
                border=True
            )

        with col3:
            st.metric(
                label="W 2x4",
                value=num[2],
                border=True
            )

        # Second row
        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                label="B 2x2",
                value=num[3],
                border=True
            )

        with col5:
            st.metric(
                label="B 1x6", 
                value=num[4],
                border=True
            )

        with col6:
            st.metric(
                label="B 2x6",
                value=num[5],
                border=True
            )

def parameter_metrics(parameters, label): 
    with st.container():
        st.markdown(f"### {label}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Metric 1",
                value=parameters['A'], 
                border=True
            )

        with col2:
            st.metric(
                label="Metric 2",  
                value=parameters['B'], 
                border=True
            )

        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric(
                label="Metric 3",
                value=parameters['C'],
                border=True
            )

        with col4:
            st.metric(
                label="Metric 4", 
                value=parameters['D'],
                border=True
            )

        with col5:
            st.metric(
                label="Metric 5",
                value=parameters['E'],
                border=True
            )

def predictions_prg(data_df):
    with st.container():
        st.data_editor(
            data_df,
            column_config={
                "Brick": st.column_config.TextColumn(
                    "Brick",
                    help="Top 5 brick types",
                    width="small",
                ),
                "predictions": st.column_config.ProgressColumn(
                    "Confidence Score",
                    help="The 5 highest confidence scores",
                    format="percent",
                ),
            },
            hide_index=True,
        )

def predictions(df):
    with st.container():
        st.bar_chart(
            df,
            x="Top Categories",
            y="Confidence (%)",
            color="#B34949",
            horizontal=True,
            height=300,
            use_container_width=True
        )

def extract_confidence_from_filename(filepath):
    """Extract confidence value from filename ending with conf{X.XXX}.jpg"""
    try:
        filename = Path(filepath).name
        # Look for pattern like "conf0.762.jpg"
        if "conf" in filename and filename.endswith(".jpg"):
            # Find the part after "conf" and before ".jpg"
            conf_part = filename.split("conf")[-1].replace(".jpg", "")
            confidence = float(conf_part)
            return confidence * 100 if confidence <= 1.0 else confidence  # Convert to percentage if needed
    except (ValueError, IndexError):
        pass
    return None


def display_image_vertical_with_metrics(image_paths, confidence_scores=None, real_time_data=None):    
    if not image_paths:
        st.warning("No JPG images found in the snapshots folder.")
        return
    
    # Define fixed rectangular sizes for each image (width, height) - more gradual shrinking
    image_sizes = [
        (400, 200),   # Most recent image
        (360, 180),   # Second image  
        (300, 150),   # Third image
        (240, 120),   # Fourth image
        (200, 100)    # Fifth image (same as old fourth image)
    ]
    
    # Extract confidence scores from filenames
    confidence_scores = []
    for image_path in image_paths:
        conf = extract_confidence_from_filename(image_path)
        if conf is not None:
            confidence_scores.append(conf)
        else:
            # Fallback confidence if filename doesn't contain confidence
            confidence_scores.append(50.0)  # Default 50%
    
    # Add custom CSS for the image-metric containers
    st.markdown("""
    <style>
    .image-metric-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 32px;  /* Increased vertical spacing between images */
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
    
    /* Additional spacing between image containers */
    .stContainer > div {
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display all images vertically with their metrics
    for i, image_path in enumerate(image_paths[:5]):  # Limit to 5 images
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
                    if i+1 >= len(confidence_scores):  # No next confidence available
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
                
                # Add extra spacing after each image-metric pair
                st.write("")  # Extra spacing between images
                
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")


def main():
    st.title("Image Analysis Dashboard")
    
    # Initialize auto-refresh settings in session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5  # seconds
    
    # Initialize class counters
    initialize_class_counters()
    
    # Load real-time data and latest snapshots
    real_time_data = load_real_time_data()
    latest_images = load_latest_snapshots()
    
    # Update class counters if new data detected
    counter_updated = False
    detection_info = None
    if real_time_data:
        counter_updated, detection_info = update_class_counters(real_time_data)
    
    # Configuration section in the sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Auto-refresh controls
        st.subheader("Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)", 
                min_value=1, 
                max_value=30, 
                value=st.session_state.refresh_interval,
                step=1
            )
            st.session_state.refresh_interval = refresh_interval
        
        # Display data status
        if real_time_data:
            st.success("Real-time data loaded")
            if 'timestamp' in real_time_data:
                st.text(f"Data timestamp: {real_time_data['timestamp']}")
            if counter_updated and detection_info:
                st.success("Class counter updated!")
                predicted_class = detection_info['class']
                confidence = detection_info['confidence']
                # Convert confidence to percentage if needed
                if confidence <= 1.0:
                    confidence_pct = confidence * 100
                else:
                    confidence_pct = confidence
                st.info(f"Detected: **{predicted_class}** ({confidence_pct:.1f}%)")
        else:
            st.error("No real-time data")
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
        
        # Debug: Show current counters
        if st.checkbox("Show Class Counters"):
            st.subheader("Current Counts:")
            counters = st.session_state.class_counters
            total_detections = sum(counters.values())
            st.metric("Total Detections", total_detections)
            
            for class_name, count in counters.items():
                if count > 0:  # Only show non-zero counts
                    st.text(f"{class_name}: {count}")
        
        # Debug: Show raw data if available
        if st.checkbox("Show Raw Data") and real_time_data:
            st.json(real_time_data)
            
        # Debug: Show extracted confidences from image filenames
        if st.checkbox("Show Image Confidences") and latest_images:
            st.subheader("Extracted Confidences:")
            for i, img_path in enumerate(latest_images[:5]):
                filename = Path(img_path).name
                conf = extract_confidence_from_filename(img_path)
                if conf is not None:
                    st.text(f"{filename}: {conf:.1f}%")
                else:
                    st.text(f"{filename}: No confidence found")
    
    # Check if snapshots directory exists
    if not Path("snapshots").exists():
        st.error("'snapshots' folder does not exist. Please create it and add JPG images.")
        st.info("Create a 'snapshots' folder in your project directory and add JPG files to get started.")
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
        predictions_df = create_predictions_dataframe(real_time_data)
        predictions(predictions_df)
        
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
    
    # Auto-refresh functionality
    if st.session_state.auto_refresh:
        # Check for file updates every time the app runs
        updates_detected = check_for_updates()
        
        if updates_detected or counter_updated:
            if counter_updated:
                st.success("New detection processed! Counter updated.")
            else:
                st.success("File changes detected! Dashboard updated.")
        
        # Show auto-refresh status
        last_update_time = st.session_state.get('last_json_mtime', None)
        if last_update_time:
            formatted_time = time.strftime('%H:%M:%S', time.localtime(last_update_time))
            st.caption(f"Auto-refresh active • Last update: {formatted_time}")
        else:
            st.caption("Auto-refresh active • Waiting for data...")
        
        # Show auto-refresh indicator
        st.markdown(f"""
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 999; 
                    background: rgba(34, 139, 34, 0.9); color: white; 
                    padding: 8px 12px; border-radius: 20px; font-size: 12px;">
            Monitoring files every {st.session_state.refresh_interval}s
        </div>
        """, unsafe_allow_html=True)
        
        # Simple page refresh for development/demo purposes
        # In production, consider using file watchers, websockets, or other real-time mechanisms
        if st.session_state.refresh_interval <= 10:  # Only for short intervals
            st.markdown(f"""
            <meta http-equiv="refresh" content="{st.session_state.refresh_interval}">
            """, unsafe_allow_html=True)
    
    else:
        st.caption("Auto-refresh disabled • Use manual refresh to update")

if __name__ == "__main__":
    main()