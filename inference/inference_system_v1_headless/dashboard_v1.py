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


st.set_page_config(
    page_title="Image Dashboard",
    layout="wide",  # This makes the page use the full width
    initial_sidebar_state="collapsed"  # Hide sidebar for cleaner look
)

# Hide the Streamlit header/toolbar to maximize screen space
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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


def get_latest_images(image_folder, num_images=5):

    all_images = []
    all_images.extend(glob.glob(os.path.join(image_folder, '*.jpg')))
    all_images.extend(glob.glob(os.path.join(image_folder, '*.JPG')))
    
    # Sort images by modification time (newest first)
    # os.path.getmtime gets the last modification time of a file
    all_images.sort(key=os.path.getmtime, reverse=True)
    
    # Return only the requested number of images
    return all_images[:num_images]


def display_image_grid(image_paths):

    if not image_paths:
        st.warning("No JPG images found in the specified folder.")
        return
    
    # Display current (most recent) image - resized to rectangular aspect ratio
    if len(image_paths) > 0:
        try:
            current_image = Image.open(image_paths[0])
            # Resize to slightly smaller rectangular aspect ratio for better fit
            target_width = 400
            target_height = 200  # Slightly shorter (was 225)
            
            # Resize while maintaining aspect ratio, then crop to target dimensions
            resized_image = current_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            st.image(resized_image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading current image: {e}")
    
    # Add a small spacer between current and grid images
    st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)
    
    # Display the previous 4 images in a 2x2 grid with minimal spacing
    if len(image_paths) > 1:
        # Create two rows of two columns each - minimal gaps
        row1_col1, row1_col2 = st.columns(2, gap="small")
        row2_col1, row2_col2 = st.columns(2, gap="small")
        
        # List of column objects for easy iteration
        grid_columns = [row1_col1, row1_col2, row2_col1, row2_col2]
        
        # Display up to 4 previous images
        for i in range(1, min(5, len(image_paths))):
            try:
                # Open and resize the image file to rectangular format
                img = Image.open(image_paths[i])
                
                # Resize to smaller rectangular aspect ratio for grid - slightly shorter
                target_width = 200
                target_height = 100  # Reduced from 112 to save vertical space
                
                # Resize while maintaining quality
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Calculate which column to use (0-3)
                col_index = i - 1
                
                # Display in the appropriate column - use full container width for maximum space usage
                with grid_columns[col_index]:
                    st.image(resized_img, use_container_width=True)  # Fill the container completely
            except Exception as e:
                with grid_columns[col_index]:
                    st.error(f"Error loading image {i+1}: {e}")

def display_image_vertical(image_paths):

    if not image_paths:
        st.warning("No JPG images found in the specified folder.")
        return
    
    # Define fixed rectangular sizes for each image (width, height) - keeping original proportions
    # Each subsequent image will be smaller but maintain the 2:1 ratio from original code
    image_sizes = [
        (400, 200),   # Most recent image (same as original "current" image)
        (320, 160),   # Second image (80% of original)
        (256, 128),   # Third image (64% of original)  
        (200, 100),   # Fourth image (same as original "grid" images)
        (160, 80)     # Fifth image (80% of grid size)
    ]
    
    # custom CSS for left alignment
    st.markdown("""
    <style>
    .left-aligned-image {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display all images vertically with decreasing sizes
    for i, image_path in enumerate(image_paths[:5]):  # Limit to 5 images
        try:
            # Open the image
            current_image = Image.open(image_path)
            
            # Get the fixed size for this image
            target_width, target_height = image_sizes[i]
            
            # Resize to fixed rectangular dimensions (same as original code)
            resized_image = current_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Display with left alignment using HTML container
            st.markdown(f'<div class="left-aligned-image">', unsafe_allow_html=True)
            st.image(resized_image, width=target_width)
            st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")

def display_image_vertical_with_metrics(image_paths, confidence_scores=None):

    if not image_paths:
        st.warning("No JPG images found in the specified folder.")
        return
    
    # Define fixed rectangular sizes for each image (width, height) - more gradual shrinking
    image_sizes = [
        (400, 200),   # Most recent image
        (360, 180),   # Second image  
        (300, 150),   # Third image
        (240, 120),   # Fourth image
        (200, 100)    # Fifth image (same as old fourth image)
    ]
    
    # Default confidence scores if none provided
    if confidence_scores is None:
        confidence_scores = [0.92, 0.87, 0.81, 0.75, 0.68, 0.63]  # Deltas: +0.05, +0.06, +0.06, +0.07, +0.05  # Deltas: None, -0.05, -0.06, -0.06, -0.07
    
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
                    confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                    
                    # Calculate delta as change to next confidence
                    if i+1 >= len(confidence_scores):  # No next confidence available
                        delta = None
                    else:
                        next_confidence = confidence_scores[i+1]
                        delta = confidence - next_confidence
                    
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.2f}",
                        delta=f"{delta:.3f}" if delta is not None else None
                    )
                
                with img_col:
                    st.image(resized_image, width=target_width)
                
                # Add extra spacing after each image-metric pair
                st.write("")  # Extra spacing between images
                
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")


def main():
    st.title("Image Analysis Dashboard")
    
    # Configuration section in the sidebar (you can expand this)
    with st.sidebar:
        st.header("Settings")
        
        image_folder = st.text_input(
            "Image Folder Path", 
            value=r"C:\Users\atok\OneDrive - C&F S.A\Desktop\dashboard\images", 
            help="Enter the path to the folder containing your JPG images"
        )
        
        # Button to refresh images
        if st.button("Refresh Images"):
            st.rerun()
        
        # Display last update time
        st.text(f"Last updated: {time.strftime('%H:%M:%S')}")
    
    if not os.path.exists(image_folder):
        st.error(f"Folder '{image_folder}' does not exist. Please create it or specify a different path.")
        st.info("Create an 'images' folder in your project directory and add some JPG files to get started.")
        return
    
    latest_images = get_latest_images(image_folder)
    
    # Display the images
    left_col, right_col = st.columns([1, 1])  
    
    with left_col:
        st.header("Recent Snapshots")
        st.markdown("*Latest captured images with confidence scores*")
        display_image_vertical_with_metrics(latest_images)
    
    with right_col:
        st.header("Analytics & Metrics")
        st.markdown("*Real-time analysis and performance data*")
        
        df = pd.DataFrame(
            {
            'Top Categories': ['A', 'B', 'C', 'D', 'E'],
            'Confidence (%)': [25.5, 18.2, 33.1, 12.7, 10.5]
            }
        )
        predictions(df)
        inner_left_col, inner_right_col = st.columns(2)

        with inner_left_col:
            sample_numbers1 = [10, 25, 33, 18, 42, 67]
            amount_metric("Undamaged", sample_numbers1)

        with inner_right_col:
            sample_numbers2 = [15, 30, 28, 22, 38, 55]
            amount_metric("Damaged", sample_numbers2)
        parameters = {
            'A': '25.5%',
            'B': '$18.2K',
            'C': '33.1°C',
            'D': '12.7 mins',
            'E': '10.5 kg'
        }
        parameter_metrics(parameters, "Parameter Metrics")
    
    # Auto-refresh every 30 seconds (optional)
    # Uncomment the next line if you want automatic refresh
    # time.sleep(30)
    # st.rerun()

if __name__ == "__main__":
    main()