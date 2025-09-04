import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image
import time
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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

def load_latest_snapshots(image_folder, limit=5):
    """Load the most recent snapshot images"""
    snapshots_dir = Path(image_folder)
    if snapshots_dir.exists():
        image_files = sorted(
            glob.glob(str(snapshots_dir / "*.jpg")) + glob.glob(str(snapshots_dir / "*.JPG")),
            key=lambda x: Path(x).stat().st_mtime,
            reverse=True
        )
        return image_files[:limit]
    return []


def conveyor_belt_metrics():
    """Display conveyor belt metrics in a card"""
    import streamlit as st

    st.markdown(create_card_header("Conveyor Belt Status", "Real-time operational metrics"), unsafe_allow_html=True)

    # First row - Speed and Temperature
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Belt Speed",
            value="2.1 m/s",
            delta="0.1 m/s",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Temperature",
            value="22.5°C",
            delta="-0.3°C",
            delta_color="inverse"
        )

    # Second row - Vibration and Uptime
    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            label="Vibration",
            value="0.01 mm",
            delta="-0.005 mm",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Uptime",
            value="99.2%",
            delta="0.5%",
            delta_color="normal"
        )

    # Third row - Throughput and Efficiency
    col5, col6 = st.columns(2)

    with col5:
        st.metric(
            label="Throughput",
            value="782 items/hr",
            delta="45 items/hr",
            delta_color="normal"
        )

    with col6:
        st.metric(
            label="Efficiency",
            value="92.8%",
            delta="2.1%",
            delta_color="normal"
        )

    st.markdown(close_card(), unsafe_allow_html=True)


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


def amount_metric_grid(state, num):
    """Display metrics in a grid layout within a card"""
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


def parameter_metrics_grid(parameters, label):
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


def predictions_chart(df):
    """Create a horizontal bar chart with dark theme"""
    st.markdown(create_card_header("Top Categories", "Confidence Distribution"), unsafe_allow_html=True)

    # Create Plotly horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=df['Top Categories'],
            x=df['Confidence (%)'],
            orientation='h',
            marker_color='#4a9eff',
            text=df['Confidence (%)'].apply(lambda x: f'{x}%'),
            textposition='auto',
        )
    ])

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(close_card(), unsafe_allow_html=True)


def confidence_overview():
    """Display overall confidence metrics"""
    st.markdown(create_card_header("Overall Performance"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Overall Accuracy",
            value="84%",
            delta="12.4%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Processing Speed",
            value="2.3s",
            delta="-0.5s",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Images Processed",
            value="1,247",
            delta="156",
            delta_color="normal"
        )

    st.markdown(close_card(), unsafe_allow_html=True)


def display_recent_images(image_paths):
    """Display recent images in a card"""
    st.markdown(create_card_header("Recent Snapshots", "Latest captured images"), unsafe_allow_html=True)

    placeholder_path = "placeholder.jpg"

    # --- Show most recent image ---
    if image_paths:
        try:
            current_image = Image.open(image_paths[0])
            resized_image = current_image.resize((400, 200), Image.Resampling.LANCZOS)
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

    # Take only the 4 images *after* the most recent
    previous_images = image_paths[1:5]

    for i in range(4):  # always render 4 slots
        target_col = col1 if i % 2 == 0 else col2

        if i < len(previous_images):  # real image available
            try:
                img = Image.open(previous_images[i])
                resized_img = img.resize((200, 100), Image.Resampling.LANCZOS)
                with target_col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(resized_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading image {i+2}: {e}")
        else:
            # Show placeholder if not enough images
            try:
                placeholder_img = Image.open(placeholder_path)
                resized_placeholder = placeholder_img.resize((200, 100), Image.Resampling.LANCZOS)
                with target_col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(resized_placeholder, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                with target_col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.write("No image available")
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(close_card(), unsafe_allow_html=True)



def display_confidence_timeline(image_paths, confidence_scores=None):
    """Display images with confidence metrics timeline"""
    st.markdown(create_card_header("Confidence Timeline", "Image analysis confidence over time"),
                unsafe_allow_html=True)

    if confidence_scores is None:
        confidence_scores = [0.92, 0.87, 0.81, 0.75, 0.68]

    if not image_paths:
        st.warning("No images available")
        st.markdown(close_card(), unsafe_allow_html=True)
        return

    for i, image_path in enumerate(image_paths[:3]):  # Show only top 3 in timeline
        try:
            current_image = Image.open(image_path)
            resized_image = current_image.resize((300, 150), Image.Resampling.LANCZOS)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(resized_image, use_container_width=True)

            with col2:
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                delta = None
                if i + 1 < len(confidence_scores):
                    delta = confidence - confidence_scores[i + 1]

                st.metric(
                    label=f"Image {i + 1} Confidence",
                    value=f"{confidence:.2f}",
                    delta=f"{delta:.3f}" if delta is not None else None
                )

            if i < 2:  # Don't add divider after last item
                st.divider()

        except Exception as e:
            st.error(f"Error loading image {i + 1}: {e}")

    st.markdown(close_card(), unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Image Analysis Dashboard</h1>
        <p>Real-time monitoring and analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        image_folder = st.text_input(
            "Image Folder Path",
            value="../snapshots",
            help="Enter the path to the folder containing your JPG images"
        )

        if st.button("🔄 Refresh Data"):
            st.rerun()

        st.text(f"⏱️ Last updated: {time.strftime('%H:%M:%S')}")

        # Status indicators
        st.markdown("---")
        st.markdown("**System Status**")

        st.success("✅ Online")

        st.info("📊 Processing")
        st.warning("⚠️ Queue: 3 items")

        # Threading info
        st.markdown("---")
        st.markdown("**System Info**")

    # Initialize data

    mock_data = None
    if image_folder and os.path.exists(image_folder):
        latest_images = load_latest_snapshots(image_folder, limit=5)
        folder_exists = True
    else:
        latest_images = []
        folder_exists = False

    if not folder_exists:
        st.error(
            f"Folder '{image_folder}' does not exist. Please create it, specify a different path, or enable Mock Data mode for testing.")
        st.info("💡 **Tip**: Enable 'Use Mock Data' in the sidebar to test the dashboard with sample data!")
        return

    # Main dashboard grid layout
    # Row 1: Overview metrics
    confidence_overview()

    # Row 2: Main content - 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Recent images
        display_recent_images(latest_images)



    with col2:
        # Predictions chart
        df = pd.DataFrame({
            'Top Categories': ['A', 'B', 'C', 'D', 'E'],
            'Confidence (%)': [25.5, 18.2, 33.1, 12.7, 10.5]
        })
        predictions_chart(df)

        # Conveyor belt metrics
        conveyor_belt_metrics()



    with col3:
        # Damaged metrics
        sample_numbers2 = [15, 30, 28, 22, 38, 55]
        amount_metric_grid("Damaged Bricks", sample_numbers2)

        # Undamaged metrics

        sample_numbers1 = [10, 25, 33, 18, 42, 67]
        amount_metric_grid("Undamaged Bricks", sample_numbers1)

        # Parameter metrics
        parameters = {
            'A': '25.5%',
            'B': '$18.2K',
            'C': '33.1°C',
            'D': '12.7 mins',
            'E': '10.5 kg'
        }
        parameter_metrics_grid(parameters, "System Parameters")


if __name__ == "__main__":
    main()