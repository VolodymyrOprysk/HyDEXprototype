import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import setup_page

setup_page("XRD Data")

st.title("📊 X-Ray Diffraction (XRD) Analysis")
st.markdown("Upload your XRD data and visualize diffraction patterns")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload .X_Y file", type=["X_Y", "xy", "txt"])

    st.subheader("Plot Settings")
    show_peaks = st.checkbox("Show Peak Markers", value=False)

    if show_peaks:
        peak_threshold = st.slider(
            "Peak Threshold (%)",
            min_value=0,
            max_value=100,
            value=80,
            help="Minimum intensity percentage to mark as peak"
        )

if uploaded_file is not None:
    try:
        # Read the file content
        content = uploaded_file.read().decode("utf-8")

        # Try to read with different delimiters
        df = None
        for delimiter in ["\t", r"\s+", ",", ";"]:
            try:
                df = pd.read_csv(
                    StringIO(content),
                    sep=delimiter,
                    header=None,
                    skipinitialspace=True,
                    engine="python" if delimiter == r"\s+" else "c",
                )
                if len(df.columns) >= 2 and len(df) > 0:
                    # Take only first two columns
                    df = df.iloc[:, :2]
                    df.columns = ["2Theta", "Intensity"]
                    # Convert to numeric, dropping any non-numeric rows
                    df["2Theta"] = pd.to_numeric(df["2Theta"], errors="coerce")
                    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
                    df = df.dropna()
                    if len(df) > 0:
                        break
            except Exception as e:
                continue

        if df is None or len(df) == 0:
            st.error("Could not parse the file. Please check the format.")
            st.stop()

        st.success(f"✓ File loaded successfully! {len(df)} data points")

        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"2Theta range: {df['2Theta'].min():.2f} - {df['2Theta'].max():.2f}")
            st.write(f"Intensity range: {df['Intensity'].min():.2f} - {df['Intensity'].max():.2f}")
            st.dataframe(df.head(10))

        # Create the XRD plot
        fig = go.Figure()

        # Add main diffraction pattern
        fig.add_trace(
            go.Scatter(
                x=df["2Theta"],
                y=df["Intensity"],
                mode="lines",
                name="XRD Pattern",
                line=dict(color="darkblue", width=2),
            )
        )

        # Add peak markers if enabled
        if show_peaks:
            max_intensity = df["Intensity"].max()
            threshold_value = max_intensity * (peak_threshold / 100)

            # Find peaks above threshold
            peaks = df[df["Intensity"] > threshold_value]

            if len(peaks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=peaks["2Theta"],
                        y=peaks["Intensity"],
                        mode="markers",
                        name="Peaks",
                        marker=dict(color="red", size=8, symbol="triangle-down"),
                    )
                )

                st.info(f"🔍 {len(peaks)} peaks detected above {peak_threshold}% threshold")

        fig.update_layout(
            title="XRD Diffraction Pattern",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (a.u.)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            showlegend=True,
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key="xrd_plot")

        # Show some basic info about the plot
        st.caption(f"Plotted {len(df)} data points from 2θ = {int(df['2Theta'].min())}° to {int(df['2Theta'].max())}°")

        # Statistics section
        st.divider()
        st.subheader("📊 Pattern Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(df))
        with col2:
            st.metric("2θ Range", f"{int(df['2Theta'].min())}° - {int(df['2Theta'].max())}°")
        with col3:
            st.metric("Max Intensity", f"{df['Intensity'].max():.2f}")

        # Show data preview
        st.divider()
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(f"Showing first 50 of {len(df)} rows")

            # Download button for processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="xrd_data.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your file has 2 columns: 2Theta (degrees) and Intensity")

else:
    st.info("👆 Upload a .X_Y file to get started")
    st.markdown(
        """
    ### Expected file format:
    - **Column 1:** 2θ angle (degrees)
    - **Column 2:** Intensity (arbitrary units)

    Supported delimiters: tab, space, comma, semicolon

    ### Example:
    ```
    10.0    120.5
    10.5    125.3
    11.0    130.1
    ...
    ```
    """
    )
