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
    uploaded_files = st.file_uploader(
        "Upload .X_Y file(s)",
        type=["X_Y", "xy", "txt"],
        accept_multiple_files=True,
        help="Upload one or more XRD files to compare patterns"
    )


if uploaded_files is not None and len(uploaded_files) > 0:
    try:
        # Color palette for multiple files
        colors = ["darkblue", "red", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

        # Store all dataframes
        all_dfs = []
        file_names = []

        # Read all uploaded files
        for idx, uploaded_file in enumerate(uploaded_files):
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

            if df is not None and len(df) > 0:
                all_dfs.append(df)
                file_names.append(uploaded_file.name)

        if len(all_dfs) == 0:
            st.error("Could not parse any files. Please check the format.")
            st.stop()

        st.success(f"✓ {len(all_dfs)} file(s) loaded successfully!")

        # Create the XRD plot with all patterns
        fig = go.Figure()

        for idx, (df, name) in enumerate(zip(all_dfs, file_names)):
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=df["2Theta"],
                    y=df["Intensity"],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

        fig.update_layout(
            title=f"XRD Diffraction Pattern{'s' if len(all_dfs) > 1 else ''}",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (a.u.)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            showlegend=True,
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key="xrd_plot")

        # Statistics section
        st.divider()
        st.subheader("📊 Pattern Statistics")

        # Show statistics for each file
        for idx, (df, name) in enumerate(zip(all_dfs, file_names)):
            with st.expander(f"📋 {name}", expanded=(len(all_dfs) == 1)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(df))
                with col2:
                    st.metric("2θ Range", f"{int(df['2Theta'].min())}° - {int(df['2Theta'].max())}°")
                with col3:
                    st.metric("Max Intensity", f"{df['Intensity'].max():.2f}")

                # Show data preview
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Showing first 20 of {len(df)} rows")

                # Download button for processed data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"{name.rsplit('.', 1)[0]}_data.csv",
                    mime="text/csv",
                    key=f"download_{idx}"
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
