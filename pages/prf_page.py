import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import setup_page

setup_page("FullProf PRF Visualizer")

st.title("📈 FullProf PRF File Visualizer")
st.markdown("Upload your FullProf PRF file to visualize Rietveld refinement results")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload .prf file", type=["prf", "txt"])

    st.subheader("Display Options")
    show_observed = st.checkbox("Show Observed", value=True)
    show_calculated = st.checkbox("Show Calculated", value=True)
    show_difference = st.checkbox("Show Difference", value=True)
    show_bragg = st.checkbox("Show Bragg Positions", value=True)
    show_grid = st.checkbox("Show Grid Lines", value=False)

    st.subheader("Axis Controls")

    # X-axis scale controls
    st.markdown("**X-axis (2θ) Range**")
    use_auto_scale = st.checkbox("Auto scale X-axis", value=True)

    if not use_auto_scale:
        x_min_manual = st.number_input("X min (2θ)", value=15.0, step=0.1, format="%.1f")
        x_max_manual = st.number_input("X max (2θ)", value=100.0, step=0.1, format="%.1f")

    # X-axis tick step control
    st.markdown("**X-axis Tick Step**")
    use_auto_ticks = st.checkbox("Auto tick spacing", value=True)
    if not use_auto_ticks:
        x_tick_step = st.number_input("Tick step (2θ)", value=5.0, min_value=0.5, max_value=50.0, step=0.5, format="%.1f")

    # Y-axis intensity units visibility
    st.markdown("**Y-axis (Intensity) Labels**")
    y_axis_labels = st.radio(
        "Show intensity values:",
        options=["All", "Only Positive", "None"],
        index=0
    )

if uploaded_file is not None:
    try:
        # Read the file content
        content = uploaded_file.read().decode("utf-8")
        lines = content.split('\n')

        # Extract title from first line
        title = "Unknown Sample"
        if len(lines) > 0:
            first_line = lines[0].strip()
            if first_line:
                # Extract sample name (first word/token before CELL:)
                if "CELL:" in first_line:
                    title = first_line.split("CELL:")[0].strip()
                else:
                    title = first_line.split()[0] if first_line.split() else "Unknown Sample"

        # Extract 2Theta range from header (two consecutive lines with two numbers each)
        theta_min = None
        theta_max = None
        for i in range(len(lines) - 1):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()

            # Check if both lines have exactly 2 numbers
            parts1 = line1.split()
            parts2 = line2.split()

            if len(parts1) == 2 and len(parts2) == 2:
                try:
                    val1_1, val1_2 = float(parts1[0]), float(parts1[1])
                    val2_1, val2_2 = float(parts2[0]), float(parts2[1])

                    # Check if second values look like 2theta range (reasonable values)
                    if 5.0 <= val1_2 <= 200.0 and 5.0 <= val2_1 <= 200.0:
                        theta_min = val1_2
                        theta_max = val2_1
                        break
                except:
                    continue

        # Parse PRF file - FullProf format
        # Columns: 2Theta, Yobs, Ycal, Yobs-Ycal, Backg, Bragg, Posr, (hkl), K

        data_lines = []
        bragg_lines = []
        header_end = 0

        # Find where data starts (look for line starting with number)
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!'):
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        # Check if first column is a number (2Theta)
                        float(parts[0])
                        # Check if it's in reasonable range for 2Theta
                        if 5.0 <= float(parts[0]) <= 180.0:
                            header_end = i
                            break
                    except:
                        continue

        # Read data and extract Bragg positions
        for line in lines[header_end:]:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        two_theta = float(parts[0])
                        y_obs = float(parts[1])
                        y_calc = float(parts[2])
                        y_diff = float(parts[3])

                        # Optional: extract background
                        backg = float(parts[4]) if len(parts) > 4 else 0

                        # Extract Bragg position (column 6, index 5)
                        bragg_pos = float(parts[5]) if len(parts) > 5 else None

                        data_lines.append([two_theta, y_obs, y_calc, y_diff, backg])

                        # Collect unique Bragg positions
                        if bragg_pos is not None and bragg_pos > 0:
                            bragg_lines.append(bragg_pos)

                    except:
                        continue

        if len(data_lines) == 0:
            st.error("Could not parse PRF file. Please check the format.")
            st.stop()

        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=["2Theta", "Y_obs", "Y_calc", "Y_diff", "Background"])

        # Get unique Bragg positions and sort them
        unique_bragg_positions = sorted(list(set(bragg_lines)))

        st.success(f"✓ File loaded successfully! {len(df)} data points, {len(unique_bragg_positions)} Bragg reflections")

        # Display sample name
        st.subheader(f"Sample: {title}")

        # Create the Rietveld plot (single plot)
        fig = go.Figure()

        # Add Observed pattern
        if show_observed:
            fig.add_trace(
                go.Scatter(
                    x=df["2Theta"],
                    y=df["Y_obs"],
                    mode="lines+markers",
                    name="Observed",
                    line=dict(color="red", width=1),
                    marker=dict(color="red", size=4),
                    showlegend=True
                )
            )

        # Add Calculated pattern
        if show_calculated:
            fig.add_trace(
                go.Scatter(
                    x=df["2Theta"],
                    y=df["Y_calc"],
                    mode="lines+markers",
                    name="Calculated",
                    line=dict(color="black", width=1),
                    marker=dict(color="black", size=3),
                    showlegend=True
                )
            )

        # Calculate offset for positioning elements
        y_min = min(df["Y_obs"].min(), df["Y_calc"].min())
        y_max = max(df["Y_obs"].max(), df["Y_calc"].max())
        y_range = y_max - y_min
        offset = y_min - y_range * 0.2  # Place difference 20% below

        # Add Difference curve (offset below the main patterns)
        if show_difference:
            fig.add_trace(
                go.Scatter(
                    x=df["2Theta"],
                    y=df["Y_diff"] + offset,
                    mode="lines",
                    name="Difference",
                    line=dict(color="blue", width=2),
                    showlegend=True
                )
            )

        # Add Bragg reflection markers (between difference and main patterns)
        if show_bragg and len(unique_bragg_positions) > 0:
            # Position Bragg markers between difference curve and main pattern
            bragg_y_position = offset + y_range * 0.1  # 10% above difference curve
            marker_height = y_range * 0.05  # Height of vertical lines

            # Add vertical lines for each unique Bragg position
            for bragg_pos in unique_bragg_positions:
                fig.add_trace(
                    go.Scatter(
                        x=[bragg_pos, bragg_pos],
                        y=[bragg_y_position, bragg_y_position + marker_height],
                        mode="lines",
                        line=dict(color="darkgreen", width=1.5),
                        showlegend=False,
                        hovertemplate=f"2θ = {bragg_pos:.4f}°<extra></extra>"
                    )
                )

            # Add a legend entry for Bragg positions (using one invisible trace)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="darkgreen", width=1.5),
                    name="Bragg Positions",
                    showlegend=True
                )
            )

        # Determine X-axis range
        if use_auto_scale:
            x_range = [theta_min, theta_max] if theta_min is not None and theta_max is not None else None
            x_min_plot = theta_min if theta_min is not None else df["2Theta"].min()
            x_max_plot = theta_max if theta_max is not None else df["2Theta"].max()
        else:
            x_range = [x_min_manual, x_max_manual]
            x_min_plot = x_min_manual
            x_max_plot = x_max_manual

        # Determine X-axis tick values
        if use_auto_ticks:
            x_tickvals = None
            x_ticktext = None
        else:
            # Generate tick values based on custom step
            import math
            tick_start = math.ceil(x_min_plot / x_tick_step) * x_tick_step
            tick_end = math.floor(x_max_plot / x_tick_step) * x_tick_step
            x_tickvals = list(np.arange(tick_start, tick_end + x_tick_step, x_tick_step))
            x_ticktext = [f"{val:.1f}" for val in x_tickvals]

        # Determine Y-axis tick settings based on label preference
        if y_axis_labels == "None":
            yaxis_showticklabels = False
            yaxis_tickvals = None
            yaxis_ticktext = None
        elif y_axis_labels == "Only Positive":
            # We'll need to customize tick values to show only positive values
            yaxis_showticklabels = True
            # Get the range of y-values including offset difference curve
            y_vals_min = offset + df["Y_diff"].min()
            y_vals_max = y_max
            # Create tick values - only show positive values
            tick_vals = []
            tick_texts = []
            # Generate negative ticks but hide labels
            if y_vals_min < 0:
                step = abs(y_vals_min) / 5  # 5 ticks for negative region
                for i in range(6):
                    val = y_vals_min + i * step
                    tick_vals.append(val)
                    if val < 0:
                        tick_texts.append("")  # Hide negative labels
                    else:
                        tick_texts.append(f"{val:.0f}")
            # Generate positive ticks and show labels
            if y_vals_max > 0:
                step = y_vals_max / 5
                for i in range(1, 6):
                    val = i * step
                    tick_vals.append(val)
                    tick_texts.append(f"{val:.0f}")
            yaxis_tickvals = tick_vals
            yaxis_ticktext = tick_texts
        else:  # "All"
            yaxis_showticklabels = True
            yaxis_tickvals = None
            yaxis_ticktext = None

        # Update layout for single plot
        yaxis_config = dict(
            showgrid=show_grid,
            gridwidth=1,
            gridcolor="lightgray",
            showticklabels=yaxis_showticklabels
        )

        if y_axis_labels == "Only Positive" and yaxis_tickvals is not None:
            yaxis_config['tickvals'] = yaxis_tickvals
            yaxis_config['ticktext'] = yaxis_ticktext

        # Configure X-axis
        xaxis_config = dict(
            showgrid=show_grid,
            gridwidth=1,
            gridcolor="lightgray",
            range=x_range
        )

        if x_tickvals is not None:
            xaxis_config['tickvals'] = x_tickvals
            xaxis_config['ticktext'] = x_ticktext

        fig.update_layout(
            title=f"Rietveld Refinement - {title}",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity (a.u.)",
            height=700,
            hovermode="x unified",
            template="plotly_white",
            showlegend=True,
            legend=dict(x=0.7, y=0.95),
            xaxis=xaxis_config,
            yaxis=yaxis_config
        )

        # Display plot with built-in download controls
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="prf_plot",
            config={
                'toImageButtonOptions': {
                    'format': 'png',  # or 'svg', 'jpeg'
                    'filename': f"rietveld_{title.replace(' ', '_')}",
                    'height': 700,
                    'width': 1200,
                    'scale': 2  # Higher quality
                },
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'modeBarButtonsToRemove': []
            }
        )

        # Statistics section
        st.divider()
        st.subheader("📊 Refinement Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(df))
        with col2:
            # Use extracted range if available, otherwise use data range
            if theta_min is not None and theta_max is not None:
                st.metric("2θ Range", f"{int(theta_min)}° - {int(theta_max)}°")
            else:
                st.metric("2θ Range", f"{int(df['2Theta'].min())}° - {int(df['2Theta'].max())}°")
        with col3:
            st.metric("Max Intensity", f"{df['Y_obs'].max():.2f}")
        with col4:
            # Calculate R-factor approximation (simplified)
            residual = np.sum(np.abs(df["Y_diff"]))
            total = np.sum(np.abs(df["Y_obs"]))
            r_factor = (residual / total) * 100 if total > 0 else 0
            st.metric("R-factor (approx.)", f"{r_factor:.2f}%")

        # Show data preview
        st.divider()
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(f"Showing first 50 of {len(df)} rows")

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="prf_data.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your PRF file has the correct FullProf format")

else:
    st.info("👆 Upload a .prf file to get started")
    st.markdown(
        """
    ### About FullProf PRF Files:
    PRF files contain Rietveld refinement results from FullProf software, typically including:
    - **Observed intensity** (Y_obs): Experimental diffraction pattern
    - **Calculated intensity** (Y_calc): Fitted pattern from refinement
    - **Difference curve** (Y_obs - Y_calc): Residuals showing fit quality
    - **Bragg positions**: Peak positions from the refined structure
    """
    )
