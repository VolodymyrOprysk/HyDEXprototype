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

setup_page("Charge-Discharge Analyzer (new)")

st.title("⚡ Charge-Discharge Analyzer")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

    st.subheader("Calculation Parameters")
    current_density = st.number_input(
        "Current Density (mA/g)",
        min_value=0.0,
        value=50.0,
        step=10.0,
        help="Enter the current density used in the experiment",
    )

    st.subheader("Y-axis Scale (Cyclic Stability)")
    auto_scale = st.checkbox(
        "Auto Scale",
        value=True,
        help="Automatically scale Y-axis based on data range",
    )

    if not auto_scale:
        y_min = st.number_input(
            "Y-axis Min (mAh/g)",
            value=0.0,
            step=10.0,
            help="Minimum value for Y-axis",
        )
        y_max = st.number_input(
            "Y-axis Max (mAh/g)",
            value=500.0,
            step=10.0,
            help="Maximum value for Y-axis",
        )

if uploaded_file is not None:
    # Read the data
    try:
        # Read the file content
        content = uploaded_file.read().decode("utf-8")

        # Try to read with different delimiters (header in first row)
        for delimiter in ["\t", r"\s+", ",", ";"]:
            try:
                df = pd.read_csv(
                    StringIO(content),
                    sep=delimiter,
                    header=0,
                    skipinitialspace=True,
                    engine="python" if delimiter == r"\s+" else "c",
                )
                if len(df.columns) >= 3 and len(df) > 0:
                    # Rename columns to standard names if needed
                    if len(df.columns) >= 3:
                        df.columns = ["Time", "Voltage", "Current"] + list(
                            df.columns[3:]
                        )
                    break
            except:
                continue

        st.success(f"✓ File loaded successfully! {len(df)} data points")

        # Classify each row by current type
        # Charge: negative current, Pause: zero or very small current, Discharge: positive current
        current_threshold = 0.01  # mA threshold to consider as "zero" current

        def classify_current(current):
            if current < -current_threshold:
                return "charge"
            elif current > current_threshold:
                return "discharge"
            else:
                return "pause"

        df["Phase"] = df["Current"].apply(classify_current)

        # Detect phase changes to identify segment boundaries
        phase_changes = np.where(df["Phase"].ne(df["Phase"].shift()))[0]
        segment_boundaries = [0] + list(phase_changes) + [len(df)]

        # Create segments with their phase type
        segments = []
        for i in range(len(segment_boundaries) - 1):
            start_idx = segment_boundaries[i]
            end_idx = segment_boundaries[i + 1]
            segment_data = df.iloc[start_idx:end_idx].copy()

            if len(segment_data) > 0:
                phase_type = segment_data["Phase"].iloc[0]
                segments.append({
                    "type": phase_type,
                    "data": segment_data,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })

        # Group segments into cycles (charge -> pause -> discharge pattern)
        cycles = []
        cycle_capacities = []
        cycle_times = []
        cycle_discharge_data = []
        cycle_charge_data = []  # Store charge data for each cycle

        i = 0
        cycle_num = 0

        while i < len(segments):
            # Look for a cycle pattern: charge -> pause -> discharge
            # Or simplified: charge -> discharge (pause is optional)

            if segments[i]["type"] == "charge":
                charge_segment = segments[i]
                pause_segment = None
                discharge_segment = None

                # Look ahead for pause and/or discharge
                next_idx = i + 1

                # Check for optional pause
                if next_idx < len(segments) and segments[next_idx]["type"] == "pause":
                    pause_segment = segments[next_idx]
                    next_idx += 1

                # Check for discharge
                if next_idx < len(segments) and segments[next_idx]["type"] == "discharge":
                    discharge_segment = segments[next_idx]

                    # Found a complete cycle
                    cycle_num += 1

                    # Calculate discharge capacity from discharge segment only
                    discharge_data = discharge_segment["data"]
                    charge_data = charge_segment["data"]

                    time_start = discharge_data["Time"].values[0]
                    time_end = discharge_data["Time"].values[-1]
                    time_diff_hours = (time_end - time_start) / 60  # convert min to hours

                    capacity = time_diff_hours * current_density  # mAh/g

                    cycles.append(cycle_num)
                    cycle_capacities.append(capacity)
                    cycle_times.append(time_diff_hours)
                    cycle_discharge_data.append(discharge_data.copy())
                    cycle_charge_data.append(charge_data.copy())

                    # Move index past this complete cycle
                    i = next_idx + 1
                    continue

            # If no cycle pattern found, move to next segment
            i += 1

        # Plot cyclic stability at the top if multiple cycles detected
        if len(cycles) > 1:
            st.subheader("Cyclic Stability")
            st.caption("💡 Click on any point to view the discharge curve for that cycle")

            # Create two columns: 2/3 for plot, 1/3 for table
            plot_col, table_col = st.columns([2, 1])

            with plot_col:
                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=cycle_capacities,
                        mode="lines+markers",
                        line=dict(color="purple", width=2),
                        marker=dict(size=8),
                        name="Discharge Capacity",
                        customdata=cycles,
                    )
                )

                # Configure Y-axis range based on user settings
                yaxis_config = dict(title="Specific Discharge Capacity (mAh/g)")
                if not auto_scale:
                    yaxis_config["range"] = [y_min, y_max]

                fig3.update_layout(
                    title="Cyclic Stability (Click on a point)",
                    xaxis_title="Cycle Number",
                    yaxis=yaxis_config,
                    hovermode="x unified",
                    template="plotly_white",
                )

                selected_point = st.plotly_chart(fig3, use_container_width=True,
                                                  on_select="rerun", key="cycle_stability_plot")

                # Handle point selection
                if selected_point and "selection" in selected_point:
                    selection = selected_point["selection"]
                    if selection and "points" in selection and len(selection["points"]) > 0:
                        selected_cycle_idx = selection["points"][0]["point_index"]
                        st.session_state["selected_cycle"] = selected_cycle_idx

            with table_col:
                # Display per-cycle discharge data table
                st.write("**Per-Cycle Data:**")
                cycle_df = pd.DataFrame({
                    'Cycle': cycles,
                    'Time (h)': [f"{t:.2f}" for t in cycle_times],
                    'Capacity (mAh/g)': [f"{c:.2f}" for c in cycle_capacities]
                })
                st.dataframe(cycle_df, use_container_width=True, hide_index=True, height=400)

            # Display charge and discharge curves side by side below if a cycle is selected
            if "selected_cycle" in st.session_state and st.session_state["selected_cycle"] is not None:
                selected_idx = st.session_state["selected_cycle"]

                if 0 <= selected_idx < len(cycle_discharge_data):
                    selected_discharge_data = cycle_discharge_data[selected_idx]
                    selected_charge_data = cycle_charge_data[selected_idx]
                    selected_cycle_num = cycles[selected_idx]

                    st.divider()
                    st.subheader(f"Cycle {selected_cycle_num} - Charge & Discharge Curves")

                    # Create two columns for charge and discharge curves
                    charge_col, discharge_col = st.columns(2)

                    with charge_col:
                        # Create charge curve plot
                        fig_charge = go.Figure()

                        # Normalize time to start from 0 for better visualization
                        time_normalized_charge = selected_charge_data["Time"].values - selected_charge_data["Time"].values[0]

                        fig_charge.add_trace(
                            go.Scatter(
                                x=time_normalized_charge,
                                y=selected_charge_data["Voltage"].values,
                                mode="lines",
                                line=dict(color="blue", width=2),
                                name="Charge",
                            )
                        )

                        fig_charge.update_layout(
                            title="Charge Curve",
                            xaxis_title="Time (min)",
                            yaxis_title="Voltage (V)",
                            hovermode="x unified",
                            template="plotly_white",
                        )

                        st.plotly_chart(fig_charge, use_container_width=True)

                    with discharge_col:
                        # Create discharge curve plot
                        fig_discharge = go.Figure()

                        # Normalize time to start from 0 for better visualization
                        time_normalized_discharge = selected_discharge_data["Time"].values - selected_discharge_data["Time"].values[0]

                        fig_discharge.add_trace(
                            go.Scatter(
                                x=time_normalized_discharge,
                                y=selected_discharge_data["Voltage"].values,
                                mode="lines",
                                line=dict(color="green", width=2),
                                name="Discharge",
                            )
                        )

                        fig_discharge.update_layout(
                            title="Discharge Curve",
                            xaxis_title="Time (min)",
                            yaxis_title="Voltage (V)",
                            hovermode="x unified",
                            template="plotly_white",
                        )

                        st.plotly_chart(fig_discharge, use_container_width=True)

                    # Add button to clear selection
                    if st.button("Clear Selection"):
                        st.session_state["selected_cycle"] = None
                        st.rerun()
            else:
                st.info("👆 Click on a point in the Cyclic Stability plot to view the charge and discharge curves")

        elif len(cycles) == 1:
            st.info(
                "Single cycle detected. Upload multi-cycle data to see cyclic stability."
            )

        st.divider()

        # Analysis results and statistics section
        st.divider()
        st.subheader("📊 Overall Statistics")

        if len(cycle_capacities) > 0:
            # Show statistics in columns
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Total Cycles", len(cycles))
            with stat_col2:
                st.metric("Max Capacity", f"{max(cycle_capacities):.2f} mAh/g")
            with stat_col3:
                st.metric("Min Capacity", f"{min(cycle_capacities):.2f} mAh/g")
            with stat_col4:
                st.metric("Avg Capacity", f"{np.mean(cycle_capacities):.2f} mAh/g")
        else:
            st.warning("No discharge cycles detected")

        st.divider()

        # Voltage vs Time plot at the bottom
        st.subheader("Charge-Discharge Curves")
        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["Voltage"],
                mode="lines",
                name="Voltage",
                line=dict(color="blue", width=2),
            )
        )

        fig1.update_layout(
            title="Voltage vs Time (All Cycles)",
            xaxis_title="Time (min)",
            yaxis_title="Voltage (V)",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Show data preview
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(df)} rows")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your file has 3 columns: Time, Voltage, Current")

else:
    st.info("👆 Upload a .txt file to get started")
    st.markdown(
        """
    ### Expected file format:
    - **Column 1:** Time (minutes)
    - **Column 2:** Voltage (V)
    - **Column 3:** Current (mA)
        - Negative values = Charge
        - Positive values = Discharge

    Supported delimiters: tab, space, comma, semicolon
    """
    )
