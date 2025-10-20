import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Discharge Curve Analyzer", layout="wide")

st.title("⚡ Discharge Curve Analyzer")
st.markdown("Upload your discharge curve data and analyze battery performance")

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

    ignore_first_cycle = st.checkbox(
        "Ignore First Cycle",
        value=False,
        help="Exclude the first cycle from analysis (useful if first cycle is formation)",
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

        # Separate charge and discharge for calculations
        df_discharge = df[df["Current"] > 0].copy()
        df_charge = df[df["Current"] < 0].copy()

        # Detect cycles by finding when current changes sign
        # Cycling starts with charge (negative current), followed by discharge (positive current)
        current_sign = np.sign(df["Current"])
        sign_changes = np.where(np.diff(current_sign) != 0)[0] + 1

        # Split data into segments (charge or discharge)
        segment_boundaries = [0] + list(sign_changes) + [len(df)]

        cycles = []
        cycle_capacities = []
        cycle_times = []
        cycle_discharge_data = []  # Store discharge data for each cycle

        # Group segments into cycles
        # A cycle can start with either charge or discharge
        i = 0
        cycle_num = 0
        while i < len(segment_boundaries) - 1:
            segment_data = df.iloc[segment_boundaries[i] : segment_boundaries[i + 1]]

            if len(segment_data) > 0:
                # Check if this is a charge segment (negative current)
                if segment_data["Current"].iloc[0] < 0:
                    # This is a charge phase - look for the following discharge phase
                    if i + 1 < len(segment_boundaries) - 1:
                        discharge_data = df.iloc[segment_boundaries[i + 1] : segment_boundaries[i + 2]]

                        if len(discharge_data) > 0 and discharge_data["Current"].iloc[0] > 0:
                            # Found a complete charge-discharge cycle
                            cycle_num += 1

                            # Calculate discharge capacity: C = Δt × I
                            time_start = discharge_data["Time"].values[0]
                            time_end = discharge_data["Time"].values[-1]
                            time_diff_hours = (time_end - time_start) / 60  # convert min to hours

                            capacity = time_diff_hours * current_density  # mAh/g

                            cycles.append(cycle_num)
                            cycle_capacities.append(capacity)
                            cycle_times.append(time_diff_hours)
                            cycle_discharge_data.append(discharge_data.copy())

                            i += 2  # Skip both charge and discharge segments
                            continue

                # Check if this is a discharge segment (positive current)
                elif segment_data["Current"].iloc[0] > 0:
                    # This is a discharge phase (may be first cycle without initial charge)
                    cycle_num += 1

                    # Calculate discharge capacity: C = Δt × I
                    time_start = segment_data["Time"].values[0]
                    time_end = segment_data["Time"].values[-1]
                    time_diff_hours = (time_end - time_start) / 60  # convert min to hours

                    capacity = time_diff_hours * current_density  # mAh/g

                    cycles.append(cycle_num)
                    cycle_capacities.append(capacity)
                    cycle_times.append(time_diff_hours)
                    cycle_discharge_data.append(segment_data.copy())

            i += 1

        # Apply filter to ignore first cycle if requested
        if ignore_first_cycle and len(cycles) > 0:
            cycles = cycles[1:]
            cycle_capacities = cycle_capacities[1:]
            cycle_times = cycle_times[1:]
            cycle_discharge_data = cycle_discharge_data[1:]
            # Renumber cycles to start from 1
            cycles = list(range(1, len(cycles) + 1))

        # Plot cyclic stability at the top if multiple cycles detected
        if len(cycles) > 1:
            st.subheader("Cyclic Stability")
            st.caption("💡 Click on any point to view the discharge curve for that cycle")

            # Create two columns for cyclic stability and discharge curve
            cycle_col1, cycle_col2 = st.columns([1, 1])

            with cycle_col1:
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

                fig3.update_layout(
                    title="Cyclic Stability (Click on a point)",
                    xaxis_title="Cycle Number",
                    yaxis_title="Specific Discharge Capacity (mAh/g)",
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

            with cycle_col2:
                # Display discharge curve if a cycle is selected
                if "selected_cycle" in st.session_state and st.session_state["selected_cycle"] is not None:
                    selected_idx = st.session_state["selected_cycle"]

                    if 0 <= selected_idx < len(cycle_discharge_data):
                        selected_data = cycle_discharge_data[selected_idx]
                        selected_cycle_num = cycles[selected_idx]

                        # Create discharge curve plot
                        fig_discharge = go.Figure()

                        # Normalize time to start from 0 for better visualization
                        time_normalized = selected_data["Time"].values - selected_data["Time"].values[0]

                        fig_discharge.add_trace(
                            go.Scatter(
                                x=time_normalized,
                                y=selected_data["Voltage"].values,
                                mode="lines",
                                line=dict(color="green", width=2),
                                name=f"Cycle {selected_cycle_num}",
                            )
                        )

                        fig_discharge.update_layout(
                            title=f"Discharge Curve - Cycle {selected_cycle_num}",
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
                    st.info("👈 Click on a point in the Cyclic Stability plot to view the discharge curve")

        elif len(cycles) == 1:
            st.info(
                "Single cycle detected. Upload multi-cycle data to see cyclic stability."
            )

        st.divider()

        # Analysis results and statistics section
        st.subheader("📊 Analysis Results")

        # Show cycle information
        if len(cycles) > 0:
            st.metric("Total Cycles", len(cycles))

        # Display individual cycle capacities with discharge time
        if len(cycle_capacities) > 0:
            st.divider()
            st.write("**Per-Cycle Discharge Data:**")

            # Create a dataframe for better display
            cycle_df = pd.DataFrame({
                'Cycle': cycles,
                'Time (h)': [f"{t:.2f}" for t in cycle_times],
                'Capacity (mAh/g)': [f"{c:.2f}" for c in cycle_capacities]
            })
            st.dataframe(cycle_df, use_container_width=True, hide_index=True)

            st.divider()
            st.write("**Overall Statistics:**")
            st.write(f"• Max capacity: {max(cycle_capacities):.2f} mAh/g")
            st.write(f"• Min capacity: {min(cycle_capacities):.2f} mAh/g")
            st.write(f"• Average capacity: {np.mean(cycle_capacities):.2f} mAh/g")
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
