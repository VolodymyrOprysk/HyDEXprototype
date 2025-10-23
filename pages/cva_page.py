import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import setup_page

setup_page("CVA Data")

st.title("🔬 Cyclic Voltammetry Analysis (CVA)")
st.markdown("Analyze cyclic voltammetry data and electrochemical processes")

st.info("🚧 This page is under construction. CVA analysis features coming soon!")

st.markdown("""
### Planned Features:
- CV curve visualization
- Peak detection and analysis
- Redox potential calculations
- Multi-cycle comparison
- Export and reporting tools
""")
