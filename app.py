import streamlit as st
from config import setup_page

setup_page("HyDEX")

pages = {
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
    ],
    "XRD Tools": [
        st.Page("pages/xrd_page.py", title="XRD Data (XY format)", icon="📊"),
        st.Page("pages/prf_page.py", title="FullProf PRF Visualizer", icon="📈"),
    ],
    "Electrochemistry Tools": [
        st.Page("pages/cd_page.py", title="Charge-Discharge Data", icon="⚡"),
        st.Page("pages/cva_page.py", title="CVA Data", icon="🔬"),
    ]
}

pg = st.navigation(pages)
pg.run()
