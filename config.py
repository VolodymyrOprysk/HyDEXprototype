import streamlit as st

def setup_page(page_title="HyDEX"):
    """Common page configuration for all HyDEX pages"""
    st.set_page_config(
        page_title=page_title,
        page_icon="⚡",
        layout="wide"
    )
