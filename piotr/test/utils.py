import sys
from pathlib import Path
import streamlit as st

def fix_imports():
    sys.path.append(str(Path(__file__).parent.parent / 'src'))

def suppress_streamlit_cache():
    st.cache_data = lambda f=None, **k: f if f else lambda x: x
