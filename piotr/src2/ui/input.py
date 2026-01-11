import streamlit as st

def sidebar_value_agg_func():
    return st.sidebar.selectbox(
        'Value aggregation function',
        ['Mean', 'Median', 'Min', 'Max', 'Sum', 'Size']
    )
