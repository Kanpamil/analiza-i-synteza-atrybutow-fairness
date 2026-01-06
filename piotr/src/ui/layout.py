import streamlit as st

def column_comparison(callback):
    columns = st.columns(2 if st.session_state.is_comparison else 1)

    for id, column in enumerate(columns):
        with column:
            callback(id)
