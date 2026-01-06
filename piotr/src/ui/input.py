import streamlit as st
from core.metrics import ClassReport, FairReport

def metric_selection(id_):
    settings = st.expander('Settings')

    with settings:
        category = st.radio(
            'Category',
            ['Classification', 'Fairness'],
            horizontal=True,
            key=f'category{id_}'
        )

        is_classification = category == 'Classification'

        metric_name = st.selectbox(
            'Metric',
            ClassReport.available_metrics() if is_classification else FairReport.available_metrics(),
            key=f'metric_name{id_}'
        )

    return is_classification, metric_name, settings
