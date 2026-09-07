"""Timber pile replacement forecasting pipeline.

Separate from ``src/data``, ``src/features``, and ``src/models`` (the
concrete-pile dashboard's packages) because the timber dataset has an
unrelated schema. The core pipeline here (data_prep, features, model,
scenarios) has no Streamlit dependency; its Streamlit UI lives under
``src/timber/streamlit/`` so the pipeline can be run and analysed on its
own -- see ``src/timber/model.py`` and ``docs/timber_model_results.md``.
"""
