import os
import pickle

import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model(model_dir_path):
    with open(f'{model_dir_path}/vectorizer.pkl', 'rb') as pickle_file:
        vectorizer = pickle.load(pickle_file)

    models = {}
    for file_name_with_format in os.listdir(model_dir_path):
        splits = file_name_with_format.split('.')
        file_name, file_format = splits[0], '.'.join(splits[1:])

        if file_format != 'pkl' or file_name == 'vectorizer':
            continue

        with open(f'{model_dir_path}/{file_name_with_format}', 'rb') as pickle_file:
            models[file_name] = pickle.load(pickle_file)

    return vectorizer, models


if __name__ == '__main__':
    st.set_page_config('Medical Review Analyzer')

    model_dir_path = 'data'
    vectorizer, models = load_model(model_dir_path)

    st.markdown(
        "<h1 style='text-align: center;color: #0a2786;font-size: 30px;'>"
        "Condition Analyzer based on Medical Reviews in Social Networks  </h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: left; color: #032aac;font-size: 20px;' >write down your review, then press enter </p>",
        unsafe_allow_html=True
    )
    review = st.text_area('')

    if st.button(label='Run'):
        if len(review) == 0:
            st.markdown("<p style=color: red;>Please fill review area!</p>", unsafe_allow_html=True)
        else:
            feature_vectors = vectorizer.transform([review])
            for model_name, model in models.items():
                predictions = model.predict(feature_vectors)
                condition = predictions[0]
                st.markdown(
                    f"<p style='color: #0a2786; font-size: 20px;'>{model_name}: {condition}</p>",
                    unsafe_allow_html=True
                )
