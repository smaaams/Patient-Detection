import pickle

import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model(model_dir_path):
    with open(f'{model_dir_path}/vectorizer.pkl', 'rb') as pickle_file:
        vectorizer = pickle.load(pickle_file)
    with open(f'{model_dir_path}/model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    return vectorizer, model


if __name__ == '__main__':
    st.set_page_config('Medical Review Analyzer')

    model_dir_path = 'data'
    vectorizer, model = load_model(model_dir_path)

    st.markdown(
        "<h1 style='text-align: center;color: #0a2786;font-size: 30px;'>"
        "Condition Analyzer based on Medical Reviews in Social Networks  </h1>",
        unsafe_allow_html=True)

    review = st.text_area(label='write down your review, then press enter')

    if st.button(label='Run'):
        if len(review) == 0:
            st.markdown("<p style=color: red;>Please fill review area!</p>", unsafe_allow_html=True)
        else:
            feature_vectors = vectorizer.transform([review])
            predictions = model.predict(feature_vectors)
            condition = predictions[0]
            st.markdown(
                f"<p style='text-align: left; color: #035e17;font-size: 20px;' >detected condition: {condition}</p>",
                unsafe_allow_html=True
            )
