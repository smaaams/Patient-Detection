import streamlit as st
import pandas as pd
from constants import classifier_type_index
from classifiers import PatientDetector

if __name__ == '__main__':
    st.set_page_config('Medical Review Analyzer')

    # model_dir_path = 'linear/data'
    # vectorizer, classifier = load_model(model_dir_path)
    #
    st.markdown(
        "<h1 style='text-align: center;color: #0a2786;font-size: 30px;'>"
        "Condition Analyzer based on Medical Reviews in Social Networks  </h1>",
        unsafe_allow_html=True)

    df = pd.DataFrame({
        'classifier name': list(classifier_type_index.keys()),
        'classifier index': list(classifier_type_index.values())
    })

    # df
    st.markdown(
        "<p style='text-align: left; color: #032aac;font-size: 20px;' >write down your review, then press enter </p>",
        unsafe_allow_html=True)

    review = st.text_input('')

    st.markdown(
        "<p style='text-align: left; color: #032aac;font-size: 20px;' >Which classifier do you want to use   </p>",
        unsafe_allow_html=True)
    option = st.selectbox(
        '',
        df['classifier name'])
    st.markdown(
        "<p style='text-align: left; color: #032aac;font-size: 20px;' >input review: {} </p>".format(review),
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: left; color: #032aac;font-size: 20px;' >you selected: {} </p>".format(option),
        unsafe_allow_html=True)

    if len(review) > 0:
        st.markdown(
            "<p style='text-align: left; color: #5b0475;font-size: 20px;' >to run classifier on chosen model for given "
            "review press below button</p>",
            unsafe_allow_html=True)

        run_model = st.button(label='Run')

        if run_model:
            model = PatientDetector(classifier_type_index[option])
            condition = model.query(review)
            st.markdown(
                "<p style='text-align: left; color: #035e17;font-size: 20px;' >detected condition: {}</p>".format(
                    condition),
                unsafe_allow_html=True)
