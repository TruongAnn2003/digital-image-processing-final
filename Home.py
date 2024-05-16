import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="⭐",
)

st.markdown(
    """
    <h1 style="text-align: center;">👋 Welcome to 👋<br/>Digital Image Processing<br/> Final Project! </h1>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://heeap.org/sites/default/files/pictures/hcmute.jpg" alt="HCMUTE" width="500">
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.success("Select a demo above.")
