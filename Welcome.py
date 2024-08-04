import streamlit as st

st.set_page_config(
    layout = 'wide',
    page_title="Welcome"
)

st.title("SmartSolarSizer")

st.image('solarhome.jpg')

st.markdown(
    """
    **Are you a homeowner looking to install solar panels but don't know where to begin?**\n
    SmartSolarSizer is a machine learning-backed tool that provides personlized estimates for your solar power system size and potential savings based on your annual energy requirements, solar potential, and usable roof area.
    To get your estimates, go to the **Estimator** tab and fill out a short survey about your home.
"""
)

st.markdown(
    """
    **Privacy Disclaimers:**
    - Your address and data will **never** be stored.
    - If you don't feel comfortable entering your address, you can enter your city/state or zip code. However, the accuracy of your results may be reduced if you don't enter your full address
    - If you don't feel comfortable answering a question or don't know the answer, select the 'Not sure' option.

"""
)
