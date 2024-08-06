import streamlit as st

st.set_page_config(
    layout = 'wide',
    page_title="About"
)

st.title("SmartSolarSizer")

st.title("MIDS Capstone Project Summer 2024")

st.header("Mission Statement")
st.markdown(
    """
    **Our mission** is to streamline homeowners' adoption of solar energy by providing an unbiased, data-driven solar system recommendation, resulting in a reduction of residential carbon footprints and accelerating us towards a more sustainable future!
"""
)

st.header("Team")
team_members = [{"name":"Theresa Azinge",
                 "title":"Energy Model Developer",
                 "photo": "theresa.jpg"},
                 {"name": "Madeline Bourquin",
                  "title": "Project Manager\nUI Developer",
                  "photo": "madi.jpg"},
                  {"name": "Naikaj Pandya",
                   "title": "Product Manager\nSolution Architect",
                   "photo": "naikaj.jpg"},
                   {"name": "Mohammad Kanawati",
                   "title": "Solar Model Developer",
                   "photo": "moh.jpg"},
                   {"name": "Nathan Eppler",
                   "title": "Solar Model Developer",
                   "photo": "nathan.jpg"}]

cols = st.columns(5)
for idx, member in enumerate(team_members):
    with cols[idx]:
        st.image(member["photo"], width=100, use_column_width=True)
        st.markdown(f"<p style='text-align: center;'><strong>{member['name']}</strong><br>{member['title']}</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    img {
        border-radius: 50%;
    }
    .caption {
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Data")
st.markdown(
    """
    Our data comes from the **National Renewable Energy Laboratory (NREL)**'s Residential Building Simulation Database and the Solar Radiation Database. NREL is an organization under the U.S Department of Energy and provides comprehensive and authoritative datasets critical for advancing renewable energy technologies and energy efficiency.\n
The Residential Building Simulation Database offers detailed information on energy use and efficiency in residential buildings, supporting analysis and modeling efforts aimed at improving home energy performance. The database is publicly available and consists building simulation outputs for residential stock of the whole country. The data is based on a decade worth of various national studies conducted by NREL in conjunction with multiple other government agencies.\n
The Solar Radiation Database, part of the National Solar Radiation Database (NSRDB), supplies high-resolution solar radiation and meteorological data spanning the entire continental US from over two decades (1998-2019) at 30 min temporal resolution, essential for assessing solar energy potential and planning solar installations. These datasets are publicly accessible and maintained with rigorous standards, ensuring their reliability and relevance for research, development, and practical applications in the renewable energy sector.
"""
)


