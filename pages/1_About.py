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