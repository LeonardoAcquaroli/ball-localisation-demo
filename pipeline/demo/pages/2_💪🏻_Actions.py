import streamlit as st

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    .top-center-content {
        text-align: center;
        padding-top: 2rem;
    }
    .big-header {
        font-size: 175px;
        margin-bottom: 0;
        color: #1b8cab;
        text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.3);
    }
    .subtext {
        font-size: 40px;
        margin-top: auto;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .mid-text {
        font-size: 20px;
        margin-top: auto;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .arrow-text {
        font-size: 32px;
        font-weight: bold;
        margin-top: auto;
        color: #1b8cab;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '''
    <div class="top-center-content">
        <h1 class="big-header">2h 8m</h1>
        <p class="subtext">Time saved</p>
        <p class="mid-text">for tagging a match without annotating the ball position.</p>
        <p class="mid-text">Estimated reduction of 45%</p>
        <p class="arrow-text">4h 45m → 2h 37m</p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.write("---")  # Add some vertical space between images

st.markdown("# Deployment options 🚀")
st.markdown("### Cloud managed endpoint")
col1, col2 = st.columns(2)
with col1:
    st.image(r"C:\Users\leoac\vtg-automation\ball_position_estimation\pipeline\images\vertexai1.jpg", caption="Vertex AI")
with col2:
    st.image(r"C:\Users\leoac\vtg-automation\ball_position_estimation\pipeline\images\sm.png", caption="Sagemaker")
st.image(r"C:\Users\leoac\vtg-automation\ball_position_estimation\pipeline\images\API_workflow1.png", caption="API workflow")


st.markdown("### Local deployment")
st.write("Load models with the tagging application.")
st.image(r"C:\Users\leoac\vtg-automation\ball_position_estimation\pipeline\images\laptop.png", caption="Local")



st.markdown("### Self managed endpoint")
st.write("Host a machine with the tagging models.")
st.image(r"C:\Users\leoac\vtg-automation\ball_position_estimation\pipeline\images\server.png", caption="Self managed server")