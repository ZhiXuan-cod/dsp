import streamlit as st
import base64

# --- Page config ---
st.set_page_config(
    page_title="No Code Platform For Machine Learning",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to set background image ---
def set_bg(Front page.png):
    with open(Front page.png, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# --- Set Canva image as background ---
set_bg("Front page.png")   # your Canva exported image

# --- Empty space to push button down ---
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

# --- Center button ---
st.markdown("""
<style>

.button-container{
    position:absolute;
    top:72%;
    left:50%;
    transform:translate(-50%,-50%);
}

/* invisible button */
.overlay-button{
    width:220px;
    height:70px;
    background:transparent;
    border:none;
    cursor:pointer;
}

</style>

<div class="button-container">
<a href="/main_page" target="_self">
<button class="overlay-button"></button>
</a>
</div>

""", unsafe_allow_html=True)