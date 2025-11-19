from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ================================
#   PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Iris Classification by Tanapon",
    page_icon="🌸",
    layout="wide",
)

# ================================
#   CUSTOM CSS
# ================================
st.markdown("""
<style>
    .title-main {
        font-size: 38px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        padding: 10px;
    }
    .section-box {
        background-color: #F2F4F4;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 0px 10px #D5D8DC;
        margin-bottom: 30px;
    }
    .section-title {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #117A65;
    }
</style>
""", unsafe_allow_html=True)

# ================================
#   HEADER
# ================================
st.markdown("<div class='title-main'>🌸 Tanapon – ระบบจำแนกข้อมูลดอกไม้ (Iris)</div>", unsafe_allow_html=True)
st.image("./img/Tanapon.jpg", width=250)

# ================================
#   FLOWER IMAGES SECTION
# ================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>ตัวอย่างดอกไม้ทั้ง 3 ประเภท</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.image("./img/iris1.jpg", caption="Versicolor")

with col2:
    st.image("./img/iris2.jpg", caption="Verginica")

with col3:
    st.image("./img/iris3.jpg", caption="Setosa")

st.markdown("</div>", unsafe_allow_html=True)

# ================================
#   DATA STATISTICS
# ================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>📊 สถิติข้อมูลดอกไม้</p>", unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["petallength", "petalwidth", "sepallength", "sepalwidth"])

if st.button("แสดงการจินตทัศน์ข้อมูล (Bar Chart)"):
    st.bar_chart(dx2)
else:
    st.info("คลิกปุ่มด้านบนเพื่อแสดงข้อมูล")

st.markdown("</div>", unsafe_allow_html=True)

# ================================
#   PREDICT SECTION
# ================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>🔍 ทำนายข้อมูลดอกไม้</p>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    pt_len = st.slider("เลือก petallength", 0.0, 10.0, 1.0)
    pt_wd  = st.slider("เลือก petalwidth", 0.0, 10.0, 1.0)

with colB:
    sp_len = st.number_input("เลือก sepallength", 0.0, 10.0, 1.0)
    sp_wd  = st.number_input("เลือก sepalwidth", 0.0, 10.0, 1.0)

if st.button("ทำนายผล"):
    dt = pd.read_csv("./data/iris.csv")
    X = dt.drop('variety', axis=1)
    y = dt.variety

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = model.predict(x_input)

    st.subheader(f"ผลการทำนาย: **{result[0]}** 🌸")

    if result[0] == 'Setosa':
        st.image("./img/iris3.jpg", width=250)
    elif result[0] == 'Versicolor':
        st.image("./img/iris1.jpg", width=250)
    else:
        st.image("./img/iris2.jpg", width=250)
else:
    st.info("กรอกข้อมูลให้ครบ แล้วกดปุ่มเพื่อทำนายผล")

st.markdown("</div>", unsafe_allow_html=True)
