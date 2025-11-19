from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #F8F8F8;
        padding: 20px;
        border-radius: 15px;
    }

    .title-box {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        padding: 15px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 28px;
        margin-bottom: 20px;
    }

    .sub-box {
        background: #33FF00;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: black;
    }

    .predict-box {
        background: #FF69B4;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
    }

    .iris-card {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------

st.markdown('<div class="title-box">🌸 ระบบจำแนกดอกไม้ (Iris Classification)</div>', unsafe_allow_html=True)

# รูปตรงกลาง
col1, col2, col3 = st.columns([1.7,1,1])
with col2:
    st.image("./img/Tanapon.jpg", width=150)

# ชื่อกลาง
st.markdown("<h4 style='text-align:center;'>by Tanapon</h4>", unsafe_allow_html=True)
st.markdown("---")


# ------------------ รูปภาพดอกไม้ ------------------
st.header("📌 ตัวอย่างข้อมูลดอกไม้")

col1, col2, col3 = st.columns(3)

with col1:
   st.markdown('<div class="iris-card" style=color:black><h4>Versicolor</h4>', unsafe_allow_html=True)
   st.image("./img/iris1.jpg")
   st.markdown("</div>", unsafe_allow_html=True)

with col2:
   st.markdown('<div class="iris-card" style=color:black><h4>Virginica</h4>', unsafe_allow_html=True)
   st.image("./img/iris2.jpg")
   st.markdown("</div>", unsafe_allow_html=True)

with col3:
   st.markdown('<div class="iris-card" style=color:black><h4>Setosa</h4>', unsafe_allow_html=True)
   st.image("./img/iris3.jpg")
   st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ สถิติข้อมูล ------------------
st.markdown('<div class="sub-box">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/iris.csv")
st.dataframe(dt.head(10))

dx = [
    dt['petallength'].sum(),
    dt['petalwidth'].sum(),
    dt['sepallength'].sum(),
    dt['sepalwidth'].sum()
]
dx2 = pd.DataFrame(dx, index=["petal length", "petal width", "sepal length", "sepal width"])

if st.button("📌 แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
else:
    st.write("กดปุ่มด้านบนเพื่อแสดงข้อมูล")

st.markdown("---")

# ------------------ ทำนายข้อมูล ------------------
st.markdown('<div class="predict-box">🔮 ระบบทำนายข้อมูล</div>', unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("เลือกค่า **petallength**", 0.0, 7.0, 1.0)
pt_wd  = st.slider("เลือกค่า **petalwidth**", 0.0, 3.0, 0.5)

sp_len = st.number_input("กรุณากรอกค่า **sepallength**", 0.0, 10.0, 5.0)
sp_wd  = st.number_input("กรุณากรอกค่า **sepalwidth**", 0.0, 5.0, 3.0)

st.markdown("")

if st.button("🔍 ทำนายผลดอกไม้"):
   dt = pd.read_csv("./data/iris.csv")
   X = dt.drop('variety', axis=1)
   y = dt['variety']

   model = KNeighborsClassifier(n_neighbors=3)
   model.fit(X, y)

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
   output = model.predict(x_input)

   st.success(f"🌼 ผลการทำนาย: **{output[0]}**")

   if output[0] == 'Setosa':
        st.image("./img/iris3.jpg", caption="Setosa")
   elif output[0] == 'Versicolor':
        st.image("./img/iris1.jpg", caption="Versicolor")
   else:
        st.image("./img/iris2.jpg", caption="Virginica")
else:
    st.info("กรุณากรอกข้อมูลและกดปุ่มเพื่อทำนาย")

st.markdown("---")
st.markdown("<center>© 2025 Tanapon | Streamlit Machine Learning App</center>", unsafe_allow_html=True)
