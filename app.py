import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="AI 비 예측기", page_icon="🌦️")
st.title("🌦️ 인공지능 실시간 비 예측기")
st.markdown("기상청 데이터를 학습한 머신러닝 모델이 현재 날씨를 분석해 비가 올지 예측합니다.")
st.divider()

@st.cache_resource
def load_model():
    return joblib.load('rain_model.pkl')

model = load_model()

st.subheader("현재 날씨 상태를 입력하세요")
col1, col2 = st.columns(2)

with col1:
    temp = st.slider("기온 (℃)", min_value=-10.0, max_value=35.0, value=15.0)
    pressure = st.slider("현지기압 (hPa)", min_value=950.0, max_value=1050.0, value=1000.0)
    humidity = st.slider("습도 (%)", min_value=0.0, max_value=100.0, value=60.0)

with col2:
    wind_speed = st.slider("풍속 (m/s)", min_value=0.0, max_value=20.0, value=2.0)
    wind_dir = st.slider("풍향 (deg)", min_value=0.0, max_value=360.0, value=180.0)

if st.button("결과 예측하기", type="primary"):
    input_df = pd.DataFrame({
        '풍향(deg)': [wind_dir],
        '풍속(m/s)': [wind_speed],
        '기온(℃)': [temp],
        '습도(%)': [humidity],
        '현지기압(hPa)': [pressure]
    })
    
    result = model.predict(input_df)[0]
    
    st.divider()
    if result == 1:
        st.error("☔ **우산을 챙기세요!** 현재 조건에서는 비가 내릴 확률이 매우 높습니다.")
    else:
        st.success("☀️ **날씨가 맑습니다!** 비가 오지 않을 것으로 예상됩니다.")
