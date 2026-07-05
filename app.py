import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="AI 날씨 예측기", page_icon="🌤️")
st.title("🌤️ 인공지능 실시간 눈/비 예측기")
st.markdown("머신러닝이 강수 여부를 예측하고, 날짜와 기온을 분석해 눈인지 비인지 판독합니다.")
st.divider()

@st.cache_resource
def load_model():
    return joblib.load('rain_model_light.pkl')

model = load_model()

st.subheader("현재 날씨 상태를 입력하세요")
# 1. 달력 입력기 추가 (날짜를 선택받음)
today_date = st.date_input("예측할 날짜를 선택하세요", datetime.today())

col1, col2 = st.columns(2)

with col1:
    # 겨울철 온도를 테스트하기 위해 최저기온 범위를 -20도까지 넓혔습니다.
    temp = st.slider("기온 (℃)", min_value=-20.0, max_value=40.0, value=15.0)
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
    
    # AI의 1차 판단 (비가 오는가? 1=옴, 0=안 옴)
    result = model.predict(input_df)[0]
    
    st.divider()
    
    if result == 1:
        # 2. 눈/비 판독 로직 
        # 선택한 날짜의 '월(month)'과 '기온'을 확인합니다.
        # 기상학적으로 기온이 2~3도 이하이면서 겨울철(11월~3월)이면 눈으로 판독합니다.
        month = today_date.month
        
        if temp <= 2.0 and (month in [11, 12, 1, 2, 3]):
            st.info(f"❄️ **눈이 내릴 확률이 높습니다!** ({today_date.strftime('%m월 %d일')}) 빙판길에 주의하세요.")
        else:
            st.error(f"☔ **비가 내릴 확률이 높습니다!** ({today_date.strftime('%m월 %d일')}) 우산을 챙기세요.")
            
    else:
        st.success(f"☀️ **날씨가 맑습니다!** ({today_date.strftime('%m월 %d일')}) 강수 확률이 낮습니다.")
