import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import datetime

# -------------------------------
# 전역 설정
# -------------------------------
st.set_page_config(page_title="대한민국 연안 해수면 상승 대시보드", layout="wide")

# Pretendard 폰트 적용 (없으면 무시)
plt.rcParams['font.family'] = 'Pretendard' if 'Pretendard' in plt.rcParams['font.family'] else 'Malgun Gothic'

st.title("🌊 대한민국 연안 해수면 상승 대시보드")

# -------------------------------
# 탭 구성
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 공식 데이터", "✍ 사용자 데이터", "🖼 설명 + 정적 그래프"])

# -------------------------------
# 탭 1: 공식 데이터
# -------------------------------
with tab1:
    st.subheader("📊 공식 데이터 기반 시각화")

    # 데이터 출처: NOAA Global Mean Sea Level (예시 CSV 링크)
    # URL: https://www.star.nesdis.noaa.gov/sod/lsa/SeaLevelRise/LSA_SLR_timeseries_global.php
    try:
        df = pd.read_csv("gmsl.csv")
        df = pd.read_csv(url)
        df = df.rename(columns={"Time": "date", "GMSL": "value"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df[df["date"] <= pd.Timestamp(datetime.date.today())]

        fig = px.line(df, x="date", y="value",
                      labels={"date": "연도", "value": "해수면(mm)"},
                      title="전지구 평균 해수면 상승 (NOAA)")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 전처리된 CSV 다운로드", df.to_csv(index=False), "official_data.csv", "text/csv")
    except Exception as e:
        st.error(f"데이터 불러오기 실패: {e}")
        st.info("예시 데이터를 대신 표시합니다.")
        data = {
            "연도": list(range(2000, 2021)),
            "해수면(mm)": [i*0.3 + (i%3)*0.5 for i in range(21)]
        }
        df = pd.DataFrame(data)
        st.line_chart(df.set_index("연도"))

# -------------------------------
# 탭 2: 사용자 입력 데이터
# -------------------------------
with tab2:
    st.subheader("✍ 사용자 데이터 시각화")

    # 프롬프트 기반 예시 데이터 (대한민국 연안 해수면, 연 3.03mm 상승 가정)
    years = list(range(1989, 2023))
    base_level = 0
    values = [base_level + (i-1989)*3.03/10 for i in years]  # cm 단위 변환

    df_user = pd.DataFrame({"연도": years, "해수면(cm)": values})

    fig, ax = plt.subplots()
    ax.plot(df_user["연도"], df_user["해수면(cm)"], marker="o", linestyle="-", color="blue")
    ax.set_xlabel("연도")
    ax.set_ylabel("해수면(cm)")
    ax.set_title("대한민국 연안 해수면 상승 추이 (사용자 데이터 기반)")
    st.pyplot(fig)

    st.download_button("📥 사용자 데이터 CSV 다운로드", df_user.to_csv(index=False), "user_data.csv", "text/csv")

# -------------------------------
# 탭 3: 설명 + 정적 그래프
# -------------------------------
with tab3:
    st.subheader("🖼 설명 + 정적 그래프")
    st.write(
        """
        위 그래프는 1989년부터 2022년까지 대한민국 연안 21개 지점에서 관측한 연평균 해수면 높이 변화를 나타낸 것입니다.  
        그래프에 따르면 해수면은 연평균 약 **3.03mm**씩 상승하는 추세를 보이고 있으며,  
        특히 2010년 이후부터는 상승 폭이 더 뚜렷하게 나타나고 있습니다.  
        이는 기후 변화와 지구 온난화로 인한 해수면 상승 현상을 잘 보여줍니다.
        """
    )

    # 업로드된 PNG 불러오기
    try:
        image = Image.open("images/sealevel.png")
        st.image(image, caption="연평균 해수면 높이 (1989~2022, 21개소)", use_column_width=True)
        st.info("그래프는 마우스로 드래그하여 조정할 수 있습니다 (Streamlit 기본 뷰어).")
    except Exception as e:
        st.error(f"이미지 표시 실패: {e}")
