# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터 대시보드: 글로벌/국가 해수면 관련 공식 데이터(시도) 불러오기 → 시계열 시각화
  (코드 주석에 출처 URL 명시)
- 사용자 입력 대시보드: 프롬프트에 제공된 한국 연안 해수면(1989-2022) 설명/이미지 기반의 내부 데이터 사용(앱 실행 중 업로드 요구 없음)
- 규칙: 한국어 라벨, @st.cache_data 캐싱, 미래 날짜 제거, 전처리/다운로드 버튼 제공
- 폰트: /fonts/Pretendard-Bold.ttf 적용 시도 (없으면 무시)
"""

import io
import time
from datetime import datetime
from functools import wraps

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# -------------------------
# 설정: 한국어 폰트(가능하면) 적용
# -------------------------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
try:
    import matplotlib as mpl

    mpl.font_manager.fontManager.addfont(PRETENDARD_PATH)
    mpl.rcParams["font.family"] = mpl.font_manager.FontProperties(fname=PRETENDARD_PATH).get_name()
except Exception:
    # 폰트가 없으면 기본 폰트 사용
    pass

# Streamlit 측 스타일 (한국어 UI 폰트 적용 시도)
st.markdown(
    f"""
    <style>
    @font-face {{
      font-family: 'PretendardLocal';
      src: url('{PRETENDARD_PATH}');
    }}
    html, body, [class*="css"]  {{
      font-family: PretendardLocal, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕", "Noto Sans KR", sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# 유틸: 재시도 데코레이터
# -------------------------
def retry_request(retries=2, delay=1.0):
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(retries + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if i < retries:
                        time.sleep(delay)
                        continue
                    raise last_exc
        return wrapper
    return deco

# -------------------------
# 공개 데이터 불러오기 (시도)
# - 예시로 NOAA/University/NASA 계열의 공개 시계열을 시도하여 가져오되,
#   실패하면 예시 데이터로 대체하고 사용자에게 안내 표시
#
# 출처(예시 주석):
# - NOAA Sea Level Trends & Mean Sea Level: https://tidesandcurrents.noaa.gov/sltrends/
# - NASA Global Mean Sea Level (illustrative): https://sealevel.nasa.gov/
# - PSMSL (Permanent Service for Mean Sea Level): https://psmsl.org/
# -------------------------
@st.cache_data(ttl=3600)
def fetch_public_sea_level_data():
    """
    시도 단계:
    1) NOAA/PSMSL/NASA 등의 공개 CSV/API를 시도(여러 URL 중 하나)
    2) 실패 시 예시 데이터(합성 또는 내장) 반환
    """
    # 시도 URL 목록 (참고용 - 실제 파일 형식/엔드포인트는 변경될 수 있음)
    urls = [
        # NOAA 해수면 경향(지역별) - 웹페이지를 직접 파싱해야 하는 경우가 많아 우선 CSV 시도
        "https://www.psmsl.org/data/psmsl.csv",  # (예시: PSMSL 메타데이터, 실제 파일 경로는 다를 수 있음)
        # NASA/CSRIO 등 공개 CSV 예시 (여기선 단순 시도)
        "https://sealevel.nasa.gov/media/2018/08/sea_level_timeseries.csv",
    ]

    last_exc = None
    for url in urls:
        try:
            resp = _safe_get(url, timeout=10)
            # 직접 CSV 파싱 시도
            content = resp.content.decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(content))
            # 표준화: date, value, group(optional)
            if "date" not in df.columns:
                # 시계열이 연도/년 컬럼 등으로 있을 수 있음 - 단순 처리 시도
                # 여기서는 가능한 컬럼을 찾아서 변환
                possible_dt = [c for c in df.columns if "year" in c.lower() or "date" in c.lower()]
                if possible_dt:
                    df = df.rename(columns={possible_dt[0]: "date"})
                else:
                    # 못찾으면 실패로 간주하고 예외 발생
                    raise ValueError("CSV에 'date' 컬럼이 없음")
            df = df.rename(columns={df.columns[0]: "date"}) if "date" not in df.columns and len(df.columns) >= 2 else df
            # 간단 전처리
            df = df[["date"] + [c for c in df.columns if c != "date"]]
            return df
        except Exception as e:
            last_exc = e
            continue

    # 최종 실패 → 예시 데이터 생성 (글로벌 평균 해수면 합성 데이터)
    years = np.arange(1993, datetime.now().year + 1)  # 위성관측 일반 시작연도 ~ 현재
    # 합성: 연평균 상승 3.3 mm/yr(지구 평균 근접값 가정) + 랜덤 변동
    trend_mm_per_year = 3.3
    trend_cm = trend_mm_per_year / 10.0  # mm -> cm
    values = (years - years[0]) * trend_cm + np.random.normal(0, 0.15, size=len(years)).cumsum()
    df_example = pd.DataFrame({"date": pd.to_datetime(years.astype(str) + "-01-01"), "value_cm": values, "group": "Global (예시)"})
    return df_example

@retry_request(retries=2, delay=1.0)
def _safe_get(url, **kwargs):
    r = requests.get(url, **kwargs)
    r.raise_for_status()
    return r

# -------------------------
# 사용자 입력 데이터: 프롬프트에 제공된 "대한민국 연안 해수면(1989-2022), 연평균 +3.03 mm/yr"
# - 앱 내에서 파일 업로드 요구 금지. 대신, 프롬프트 내용만으로 합성 데이터 생성.
# - 이미지(제공 경로) 표시 및 분석 패널 제공.
# -------------------------
@st.cache_data
def make_korea_coastal_series(start_year=1989, end_year=2022, mm_per_year=3.03):
    years = np.arange(start_year, end_year + 1)
    # mm -> cm
    cm_per_year = mm_per_year / 10.0
    # 기본 선형 트렌드
    trend = (years - years[0]) * cm_per_year
    # 연도별 변동성: 계절/연간 변동, 임의 잡음(사용자 데이터 기반 합성)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 0.6, size=len(years)).cumsum() * 0.1  # 누적 노이즈 (변동성)
    values = trend + noise
    df = pd.DataFrame({"date": pd.to_datetime(years.astype(str) + "-01-01"), "value_cm": values})
    df["source"] = "국립해양조사원 (프롬프트 기반 합성 데이터)"
    return df

# -------------------------
# 전처리 공통 함수
# -------------------------
def preprocess_timeseries(df, date_col="date", value_col=None):
    # 복사
    df = df.copy()
    # 날짜 형변환
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # 제거: 미래 날짜(오늘 자정을 기준)
    today_midnight = pd.to_datetime(datetime.now().date())
    df = df[df[date_col] <= today_midnight]
    # value 컬럼 찾기
    if value_col is None:
        # pick first numeric column that's not date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            raise ValueError("수치형 값 컬럼을 찾을 수 없습니다.")
    df = df[[date_col, value_col] + [c for c in df.columns if c not in (date_col, value_col)]]
    df = df.rename(columns={date_col: "date", value_col: "value"})
    # 결측치 처리: 선형 보간 (시계열)
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    if df["value"].isnull().any():
        df["value"] = df["value"].interpolate().fillna(method="bfill").fillna(method="ffill")
    return df.reset_index(drop=True)

# -------------------------
# 메인 UI
# -------------------------
st.set_page_config(page_title="해수면 상승 & 청소년 주거 안정 대시보드", layout="wide")

st.title("해수면 상승 및 청소년 주거 불안 대시보드")
st.caption("공개 데이터 + 프롬프트 기반 대한민국 연안 데이터(1989-2022)를 함께 제공합니다. 모든 UI는 한국어입니다.")

# 상단 요약 카드
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("데이터: 공개(글로벌) + 사용자(한국 연안 합성)", "공개(시도) / 프롬프트 기반 합성")
    with col2:
        st.markdown(
            """
            **목표**: (1) 공식 공개 데이터 기반 분석, (2) 사용자가 제공한 프롬프트(문장 & 이미지) 기반 대한민국 연안 시계열 시각화 및 정책 제언 패널 제공
            """
        )
    with col3:
        st.markdown(f"데시보드 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# -------------------------
# 공개 데이터 섹션
# -------------------------
st.header("공식 공개 데이터(시도) 기반 해수면 시계열")
st.markdown(
    """
    **설명**: 여러 공식 출처(예: NOAA, NASA, PSMSL)를 시도하여 데이터를 불러옵니다.
    - 출처 예시:
      - NOAA Tides & Currents (지역별 경향): https://tidesandcurrents.noaa.gov/sltrends/
      - PSMSL (Permanent Service for Mean Sea Level): https://psmsl.org/
      - NASA Sea Level: https://sealevel.nasa.gov/
    (실제 연결 가능한 CSV/엔드포인트가 있으면 해당 데이터를 표시합니다. 연결 실패 시 예시 합성 데이터로 자동 대체됩니다.)
    """
)

# 시도해서 데이터 가져오기
try:
    public_raw = fetch_public_sea_level_data()
    public_df = preprocess_timeseries(public_raw, date_col="date", value_col=None)
    st.success("공개 데이터 로드 성공 (또는 예시 데이터로 대체됨).")
    st.write("데이터 미리보기:")
    st.dataframe(public_df.head(10))
except Exception as e:
    st.error("공개 데이터 로드 실패: 예시 데이터로 대체합니다.")
    public_df = make_korea_coastal_series(start_year=1993, end_year=datetime.now().year, mm_per_year=3.3)
    public_df = preprocess_timeseries(public_df, date_col="date", value_col="value_cm")

# 공개 데이터 시계열 시각화 (Plotly)
st.subheader("공개 데이터 시계열 (예시: 글로벌 평균 또는 불러온 데이터)")
with st.expander("시각화 옵션"):
    smoothing_days = st.slider("이동평균 (년)", min_value=1, max_value=10, value=3)
    show_points = st.checkbox("데이터 포인트 표시", value=False)

df_vis = public_df.copy()
df_vis["year"] = df_vis["date"].dt.year
# 연 단위로 집계(평균)
df_year = df_vis.groupby("year", as_index=False)["value"].mean()
df_year["date"] = pd.to_datetime(df_year["year"].astype(str) + "-01-01")
# smooth
df_year["smoothed"] = df_year["value"].rolling(window=smoothing_days, min_periods=1, center=True).mean()

fig_public = px.line(df_year, x="date", y="value", title="공개 데이터: 연도별 평균 해수면 (cm, 예시/실제)",
                     labels={"date": "연도", "value": "해수면 높이 (cm)"})
fig_public.add_scatter(x=df_year["date"], y=df_year["smoothed"], mode="lines", name=f"{smoothing_days}-년 이동평균")
if show_points:
    fig_public.add_scatter(x=df_year["date"], y=df_year["value"], mode="markers", name="데이터 포인트")
st.plotly_chart(fig_public, use_container_width=True)

# 공개 데이터 CSV 다운로드
csv_buf = io.StringIO()
df_year.to_csv(csv_buf, index=False)
st.download_button("공개 데이터(표준화된 연도별) CSV 다운로드", csv_buf.getvalue().encode("utf-8"), file_name="public_sea_level_yearly.csv", mime="text/csv")

st.markdown("---")

# -------------------------
# 사용자 입력(프롬프트) 섹션
# - 프롬프트에서 제공된 텍스트 설명, 수치(3.03 mm/yr), 이미지를 사용
# -------------------------
st.header("사용자 입력 데이터: 대한민국 연안 해수면 (프롬프트 기반)")
st.markdown(
    """
    제공된 입력(프롬프트) 내용:
    - 기간: 1989 ~ 2022 (34년)
    - 평균 상승률: 연평균 +3.03 mm/yr (→ 약 0.303 cm/yr, 총 약 10.3 cm 상승)
    - 보고서(서론/본론/결론) 텍스트가 포함되어 있으며, 이미지(그래프)도 제공됨.
    이 섹션은 업로드 요청 없이 프롬프트 내용만으로 합성/재현한 시계열과 분석을 보여줍니다.
    """
)

# 이미지 표시 (개발자가 제공한 경로)
try:
    img = Image.open("/mnt/data/9caf0f9d-2090-4f35-b1ef-89a96bb8be48.png")
    st.image(img, caption="입력: 제공된 해수면 상승 그래프 이미지", use_column_width=True)
except Exception:
    st.info("제공된 이미지 파일을 찾을 수 없습니다. (경로: /mnt/data/9caf0f9d-2090-4f35-b1ef-89a96bb8be48.png)")

# 프롬프트 기반 합성 시계열 생성
korea_df_raw = make_korea_coastal_series(start_year=1989, end_year=2022, mm_per_year=3.03)
korea_df = preprocess_timeseries(korea_df_raw, date_col="date", value_col="value_cm")

st.subheader("대한민국 연안(프롬프트 기반 합성) 시계열")
with st.expander("시각화 옵션 (한국 연안 데이터)"):
    years_range = st.slider("표시 연도 범위", min_value=1989, max_value=2022, value=(1989, 2022), step=1)
    smooth_k = st.slider("이동평균 (년) - 한국 데이터", min_value=1, max_value=10, value=3)
    show_trendline = st.checkbox("추세선 표시 (선형회귀)", value=True)

df_k_vis = korea_df[(korea_df["date"].dt.year >= years_range[0]) & (korea_df["date"].dt.year <= years_range[1])]
df_k_vis["year"] = df_k_vis["date"].dt.year
df_k_vis = df_k_vis.groupby("year", as_index=False)["value"].mean()
df_k_vis["date"] = pd.to_datetime(df_k_vis["year"].astype(str) + "-01-01")
df_k_vis["smoothed"] = df_k_vis["value"].rolling(window=smooth_k, min_periods=1, center=True).mean()

fig_k = px.line(df_k_vis, x="date", y="value", title="대한민국 연안 해수면 (1989-2022) - 합성",
                labels={"date": "연도", "value": "해수면 높이 (cm)"})
fig_k.add_scatter(x=df_k_vis["date"], y=df_k_vis["smoothed"], mode="lines", name=f"{smooth_k}-년 이동평균")
if show_trendline:
    # 선형 회귀(간단)
    coef = np.polyfit(df_k_vis["year"], df_k_vis["value"], 1)
    trend = np.poly1d(coef)
    fig_k.add_scatter(x=df_k_vis["date"], y=trend(df_k_vis["year"]), mode="lines", name="선형 추세선", line=dict(dash="dash"))
    slope_cm_per_year = coef[0]
    st.markdown(f"> 선형 회귀 추정: 약 **{slope_cm_per_year:.4f} cm/년** (≈ {slope_cm_per_year*10:.2f} mm/년)")

st.plotly_chart(fig_k, use_container_width=True)

# 한국 연안 데이터 다운로드
buf_k = io.StringIO()
df_k_vis.to_csv(buf_k, index=False)
st.download_button("대한민국(프롬프트 합성) 연도별 CSV 다운로드", buf_k.getvalue().encode("utf-8"), file_name="korea_coastal_1989_2022.csv", mime="text/csv")

st.markdown("---")

# -------------------------
# 위험도 요약 및 청소년 영향(정성 패널)
# -------------------------
st.header("청소년 주거 안정 관점의 요약 및 제언")
st.markdown(
    """
    **요약 (프롬프트 기반 자료와 공개 데이터 관찰 결과)**:
    - 프롬프트에 따르면 우리나라 연안 해수면은 1989-2022 사이 연평균 약 **3.03 mm/yr** 상승(약 10.3 cm 총 상승)을 보였습니다.
    - 공개 데이터(예시) 또한 전 지구적으로 해수면 상승 추세를 보이며, 최근 수십 년간 상승 속도가 가속화되는 경향을 보입니다.
    - 해수면 상승은 저지대·해안거주 가구의 침수 위험 증가, 농업 및 지역 경제 피해, 거주지 이동(이주) 가능성 등으로 이어지며,
      이는 청소년의 주거 안정성, 학업 지속성, 심리적 안정성에 악영향을 줄 수 있습니다.
    """
)

st.subheader("실천 가능한 학생(청소년) 행동 제안 (프롬프트에 따른 권고)")
st.markdown(
    """
    1. **알고 대비하기**: 학교 차원의 '침수 위험 지도' 제작 및 정기적 안전 교육·대피훈련 참여  
    2. **실천으로 행동하기**: 탄소 줄이기 캠페인(교내), 지역 기반 기후 행동 참여  
    3. **목소리 내기**: 학생회·청소년의회 등을 통한 정책 제안 및 지역정부 요청  
    """
)

# 작은 계산 도구: 특정 연도까지 예상 상승 (단순 선형 외삽)
st.subheader("간단한 외삽 계산기 (단순 선형)")
future_year = st.number_input("예측 종료 연도 입력", min_value=2023, max_value=2100, value=2050)
base_year = df_k_vis["year"].max()
base_value = float(df_k_vis.loc[df_k_vis["year"] == base_year, "value"].iloc[0])
# slope from earlier (use slope_cm_per_year if computed)
if 'slope_cm_per_year' not in locals():
    # fallback slope from mm_per_year
    slope_cm_per_year = 3.03 / 10.0
years_to_project = future_year - base_year
projected_increase = slope_cm_per_year * years_to_project
projected_value = base_value + projected_increase
st.markdown(f"- 기준 연도: **{base_year}년** (값: {base_value:.2f} cm)")
st.markdown(f"- **{future_year}년**까지 단순 선형 외삽 시 예상 해수면: **{projected_value:.2f} cm** (추가 상승 {projected_increase:.2f} cm)")

st.markdown("---")

# -------------------------
# 출처 및 참고자료 표시(앱 내부 주석/문서에 명시)
# -------------------------
st.header("참고 자료(예시 및 프롬프트에서 제공된 출처)")
st.markdown(
    """
    - 국립해양조사원, “우리나라 연안 해수면 매년 3.03mm 상승” (프롬프트 인용).  
      예시 URL: https://buly.kr/3j8qoCl  (프롬프트에 제공됨)
    - KBS 뉴스, “‘기후변화’ 해수면 상승속도 빨라져…2100년 우리나라 최대 82cm↑”, https://news.kbs.co.kr/news/pc/view/view.do?ncd=7622732
    - 뉴스칸, “해수면 상승, 생각보다 ‘심각한’ 이유”, https://www.newscan.co.kr/news/curationView.html?idxno=302741
    - 한국에너지경제신문, “[한반도가 물에 잠긴다] …”, https://edata.ekn.kr/article/view/ekn202502070008
    - NOAA Tides & Currents: https://tidesandcurrents.noaa.gov/sltrends/
    - PSMSL: https://psmsl.org/
    - NASA Sea Level: https://sealevel.nasa.gov/
    """
)

st.markdown("**주의**: 공개 데이터 소스의 실제 엔드포인트/CSV 위치는 기관에 따라 변동될 수 있습니다. 앱은 자동 재시도 후 실패 시 예시 데이터를 사용합니다.")

# -------------------------
# Kaggle 안내 (필요 시)
# -------------------------
with st.expander("Kaggle API 사용 안내 (선택)"):
    st.markdown(
        """
        - Kaggle 데이터셋을 사용하려면 Kaggle 계정 생성 → API 토큰 발급(kaggle.json) → Codespaces/로컬 환경에 업로드 후 `kaggle` CLI 사용이 필요합니다.
        - 예시 (터미널):
          1. `pip install kaggle`
          2. `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
          3. `kaggle datasets download -d <dataset-owner>/<dataset-name>`
        - Codespaces에서 비밀 관리를 통해 `kaggle.json`을 안전하게 설정하세요.
        """
    )

st.markdown("---")
st.caption("이 앱은 Streamlit + GitHub Codespaces 환경에서 즉시 실행 가능하도록 설계되었습니다. (프롬프트 기반 합성 데이터는 교육/시범 목적입니다.)")
