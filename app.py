import streamlit as st
import pandas as pd
import geopandas as gpd
import os

# 페이지 설정
st.set_page_config(page_title="응급의료 취약지 분석", page_icon="🚑", layout="wide")

st.title("🚑 응급의료 취약지 분석 프로젝트")
st.markdown("---")

@st.cache_data
def load_data():
    # 데이터 로드
    df = pd.read_csv('data/data.csv')
    # 행정구역코드 문자열 변환
    df['행정구역코드'] = df['행정구역코드'].astype(str).str.zfill(5)
    
    # 지도 데이터 로드
    gdf = gpd.read_file('data/sigungu.json')
    
    return df, gdf

try:
    with st.spinner('데이터를 불러오는 중입니다...'):
        df, gdf = load_data()
    
    st.success("✅ 데이터 로드 성공!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 통계 데이터 (CSV)")
        st.dataframe(df.head())
    with col2:
        st.subheader("🗺️ 지도 데이터 (JSON)")
        st.write(gdf.head())

except Exception as e:
    st.error(f"❌ 오류 발생: {e}")