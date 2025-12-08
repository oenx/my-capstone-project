import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="응급의료 취약지 분석 대시보드",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목 스타일링
st.markdown("""
    <h1 style='text-align: center;'>🚑 응급의료 취약지 분석 및 필수자원 예측</h1>
    <p style='text-align: center;'>데이터 기반의 응급의료 취약지역 탐지 및 자원 재배치 시뮬레이션</p>
    <hr>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 함수 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # CSV 로드
    df = pd.read_csv('data/data.csv')
    df['행정구역코드'] = df['행정구역코드'].astype(str).str.zfill(5)
    
    # GeoJSON 로드
    gdf = gpd.read_file('data/sigungu.json')
    
    # 행정구역코드 통일
    if 'SIG_CD' in gdf.columns:
        gdf = gdf.rename(columns={'SIG_CD': '행정구역코드'})
    
    gdf['행정구역코드'] = gdf['행정구역코드'].astype(str).str.zfill(5)
        
    return df, gdf

try:
    with st.spinner('데이터를 불러오는 중입니다...'):
        df, gdf = load_data()
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바 (필터링 옵션)
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 분석 옵션 설정")

# (1) 연도 선택
year_list = sorted(df['연도'].unique())
selected_year = st.sidebar.select_slider("분석할 연도를 선택하세요", options=year_list, value=2025)

# (2) 시도 선택
sido_list = sorted(df['시도명'].unique())
selected_sido = st.sidebar.multiselect("확인할 지역(시도)을 선택하세요", options=sido_list, default=sido_list)

# 데이터 필터링
df_year = df[df['연도'] == selected_year]

if selected_sido:
    df_filtered = df_year[df_year['시도명'].isin(selected_sido)]
    gdf_filtered = gdf[gdf['행정구역코드'].isin(df_filtered['행정구역코드'])]
else:
    df_filtered = df_year
    gdf_filtered = gdf

# 데이터 병합 (지도용)
merged_gdf = gdf_filtered.merge(df_filtered, on='행정구역코드', how='inner')

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 - KPI 지표 (Key Performance Indicators)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_pop = df_filtered['총인구'].sum()
vul_count = df_filtered['취약지역_여부'].sum()
avg_vul_index = df_filtered['취약지수'].mean()
needed_docs = df_filtered['추가_의사수'].sum()

with col1:
    st.metric("👥 총 인구 수", f"{total_pop:,.0f}명")
with col2:
    st.metric("🚨 취약지역 시군구 수", f"{vul_count}개", help="취약지수 상위 20% 지역")
with col3:
    st.metric("📉 평균 취약지수", f"{avg_vul_index:.3f}", help="1에 가까울수록 취약함")
with col4:
    st.metric("👨‍⚕️ 추가 필요 의사 수", f"{needed_docs:,.0f}명", delta_color="inverse")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 지도 시각화 & 차트 (2단 레이아웃)
# -----------------------------------------------------------------------------
row1_col1, row1_col2 = st.columns([3, 2])

with row1_col1:
    st.subheader(f"🗺️ {selected_year}년 응급의료 취약지수 지도")
    
    if not merged_gdf.empty:
        # 지도 중심 찾기
        center = [merged_gdf.geometry.centroid.y.mean(), merged_gdf.geometry.centroid.x.mean()]
        
        # Folium 지도 생성
        m = folium.Map(location=center, zoom_start=7 if len(selected_sido) > 1 else 9, tiles='cartodbpositron')

        # Choropleth (색칠 지도)
        folium.Choropleth(
            geo_data=merged_gdf,
            name='취약지수',
            data=merged_gdf,
            columns=['행정구역코드', '취약지수'],
            key_on='feature.properties.행정구역코드',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='취약지수 (높을수록 취약)'
        ).add_to(m)

        # 툴팁 추가
        folium.GeoJson(
            merged_gdf,
            name='지역 정보',
            style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
            tooltip=folium.GeoJsonTooltip(
                fields=['시도명', '시군구명', '취약지수', '추가_의사수', '추가_구급차수'],
                aliases=['시도', '시군구', '취약지수', '필요 의사', '필요 구급차'],
                localize=True
            )
        ).add_to(m)

        st_folium(m, width=None, height=500)
    else:
        st.warning("선택된 지역의 지도 데이터가 없습니다.")

with row1_col2:
    st.subheader("📊 자원 부족 상위 지역 (Top 10)")
    
    tab1, tab2 = st.tabs(["필요 의사 수", "취약지수 순위"])
    
    with tab1:
        top_docs = df_filtered.nlargest(10, '추가_의사수')
        if not top_docs.empty:
            fig_doc = px.bar(
                top_docs, 
                x='추가_의사수', 
                y='시군구명', 
                orientation='h',
                color='추가_의사수',
                color_continuous_scale='Reds',
                title=f"의사 부족이 심각한 지역 Top 10"
            )
            fig_doc.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_doc, use_container_width=True)
        else:
            st.info("추가로 필요한 의사가 없습니다.")

    with tab2:
        top_vul = df_filtered.nlargest(10, '취약지수')
        fig_vul = px.bar(
            top_vul,
            x='취약지수',
            y='시군구명',
            orientation='h',
            color='취약지수',
            title="취약지수가 높은 지역 Top 10"
        )
        fig_vul.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_vul, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. 상세 데이터 테이블
# -----------------------------------------------------------------------------
st.markdown("### 📋 상세 데이터 보기")
with st.expander("클릭하여 전체 데이터 확인하기"):
    # 스타일링을 try-except로 감싸기
    try:
        styled_df = (
            df_filtered[['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수']]
            .sort_values(by='취약지수', ascending=False)
            .style.background_gradient(cmap='OrRd', subset=['취약지수'])
            .format({'취약지수': '{:.3f}', '총인구': '{:,.0f}'})
        )
        st.dataframe(styled_df)
    except ImportError:
        # matplotlib가 없을 경우 스타일링 없이 표시
        st.dataframe(
            df_filtered[['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수']]
            .sort_values(by='취약지수', ascending=False)
        )
        st.info("💡 표 스타일링을 위해 matplotlib 설치가 필요합니다.")