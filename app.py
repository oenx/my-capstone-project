import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pulp

# -----------------------------------------------------------------------------
# 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="응급의료 취약지 분석 대시보드",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 데이터 로드 함수
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/data.csv')
    df['행정구역코드'] = df['행정구역코드'].astype(str).str.zfill(5)
    
    gdf = gpd.read_file('data/sigungu.json')
    
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
# 취약지수 개선 효과 계산 함수
# -----------------------------------------------------------------------------
def calculate_improvement_per_unit(row, resource_type):
    """
    자원 1단위당 취약지수 개선 효과 (w_i) 계산
    
    가정: 취약지수에 대한 자원의 기여도 × (1 / 현재 부족량)
    부족량이 적을수록 1단위의 효과가 크다고 가정
    """
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    
    col_name = col_map[resource_type]
    shortage = row[col_name]
    
    if shortage <= 0:
        return 0
    
    # 취약지수가 높고, 부족량이 적당한 곳이 효과적
    # 가중치: 취약지수 × 인구 가중치 × (1/부족량의 제곱근)
    vulnerability_weight = row['취약지수']
    population_weight = np.log1p(row['총인구']) / 10  # 인구 고려
    efficiency = 1.0 / np.sqrt(shortage)  # 부족량이 적을수록 효율적
    
    improvement = vulnerability_weight * population_weight * efficiency
    
    return improvement

# -----------------------------------------------------------------------------
# 정수계획법(ILP) 최적화 함수
# -----------------------------------------------------------------------------
def optimize_allocation_ilp(df_scope, resource_type, total_resources):
    """
    PuLP를 사용한 정수계획법(ILP) 최적화
    
    목적함수: Maximize Σ(w_i × x_i)
    제약조건:
      - Σx_i = total_resources
      - 0 ≤ x_i ≤ shortage_i
    """
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    
    col_name = col_map[resource_type]
    
    # 데이터 준비
    df_opt = df_scope.copy().reset_index(drop=True)
    df_opt['부족량'] = df_opt[col_name]
    
    # 부족량이 있는 지역만 대상
    df_opt = df_opt[df_opt['부족량'] > 0].copy()
    
    if df_opt.empty:
        st.warning("배분 가능한 지역이 없습니다.")
        return df_scope
    
    # 개선효과(w_i) 계산
    df_opt['개선효과'] = df_opt.apply(
        lambda row: calculate_improvement_per_unit(row, resource_type),
        axis=1
    )
    
    # ---------------------------------------
    # 정수계획법 모델 생성
    # ---------------------------------------
    model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)
    
    # ---------------------------------------
    # 변수 정의: x[i] = 지역 i에 배분할 자원 수 (정수)
    # ---------------------------------------
    x = pulp.LpVariable.dicts(
        "x",
        df_opt.index,
        lowBound=0,
        cat="Integer"
    )
    
    # ---------------------------------------
    # 목적함수: 총 취약지수 개선 효과 최대화
    # ---------------------------------------
    model += pulp.lpSum(
        df_opt.loc[i, '개선효과'] * x[i] for i in df_opt.index
    ), "Total_Improvement"
    
    # ---------------------------------------
    # 제약조건
    # ---------------------------------------
    # 1. 총 배분량 = 사용 가능한 자원
    model += (
        pulp.lpSum(x[i] for i in df_opt.index) == total_resources,
        "Total_Resources"
    )
    
    # 2. 각 지역 배분량 ≤ 해당 지역 부족량
    for i in df_opt.index:
        model += (
            x[i] <= df_opt.loc[i, '부족량'],
            f"Max_Shortage_{i}"
        )
    
    # ---------------------------------------
    # 최적화 실행
    # ---------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0)  # 로그 숨김
    model.solve(solver)
    
    # ---------------------------------------
    # 결과 처리
    # ---------------------------------------
    # 배분 결과를 df_opt에 저장
    df_opt['배분량'] = df_opt.index.map(lambda i: x[i].value() if x[i].value() else 0)
    
    # 원본 데이터프레임에 병합
    df_result = df_scope.copy()
    df_result['배분량'] = 0
    
    for i in df_opt.index:
        원본_인덱스 = df_opt.loc[i, '행정구역코드']
        배분량 = df_opt.loc[i, '배분량']
        df_result.loc[df_result['행정구역코드'] == 원본_인덱스, '배분량'] = 배분량
    
    # 배분 후 지표 계산
    df_result['배분_후_부족'] = df_result[col_name] - df_result['배분량']
    df_result['해소율'] = (df_result['배분량'] / df_result[col_name] * 100).fillna(0)
    
    # 취약지수 개선 계산 (간단한 선형 모델)
    df_result['취약지수_개선'] = df_result.apply(
        lambda row: row['취약지수'] * 0.3 * (row['배분량'] / max(row[col_name], 1)) 
        if row[col_name] > 0 else 0,
        axis=1
    )
    df_result['배분_후_취약지수'] = df_result['취약지수'] - df_result['취약지수_개선']
    
    return df_result

# -----------------------------------------------------------------------------
# 사이드바
# -----------------------------------------------------------------------------
st.sidebar.title("🚑 메뉴")
page = st.sidebar.radio("페이지 선택", ["📊 현황 분석", "🎯 시나리오 시뮬레이션"])

st.sidebar.markdown("---")
st.sidebar.header("🔍 분석 옵션")

year_list = sorted(df['연도'].unique())
selected_year = st.sidebar.select_slider("분석 연도", options=year_list, value=2025)

# -----------------------------------------------------------------------------
# 페이지 1: 현황 분석
# -----------------------------------------------------------------------------
if page == "📊 현황 분석":
    st.markdown("""
        <h1 style='text-align: center;'>🚑 응급의료 취약지 분석 대시보드</h1>
        <p style='text-align: center;'>데이터 기반의 응급의료 취약지역 탐지 및 현황 분석</p>
        <hr>
    """, unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year]
    df_filtered = df_year
    gdf_filtered = gdf
    
    merged_gdf = gdf_filtered.merge(df_filtered, on='행정구역코드', how='inner')
    
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    
    total_pop = df_filtered['총인구'].sum()
    vul_count = df_filtered['취약지역_여부'].sum()
    avg_vul_index = df_filtered['취약지수'].mean()
    needed_docs = df_filtered['추가_의사수'].sum()
    
    with col1:
        st.metric("👥 총 인구 수", f"{total_pop:,.0f}명")
    with col2:
        st.metric("🚨 취약지역 수", f"{vul_count}개")
    with col3:
        st.metric("📉 평균 취약지수", f"{avg_vul_index:.3f}")
    with col4:
        st.metric("👨‍⚕️ 필요 의사", f"{needed_docs:,.0f}명")
    
    st.markdown("---")
    
    # 지도 & 차트
    row1_col1, row1_col2 = st.columns([3, 2])
    
    with row1_col1:
        st.subheader(f"🗺️ {selected_year}년 응급의료 취약지수 지도")
        
        if not merged_gdf.empty:
            center = [merged_gdf.geometry.centroid.y.mean(), merged_gdf.geometry.centroid.x.mean()]
            m = folium.Map(location=center, zoom_start=7, tiles='cartodbpositron')
            
            folium.Choropleth(
                geo_data=merged_gdf,
                name='취약지수',
                data=merged_gdf,
                columns=['행정구역코드', '취약지수'],
                key_on='feature.properties.행정구역코드',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='취약지수'
            ).add_to(m)
            
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
                    color_continuous_scale='Reds'
                )
                fig_doc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_doc, use_container_width=True)
        
        with tab2:
            top_vul = df_filtered.nlargest(10, '취약지수')
            fig_vul = px.bar(
                top_vul,
                x='취약지수',
                y='시군구명',
                orientation='h',
                color='취약지수'
            )
            fig_vul.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_vul, use_container_width=True)
    
    # 상세 데이터
    st.markdown("### 📋 상세 데이터")
    with st.expander("클릭하여 전체 데이터 확인"):
        try:
            styled_df = (
                df_filtered[['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수']]
                .sort_values(by='취약지수', ascending=False)
                .style.background_gradient(cmap='OrRd', subset=['취약지수'])
                .format({'취약지수': '{:.3f}', '총인구': '{:,.0f}'})
            )
            st.dataframe(styled_df)
        except:
            st.dataframe(
                df_filtered[['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수']]
                .sort_values(by='취약지수', ascending=False)
            )

# -----------------------------------------------------------------------------
# 페이지 2: 시나리오 시뮬레이션
# -----------------------------------------------------------------------------
elif page == "🎯 시나리오 시뮬레이션":
    st.markdown("""
        <h1 style='text-align: center;'>🎯 응급자원 최적 배분 시뮬레이션</h1>
        <p style='text-align: center;'>정수계획법(ILP)을 활용한 수학적 최적해 도출</p>
        <hr>
    """, unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year]
    
    # 설명 박스
    st.info("""
    💡 **정수계획법(Integer Linear Programming)**
    - 목적함수: 전체 취약지수 개선 효과 최대화
    - 제약조건: 총 자원 = 배분 가능 수량, 각 지역 배분 ≤ 부족량
    - PuLP 라이브러리를 사용하여 수학적으로 정확한 최적해 계산
    """)
    
    # 시나리오 설정
    st.subheader("⚙️ 시나리오 설정")
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        scope = st.selectbox("📍 배분 범위", ["전국", "특정 시도"])
    
    with col2:
        if scope == "특정 시도":
            sido_list = sorted(df_year['시도명'].unique())
            selected_sido = st.selectbox("시도 선택", sido_list)
            df_scope = df_year[df_year['시도명'] == selected_sido]
        else:
            selected_sido = None
            df_scope = df_year
    
    with col3:
        resource_type = st.selectbox("🚑 자원 유형", ["구급차", "의사", "응급시설"])
    
    # 자원 수량
    col1, col2 = st.columns([3, 1])
    
    with col1:
        resource_map = {
            "구급차": ("추가_구급차수", "대", 100),
            "의사": ("추가_의사수", "명", 500),
            "응급시설": ("추가_응급시설수", "개소", 50)
        }
        
        col_name, unit, max_val = resource_map[resource_type]
        
        resource_amount = st.slider(
            f"추가 가능한 {resource_type} 수량",
            min_value=1,
            max_value=max_val,
            value=min(30, max_val)
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_simulation = st.button("🚀 최적화 실행", type="primary", use_container_width=True)
    
    # 시뮬레이션 실행
    if run_simulation:
        with st.spinner('정수계획법(ILP)으로 최적해 계산 중...'):
            df_result = optimize_allocation_ilp(df_scope, resource_type, resource_amount)
        
        st.success("✅ 최적 배분 완료!")
        
        st.markdown("---")
        st.subheader("📊 최적화 결과")
        
        # 배분받은 지역만 필터링
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        
        # 전체 개선 효과 계산
        total_improvement = df_result['취약지수_개선'].sum()
        avg_before = df_result['취약지수'].mean()
        avg_after = df_result['배분_후_취약지수'].mean()
        total_allocated = df_allocated['배분량'].sum()
        
        # KPI
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 배분 지역", f"{len(df_allocated)}개")
        with col2:
            st.metric("✅ 배분 완료", f"{int(total_allocated)}{unit}")
        with col3:
            improvement_rate = (avg_before - avg_after) / avg_before * 100 if avg_before > 0 else 0
            st.metric("📈 평균 개선율", f"{improvement_rate:.1f}%")
        with col4:
            st.metric("✨ 총 개선 효과", f"{total_improvement:.4f}")
        
        # 지도 + 표
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 🗺️ 최적 배분 결과 지도")
            
            gdf_result = gdf.merge(
                df_allocated[['행정구역코드', '배분량', '취약지수_개선']], 
                on='행정구역코드', 
                how='inner'
            )
            
            if not gdf_result.empty:
                center = [gdf_result.geometry.centroid.y.mean(), gdf_result.geometry.centroid.x.mean()]
                m = folium.Map(location=center, zoom_start=8, tiles='cartodbpositron')
                
                folium.Choropleth(
                    geo_data=gdf_result,
                    name='배분량',
                    data=gdf_result,
                    columns=['행정구역코드', '배분량'],
                    key_on='feature.properties.행정구역코드',
                    fill_color='Greens',
                    fill_opacity=0.7,
                    line_opacity=0.5,
                    legend_name=f'배분된 {resource_type} 수'
                ).add_to(m)
                
                # 툴팁 추가
                merged_for_tooltip = gdf_result.merge(
                    df_allocated[['행정구역코드', '시도명', '시군구명']], 
                    on='행정구역코드', 
                    how='left'
                )
                
                folium.GeoJson(
                    merged_for_tooltip,
                    name='배분 정보',
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['시군구명', '배분량', '취약지수_개선'],
                        aliases=['지역', f'{resource_type} 배분', '개선 효과'],
                        localize=True
                    )
                ).add_to(m)
                
                st_folium(m, width=None, height=400)
            else:
                st.warning("배분된 지역이 없습니다.")
        
        with col2:
            st.markdown("#### 📋 배분 상세 (Top 15)")
            
            if not df_allocated.empty:
                display_df = df_allocated.nlargest(15, '배분량')[
                    ['시군구명', '배분량', '취약지수_개선', '해소율']
                ]
                
                st.dataframe(
                    display_df.style.format({
                        '배분량': '{:.0f}',
                        '취약지수_개선': '{:.4f}',
                        '해소율': '{:.1f}%'
                    }),
                    height=400
                )
            else:
                st.info("배분된 지역이 없습니다.")
        
        # 취약지수 개선 효과 차트
        if not df_allocated.empty:
            st.markdown("#### 📊 취약지수 개선 효과 (Top 10)")
            
            top10 = df_allocated.nlargest(10, '배분량')
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=top10['시군구명'],
                x=top10['취약지수'],
                name='배분 전',
                orientation='h',
                marker_color='lightcoral'
            ))
            
            fig.add_trace(go.Bar(
                y=top10['시군구명'],
                x=top10['배분_후_취약지수'],
                name='배분 후',
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                barmode='group',
                yaxis={'categoryorder':'total ascending'},
                height=400,
                xaxis_title='취약지수'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 전체 결과 테이블
        with st.expander("📋 전체 지역 배분 결과 보기"):
            display_full = df_result[df_result['배분량'] > 0] if len(df_result[df_result['배분량'] > 0]) > 0 else df_result.head(20)
            
            st.dataframe(
                display_full[['시도명', '시군구명', '취약지수', '배분량', '배분_후_취약지수', '취약지수_개선', '해소율']]
                .sort_values('배분량', ascending=False)
                .style.format({
                    '취약지수': '{:.4f}',
                    '배분량': '{:.0f}',
                    '배분_후_취약지수': '{:.4f}',
                    '취약지수_개선': '{:.4f}',
                    '해소율': '{:.1f}%'
                })
            )
        
        # 다운로드
        csv = df_result.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 최적화 결과 다운로드 (CSV)",
            data=csv,
            file_name=f"ILP_최적배분_{resource_type}_{selected_year}년.csv",
            mime="text/csv"
        )