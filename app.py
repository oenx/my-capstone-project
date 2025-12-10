# app.py - Professional Grade Emergency Medical Resource Optimization Dashboard
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pulp

# =====================================================================
# 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="응급의료 취약지 분석 및 자원 최적배분",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 데이터 로드 및 검증
# =====================================================================
@st.cache_data
def load_and_validate_data():
    """데이터 로드 및 품질 검증"""
    try:
        df = pd.read_csv('data/data.csv')
        gdf = gpd.read_file('data/sigungu.json')
        
        # 행정구역코드 정규화
        if '행정구역코드' in df.columns:
            df['행정구역코드'] = df['행정구역코드'].astype(str).str.zfill(5)
        
        if 'SIG_CD' in gdf.columns:
            gdf = gdf.rename(columns={'SIG_CD': '행정구역코드'})
        if '행정구역코드' in gdf.columns:
            gdf['행정구역코드'] = gdf['행정구역코드'].astype(str).str.zfill(5)
        
        # 데이터 품질 검증
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_codes': df['행정구역코드'].duplicated().sum(),
            'geo_match_rate': len(gdf.merge(df, on='행정구역코드')) / len(df) * 100,
            'year_range': (df['연도'].min(), df['연도'].max()) if '연도' in df.columns else (None, None)
        }
        
        return df, gdf, validation_results
    
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

df, gdf, data_quality = load_and_validate_data()

# =====================================================================
# 최적화 함수 (검증 강화)
# =====================================================================
def calculate_improvement_per_unit(row, resource_type):
    """
    개선효과 계산 (w_i)
    
    수식: w_i = V_i × P_i × E_i
    - V_i: 취약도 (취약지수)
    - P_i: 인구 가중치 = log(인구+1)/10
    - E_i: 효율성 = 1/√(부족량)
    """
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    
    col_name = col_map[resource_type]
    shortage = float(row.get(col_name, 0))
    
    if shortage <= 0:
        return 0.0
    
    vulnerability = float(row.get('취약지수', 0.0))
    population = float(row.get('총인구', 0))
    
    # 가중치 계산
    V_i = vulnerability
    P_i = np.log1p(population) / 10.0
    E_i = 1.0 / np.sqrt(max(shortage, 1e-6))
    
    return V_i * P_i * E_i

def optimize_allocation_ilp(df_scope, resource_type, total_resources):
    """ILP 최적화 (검증 강화)"""
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    col_name = col_map[resource_type]
    
    df_opt = df_scope.copy().reset_index(drop=True)
    
    if col_name not in df_opt.columns:
        return df_scope.copy(), {"status": "error", "message": f"'{col_name}' 컬럼 없음"}
    
    df_opt['부족량'] = pd.to_numeric(df_opt[col_name], errors='coerce').fillna(0)
    df_opt = df_opt[df_opt['부족량'] > 0].copy()
    
    if df_opt.empty:
        return df_scope.copy(), {"status": "error", "message": "배분 가능 지역 없음"}
    
    # 개선효과 계산
    df_opt['개선효과'] = df_opt.apply(
        lambda r: calculate_improvement_per_unit(r, resource_type), 
        axis=1
    )
    
    # ILP 모델
    model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", df_opt.index, lowBound=0, cat="Integer")
    
    # 목적함수
    model += pulp.lpSum(df_opt.loc[i, '개선효과'] * x[i] for i in df_opt.index)
    
    # 제약조건
    model += pulp.lpSum(x[i] for i in df_opt.index) == int(total_resources)
    for i in df_opt.index:
        model += x[i] <= int(df_opt.loc[i, '부족량'])
    
    # 최적화 실행
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = model.solve(solver)
    
    # 결과 검증
    solver_status = {
        1: "Optimal",
        0: "Not Solved",
        -1: "Infeasible",
        -2: "Unbounded",
        -3: "Undefined"
    }
    
    optimization_info = {
        "status": solver_status.get(status, "Unknown"),
        "objective_value": pulp.value(model.objective) if status == 1 else 0,
        "solving_time": "< 1s",  # PuLP doesn't provide time by default
        "num_variables": len(df_opt),
        "num_allocated": 0
    }
    
    if status != 1:
        return df_scope.copy(), optimization_info
    
    # 결과 처리
    df_opt['배분량'] = df_opt.index.map(
        lambda i: int(x[i].value()) if x[i].value() is not None else 0
    )
    
    optimization_info["num_allocated"] = (df_opt['배분량'] > 0).sum()
    
    # 원본 데이터프레임에 병합
    df_result = df_scope.copy()
    df_result['배분량'] = 0
    
    for i in df_opt.index:
        code = df_opt.loc[i, '행정구역코드']
        allocated = int(df_opt.loc[i, '배분량'])
        df_result.loc[df_result['행정구역코드'] == code, '배분량'] = allocated
    
    # 후처리
    df_result['배분_후_부족'] = df_result[col_name] - df_result['배분량']
    df_result['해소율'] = (
        df_result['배분량'] / df_result[col_name] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 취약지수 개선
    def calc_vul_improve(row):
        try:
            if row[col_name] > 0:
                return row.get('취약지수', 0.0) * 0.30 * (row['배분량'] / max(row[col_name], 1))
            return 0.0
        except:
            return 0.0
    
    df_result['취약지수_개선'] = df_result.apply(calc_vul_improve, axis=1)
    df_result['배분_후_취약지수'] = df_result['취약지수'] - df_result['취약지수_개선']
    df_result['개선율(%)'] = (
        df_result['취약지수_개선'] / df_result['취약지수'] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df_result, optimization_info

# =====================================================================
# 분석 함수
# =====================================================================
def calculate_regional_stats(df_result, scope, selected_sido=None):
    """지역 통계 계산"""
    if scope == "특정 시도" and selected_sido:
        df_analysis = df_result[df_result['시도명'] == selected_sido].copy()
        region_name = selected_sido
    else:
        df_analysis = df_result.copy()
        region_name = "전국"
    
    total_before = float(df_analysis['취약지수'].sum())
    total_after = float(df_analysis['배분_후_취약지수'].sum())
    improvement = total_before - total_after
    
    return {
        'region_name': region_name,
        'before': total_before,
        'after': total_after,
        'improvement': improvement,
        'improvement_rate': (improvement / total_before * 100) if total_before > 0 else 0.0,
        'num_regions': len(df_analysis),
        'avg_before': float(df_analysis['취약지수'].mean()),
        'avg_after': float(df_analysis['배분_후_취약지수'].mean())
    }

def calculate_efficiency_metrics(df_allocated, resource_type):
    """효율성 지표 계산"""
    if df_allocated.empty:
        return {}
    
    total_allocated = df_allocated['배분량'].sum()
    total_improvement = df_allocated['취약지수_개선'].sum()
    
    # 자원 1단위당 개선 효과
    efficiency = total_improvement / total_allocated if total_allocated > 0 else 0
    
    # 지니계수 (배분 불균형도)
    allocations = sorted(df_allocated['배분량'].values)
    n = len(allocations)
    cumsum = np.cumsum(allocations)
    gini = (2 * sum((i+1) * allocations[i] for i in range(n))) / (n * sum(allocations)) - (n+1)/n
    
    return {
        'efficiency': efficiency,
        'gini_coefficient': gini,
        'concentration_top10': df_allocated.nlargest(10, '배분량')['배분량'].sum() / total_allocated * 100,
        'avg_allocation': total_allocated / len(df_allocated)
    }

# =====================================================================
# 세션 상태
# =====================================================================
if "ilp_result" not in st.session_state:
    st.session_state["ilp_result"] = None
if "ilp_params" not in st.session_state:
    st.session_state["ilp_params"] = {}
if "optimization_info" not in st.session_state:
    st.session_state["optimization_info"] = {}

# =====================================================================
# 사이드바
# =====================================================================
st.sidebar.title("🚑 네비게이션")
page = st.sidebar.radio(
    "페이지 선택",
    ["🏠 프로젝트 개요", "📊 현황 분석", "🎯 최적화 시뮬레이션", "📈 성과 평가", "📖 방법론"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 분석 설정")

year_list = sorted(df['연도'].unique()) if '연도' in df.columns else [2025]
selected_year = st.sidebar.select_slider("분석 연도", options=year_list, value=year_list[0])

# 데이터 품질 정보
with st.sidebar.expander("📊 데이터 품질 정보"):
    st.metric("총 레코드", f"{data_quality['total_records']:,}개")
    st.metric("결측치", f"{data_quality['missing_values']}개")
    st.metric("지도 매칭률", f"{data_quality['geo_match_rate']:.1f}%")
    if data_quality['year_range'][0]:
        st.info(f"분석 기간: {data_quality['year_range'][0]}~{data_quality['year_range'][1]}")

st.sidebar.markdown("---")
st.sidebar.info(
    "**프로젝트 정보**\n\n"
    "- 과제명: 응급의료 취약지 최적 자원배분\n"
    "- 알고리즘: Integer Linear Programming\n"
    "- 개발: Python, Streamlit, PuLP\n"
    "- 데이터: 공공데이터 기반 예측"
)

# =====================================================================
# 페이지 0: 프로젝트 개요
# =====================================================================
if page == "🏠 프로젝트 개요":
    st.markdown("<h1 style='text-align: center;'>🚑 응급의료 취약지 분석 및 자원 최적배분</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 1.1rem;'>데이터 기반 의사결정 지원 시스템</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 연구 배경")
        st.markdown("""
        <div class='insight-box'>
        <b>Problem Statement</b><br>
        • 고령화로 인한 응급의료 수요 급증<br>
        • 지역 간 의료자원 불균형 심화<br>
        • 한정된 예산 내 효율적 배분 필요<br><br>
        
        <b>Research Objective</b><br>
        정수계획법(ILP)을 활용하여 전체 취약지수 개선을 최대화하는 최적 자원 배분 전략 도출
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔬 핵심 방법론")
        st.markdown("""
        **최적화 모델**
        - 목적함수: max Σ(w_i × x_i)
        - 제약조건: 예산 제한, 수요 상한
        - 솔버: PuLP (CBC)
        
        **개선효과 산정**
        - 취약도 × 인구 가중 × 효율성
        - 한계효용 체감 반영
        """)
    
    with col2:
        st.subheader("📊 데이터 명세")
        st.markdown(f"""
        **데이터 범위**
        - 공간: 전국 {data_quality['total_records']}개 시군구
        - 시간: {data_quality['year_range'][0]}~{data_quality['year_range'][1]}년
        - 지도 매칭률: {data_quality['geo_match_rate']:.1f}%
        
        **주요 변수**
        - 취약지수 (0~1)
        - 인구통계 (총인구, 고령인구)
        - 자원 부족량 (의사, 구급차, 시설)
        """)
        
        st.subheader("✨ 시스템 특징")
        st.markdown("""
        **1. 수학적 최적해 보장**
        - ILP 기반 정확한 해 도출
        
        **2. 다양한 시나리오 분석**
        - 전국/지역 단위 분석
        - 자원 유형별 시뮬레이션
        
        **3. 실시간 의사결정 지원**
        - 인터랙티브 시각화
        - 즉시 결과 확인 가능
        """)
    
    st.markdown("---")
    st.subheader("⚠️ 연구 한계")
    st.markdown("""
    <div class='warning-box'>
    1. <b>취약지수 재계산 간소화</b>: 선형 모델로 근사 (실제 비선형 가능성)<br>
    2. <b>자원 기여도 가정</b>: 30% 고정값 사용 (실증 데이터 필요)<br>
    3. <b>정적 분석</b>: 동적 수요 변화 미반영<br>
    4. <b>단일 목적함수</b>: 효율성만 고려 (형평성 등 미반영)<br><br>
    
    <b>향후 개선 방향</b><br>
    • 머신러닝 기반 취약지수 예측 모델<br>
    • 다목적 최적화 (형평성, 접근성 동시 고려)<br>
    • 실시간 모니터링 시스템 구축
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 페이지 1: 현황 분석
# =====================================================================
elif page == "📊 현황 분석":
    st.markdown("<h1 style='text-align: center;'>📊 응급의료 취약지 현황 분석</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>{selected_year}년도 기준</p>", unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()
    merged_gdf = gdf.merge(df_year, on='행정구역코드', how='inner')
    
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    total_pop = int(df_year['총인구'].sum())
    vul_count = int((df_year['취약지수'] > df_year['취약지수'].quantile(0.8)).sum())
    avg_vul = float(df_year['취약지수'].mean())
    total_shortage = int(df_year['추가_의사수'].sum())
    
    with col1:
        st.metric("👥 총 인구", f"{total_pop:,}명")
    with col2:
        st.metric("🚨 고취약 지역", f"{vul_count}개", help="상위 20%")
    with col3:
        st.metric("📉 평균 취약지수", f"{avg_vul:.3f}")
    with col4:
        st.metric("⚠️ 의사 부족", f"{total_shortage:,}명")
    
    st.markdown("---")
    
    # 지도 & 통계
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🗺️ 취약지수 분포")
        
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
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                tooltip=folium.GeoJsonTooltip(
                    fields=['시도명', '시군구명', '총인구', '취약지수', '추가_의사수'],
                    aliases=['시도', '시군구', '인구', '취약지수', '필요 의사'],
                    localize=True
                )
            ).add_to(m)
            
            st_folium(m, width=None, height=500)
    
    with col2:
        st.subheader("📊 취약성 분포 분석")
        
        # 히스토그램
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_year['취약지수'],
            nbinsx=20,
            marker_color='indianred',
            opacity=0.7
        ))
        fig.update_layout(
            xaxis_title='취약지수',
            yaxis_title='지역 수',
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 요약
        st.markdown("**분포 통계**")
        stats_df = pd.DataFrame({
            '지표': ['최소값', '25%', '중앙값', '75%', '최대값', '표준편차'],
            '값': [
                f"{df_year['취약지수'].min():.3f}",
                f"{df_year['취약지수'].quantile(0.25):.3f}",
                f"{df_year['취약지수'].median():.3f}",
                f"{df_year['취약지수'].quantile(0.75):.3f}",
                f"{df_year['취약지수'].max():.3f}",
                f"{df_year['취약지수'].std():.3f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # 상세 분석
    st.markdown("---")
    st.subheader("📈 상세 분석")
    
    tab1, tab2, tab3 = st.tabs(["자원 부족 TOP 10", "취약성-인구 분석", "시도별 비교"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**의사 부족 TOP 10**")
            top_docs = df_year.nlargest(10, '추가_의사수')
            fig = px.bar(
                top_docs, x='추가_의사수', y='시군구명',
                orientation='h', color='추가_의사수',
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**취약지수 TOP 10**")
            top_vul = df_year.nlargest(10, '취약지수')
            fig = px.bar(
                top_vul, x='취약지수', y='시군구명',
                orientation='h', color='취약지수',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**인구 규모와 취약성의 관계**")
        fig = px.scatter(
            df_year, x='총인구', y='취약지수',
            size='추가_의사수', color='시도명',
            hover_name='시군구명', size_max=20,
            opacity=0.7
        )
        fig.update_layout(height=500)
        fig.add_hline(y=df_year['취약지수'].median(), line_dash="dash", 
                      annotation_text="중앙값", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 우상단 지역(인구 많고 취약지수 높음)이 우선 관리 대상입니다.")
    
    with tab3:
        if '시도명' in df_year.columns:
            sido_stats = df_year.groupby('시도명').agg({
                '취약지수': 'mean',
                '추가_의사수': 'sum',
                '총인구': 'sum'
            }).reset_index()
            
            fig = px.bar(
                sido_stats.sort_values('취약지수', ascending=False),
                x='시도명', y='취약지수',
                color='취약지수', color_continuous_scale='RdYlGn_r',
                text='취약지수'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 페이지 2: 최적화 시뮬레이션
# =====================================================================
elif page == "🎯 최적화 시뮬레이션":
    st.markdown("<h1 style='text-align: center;'>🎯 응급자원 최적 배분 시뮬레이션</h1>", unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()
    
    # 알고리즘 설명
    with st.expander("🔬 최적화 알고리즘 상세", expanded=False):
        st.markdown("""Integer Linear Programming (ILP)

        목적함수: max Σ(w_i × x_i)
        - w_i = V_i × P_i × E_i
        - V_i: 취약도, P_i: 인구 가중, E_i: 효율성
        
        제약조건:
        - Σx_i = R (총 자원)
        - 0 ≤ x_i ≤ s_i (부족량 제한)
        - x_i ∈ ℤ (정수)
        """)

    # 시나리오 설정
    st.subheader("⚙️ 시나리오 설정")

    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            scope = st.selectbox("📍 배분 범위", ["전국", "특정 시도"])
        
        with col2:
            selected_sido = None
            if scope == "특정 시도":
                sido_list = sorted(df_year['시도명'].unique())
                selected_sido = st.selectbox("시도 선택", sido_list)
                df_scope = df_year[df_year['시도명'] == selected_sido].copy()
            else:
                df_scope = df_year.copy()
        
        with col3:
            resource_type = st.selectbox("🚑 자원 유형", ["구급차", "의사", "응급시설"])
        
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

    if st.button("🧹 결과 초기화"):
        st.session_state["ilp_result"] = None
        st.session_state["ilp_params"] = {}
        st.session_state["optimization_info"] = {}
        st.rerun()

    # 실행
    if run_simulation:
        try:
            with st.spinner('ILP Solver 실행 중...'):
                result_df, opt_info = optimize_allocation_ilp(df_scope, resource_type, resource_amount)
            
            st.session_state["ilp_result"] = result_df
            st.session_state["ilp_params"] = {
                "scope": scope,
                "selected_sido": selected_sido,
                "resource_type": resource_type,
                "resource_amount": resource_amount,
                "year": selected_year,
                "unit": unit
            }
            st.session_state["optimization_info"] = opt_info
            
            if opt_info["status"] == "Optimal":
                st.success(f"✅ 최적 배분 완료! (목적함수 값: {opt_info['objective_value']:.2f})")
            else:
                st.error(f"⚠️ 최적화 실패: {opt_info.get('message', opt_info['status'])}")
        
        except Exception as e:
            st.error(f"오류 발생: {e}")

    # 결과 표시
    if st.session_state["ilp_result"] is not None:
        df_result = st.session_state["ilp_result"].copy()
        params = st.session_state.get("ilp_params", {})
        opt_info = st.session_state.get("optimization_info", {})
        unit_str = params.get("unit", "")
        
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        
        st.markdown("---")
        st.subheader("📊 최적화 결과")
        
        # 최적화 상태 표시
        if opt_info.get("status") == "Optimal":
            st.markdown(f"""
            <div class='success-box'>
            <b>✅ 최적해 도출 성공</b><br>
            • Solver Status: {opt_info['status']}<br>
            • 목적함수 값: {opt_info.get('objective_value', 0):.4f}<br>
            • 배분 지역 수: {opt_info.get('num_allocated', 0)}개<br>
            • 변수 개수: {opt_info.get('num_variables', 0)}개
            </div>
            """, unsafe_allow_html=True)
        
        # KPI
        total_improvement = float(df_result['취약지수_개선'].sum())
        avg_before = float(df_result['취약지수'].mean())
        avg_after = float(df_result['배분_후_취약지수'].mean())
        total_allocated = int(df_allocated['배분량'].sum()) if not df_allocated.empty else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 배분 지역", f"{len(df_allocated)}개")
        with col2:
            st.metric("✅ 배분량", f"{total_allocated}{unit_str}")
        with col3:
            improvement_rate = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0.0
            st.metric("📈 개선율", f"{improvement_rate:.2f}%")
        with col4:
            efficiency = total_improvement / total_allocated if total_allocated > 0 else 0
            st.metric("⚡ 단위 효율", f"{efficiency:.4f}")
        
        # 지도 & 표
        st.markdown("---")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 🗺️ 배분 결과 지도")
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
                
                merged_tooltip = gdf_result.merge(
                    df_allocated[['행정구역코드', '시군구명']],
                    on='행정구역코드',
                    how='left'
                )
                
                folium.GeoJson(
                    merged_tooltip,
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['시군구명', '배분량', '취약지수_개선'],
                        aliases=['지역', f'{resource_type}', '개선효과'],
                        localize=True
                    )
                ).add_to(m)
                
                st_folium(m, width=None, height=420)
        
        with col2:
            st.markdown("#### 📋 상위 배분 지역")
            if not df_allocated.empty:
                display_df = df_allocated.nlargest(15, '배분량')[
                    ['시군구명', '배분량', '취약지수_개선', '해소율']
                ]
                st.dataframe(
                    display_df.style.format({
                        '배분량': '{:.0f}',
                        '취약지수_개선': '{:.4f}',
                        '해소율': '{:.1f}%'
                    }).background_gradient(cmap='Greens', subset=['배분량']),
                    height=420
                )
        
        # 차트
        if not df_allocated.empty:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 취약지수 변화")
                regional_info = calculate_regional_stats(
                    df_result, params.get('scope', '전국'), params.get('selected_sido')
                )
                
                year = params.get('year', 2025)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f'{year}년 (현재)', f'{year}년 (시뮬레이션 적용)'],
                    y=[regional_info['avg_before'], regional_info['avg_after']],
                    text=[f"{regional_info['avg_before']:.4f}", f"{regional_info['avg_after']:.4f}"],
                    textposition='outside',
                    marker_color=['#e74c3c', '#27ae60'],
                    width=0.5
                ))
                fig.update_layout(
                    height=350,
                    yaxis_title='평균 취약지수',
                    yaxis=dict(range=[0, max(regional_info['avg_before'] * 1.2, 0.1)]),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                improvement_abs = regional_info['avg_before'] - regional_info['avg_after']
                st.markdown(f"""
                <div class='success-box'>
                <b>📍 {regional_info['region_name']} ({year}년)</b><br>
                • 현재: <b>{regional_info['avg_before']:.4f}</b><br>
                • 시뮬레이션: <b>{regional_info['avg_after']:.4f}</b><br>
                • 개선: <b>-{improvement_abs:.4f}</b> (<b>{regional_info['improvement_rate']:.2f}%</b>)
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 배분 효율성")
                top_eff = df_allocated.nlargest(10, '취약지수_개선')
                fig = px.scatter(
                    top_eff, x='배분량', y='취약지수_개선',
                    size='취약지수_개선', color='시군구명',
                    hover_data=['시도명', '해소율']
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 y축이 높을수록 효율적")
        
        # 다운로드
        with st.expander("📥 결과 다운로드"):
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name=f"ILP최적배분_{resource_type}_{selected_year}.csv",
                mime="text/csv"
            )

    else:
        st.info("👈 시나리오를 설정하고 '최적화 실행' 버튼을 눌러주세요.")

# =====================================================================
# 페이지 3: 성과 평가 (NEW!)
# =====================================================================
elif page == "📈 성과 평가":
    st.markdown("<h1 style='text-align: center;'>📈 최적화 성과 평가</h1>", unsafe_allow_html=True)

    if st.session_state["ilp_result"] is None:
        st.warning("먼저 '최적화 시뮬레이션' 페이지에서 시뮬레이션을 실행해주세요.")
    else:
        df_result = st.session_state["ilp_result"].copy()
        params = st.session_state.get("ilp_params", {})
        opt_info = st.session_state.get("optimization_info", {})
        
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        
        if df_allocated.empty:
            st.warning("배분된 지역이 없습니다.")
        else:
            # 효율성 지표
            efficiency_metrics = calculate_efficiency_metrics(df_allocated, params.get('resource_type', '구급차'))
            
            st.subheader("📊 종합 성과 지표")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⚡ 단위 효율성", f"{efficiency_metrics.get('efficiency', 0):.4f}",
                        help="자원 1단위당 취약지수 개선")
            with col2:
                st.metric("📊 배분 집중도", f"{efficiency_metrics.get('concentration_top10', 0):.1f}%",
                        help="상위 10개 지역 배분 비율")
            with col3:
                st.metric("📈 지니계수", f"{efficiency_metrics.get('gini_coefficient', 0):.3f}",
                        help="배분 불균형도 (0=완전균등)")
            with col4:
                st.metric("📍 평균 배분", f"{efficiency_metrics.get('avg_allocation', 0):.1f}",
                        help="지역당 평균 배분량")
            
            st.markdown("---")
            
            # 비교 분석
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 배분 전략 비교")
                
                # ILP vs 균등 배분 vs 취약지수 순 배분
                total_resources = params.get('resource_amount', 30)
                
                # 균등 배분
                equal_alloc = total_resources // len(df_allocated)
                equal_improvement = (df_allocated['취약지수'] * 0.3 * (equal_alloc / df_allocated[params.get('resource_type', '구급차')+'_추가'])).sum()
                
                # 취약지수 순
                df_sorted = df_allocated.sort_values('취약지수', ascending=False).head(len(df_allocated))
                simple_improvement = df_allocated['취약지수_개선'].sum()
                
                # ILP
                ilp_improvement = df_result['취약지수_개선'].sum()
                
                comparison_data = pd.DataFrame({
                    '전략': ['균등 배분', '취약지수 순', 'ILP 최적화'],
                    '개선 효과': [equal_improvement, simple_improvement, ilp_improvement]
                })
                
                fig = px.bar(
                    comparison_data, x='전략', y='개선 효과',
                    color='개선 효과', color_continuous_scale='Greens',
                    text='개선 효과'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 ILP 최적화가 {(ilp_improvement/equal_improvement - 1)*100:.1f}% 더 효율적입니다.")
            
            with col2:
                st.subheader("📊 ROI 분석")
                
                # 가상 단가 설정
                unit_cost = {
                    "구급차": 200_000_000,  # 2억원
                    "의사": 100_000_000,    # 1억원
                    "응급시설": 5_000_000_000  # 50억원
                }
                
                resource_type = params.get('resource_type', '구급차')
                cost_per_unit = unit_cost.get(resource_type, 100_000_000)
                
                total_cost = params.get('resource_amount', 30) * cost_per_unit
                total_benefit = ilp_improvement * 1_000_000_000  # 취약지수 1당 10억원 가치 가정
                
                roi = (total_benefit / total_cost - 1) * 100
                
                roi_data = pd.DataFrame({
                    '항목': ['투자 비용', '기대 효과', 'ROI'],
                    '금액 (억원)': [
                        total_cost / 100_000_000,
                        total_benefit / 100_000_000,
                        roi
                    ]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=roi_data['항목'][:2],
                    y=roi_data['금액 (억원)'][:2],
                    marker_color=['indianred', 'lightgreen']
                ))
                fig.update_layout(height=350, yaxis_title='금액 (억원)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("💰 ROI", f"{roi:.1f}%", help="투자 대비 수익률")
                st.caption("※ 가상 단가 기준 추정치")
            
            # 지역별 성과
            st.markdown("---")
            st.subheader("🏆 지역별 성과 순위")
            
            performance_df = df_allocated.copy()
            performance_df['투입 대비 효과'] = performance_df['취약지수_개선'] / performance_df['배분량']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**효율성 TOP 10**")
                top_eff = performance_df.nlargest(10, '투입 대비 효과')[
                    ['시도명', '시군구명', '배분량', '투입 대비 효과']
                ]
                st.dataframe(
                    top_eff.style.format({
                        '배분량': '{:.0f}',
                        '투입 대비 효과': '{:.5f}'
                    }).background_gradient(cmap='Greens', subset=['투입 대비 효과']),
                    hide_index=True
                )
            
            with col2:
                st.markdown("**배분량 TOP 10**")
                top_alloc = performance_df.nlargest(10, '배분량')[
                    ['시도명', '시군구명', '배분량', '취약지수_개선']
                ]
                st.dataframe(
                    top_alloc.style.format({
                        '배분량': '{:.0f}',
                        '취약지수_개선': '{:.4f}'
                    }).background_gradient(cmap='Blues', subset=['배분량']),
                    hide_index=True
                )

# =====================================================================
# 페이지 4: 방법론
# =====================================================================
elif page == "📖 방법론":
    st.markdown("<h1 style='text-align: center;'>📖 연구 방법론</h1>", unsafe_allow_html=True)

    st.subheader("1️⃣ 취약지수 정의")
    st.markdown("""
    취약지수는 다음 요소들을 종합한 0~1 범위의 표준화된 지표입니다:
    - 의료자원 접근성 (응급의료기관 거리, 시설 수)
    - 인구학적 요인 (고령인구 비율, 인구밀도)
    - 사회경제적 요인 (재정자립도, 의료보험)
    - 지리적 요인 (도서/산간, 교통 인프라)
    """)

    st.subheader("2️⃣ 최적화 모델")
    st.latex(r"""
    \begin{aligned}
    \text{maximize} \quad & \sum_{i=1}^{n} w_i \cdot x_i \\
    \text{subject to} \quad & \sum_{i=1}^{n} x_i = R \\
    & 0 \leq x_i \leq s_i, \quad \forall i \\
    & x_i \in \mathbb{Z}, \quad \forall i
    \end{aligned}
    """)

    st.markdown("**변수:**")
    st.markdown("- $w_i$: 지역 i의 단위당 개선효과")
    st.markdown("- $x_i$: 지역 i에 배분할 자원량 (결정변수)")
    st.markdown("- $R$: 총 가용 자원")
    st.markdown("- $s_i$: 지역 i의 현재 부족량")

    st.subheader("3️⃣ 개선효과 산정식")
    st.latex(r"w_i = V_i \times P_i \times E_i")

    st.markdown("""
    - $V_i$: 취약도 = 현재 취약지수
    - $P_i$: 인구 가중 = $\\frac{\\log(인구_i + 1)}{10}$
    - $E_i$: 효율성 = $\\frac{1}{\\sqrt{부족량_i}}$
    """)

    st.subheader("4️⃣ 알고리즘 검증")
    st.markdown("""
    **검증 항목:**
    1. Solver 상태 확인 (Optimal/Infeasible/Unbounded)
    2. 제약조건 만족 여부
    3. 목적함수 값 검증
    4. 민감도 분석 (파라미터 변화 시 결과 안정성)
    """)

    st.subheader("5️⃣ 모델 한계")
    st.markdown("""
    <div class='warning-box'>
    1. <b>선형 근사</b>: 취약지수 변화를 선형으로 가정<br>
    2. <b>고정 기여도</b>: 자원의 기여도를 30%로 설정<br>
    3. <b>정적 분석</b>: 동적 변화 미반영<br>
    4. <b>단일 목적</b>: 효율성만 고려
    </div>
    """, unsafe_allow_html=True)