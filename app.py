# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pulp

# =====================================================================
# 페이지 기본 설정
# =====================================================================
st.set_page_config(
    page_title="응급의료 취약지 분석 대시보드",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    div[data-testid="stMetricLabel"] {font-size: 0.9rem !important; color: #666;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem !important; color: #333; font-weight: 700;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Pretendard', sans-serif;}
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
    .methodology-box {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 데이터 로드
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/data.csv')
    if '행정구역코드' in df.columns:
        df['행정구역코드'] = df['행정구역코드'].astype(str).str.zfill(5)
    
    gdf = gpd.read_file('data/sigungu.json')
    if 'SIG_CD' in gdf.columns:
        gdf = gdf.rename(columns={'SIG_CD': '행정구역코드'})
    if '행정구역코드' in gdf.columns:
        gdf['행정구역코드'] = gdf['행정구역코드'].astype(str).str.zfill(5)
    
    return df, gdf

try:
    with st.spinner('데이터를 불러오는 중입니다...'):
        df, gdf = load_data()
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# =====================================================================
# 개선효과 계산 (학술적 근거 강화)
# =====================================================================
def calculate_improvement_per_unit(row, resource_type):
    """
    자원 1단위당 취약지수 개선 효과 계산 (w_i)
    
    수식: w_i = V_i × P_i × E_i
    - V_i (취약도): 현재 취약지수
    - P_i (인구 가중치): log(인구+1) / 10 (규모의 경제 반영)
    - E_i (효율성): 1/√(부족량) (한계효용 체감 원리)
    
    이론적 근거:
    - 취약지수가 높을수록 개선의 사회적 가치 증가
    - 인구가 많을수록 수혜자 수 증가 (로그 변환으로 과도한 가중 방지)
    - 부족량이 적을수록 단위당 효과 증가 (한계효용 체감 법칙)
    """
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    
    col_name = col_map[resource_type]
    shortage = row.get(col_name, 0)
    
    try:
        shortage = float(shortage)
    except:
        shortage = 0
    
    if shortage <= 0:
        return 0.0
    
    # 취약도 (V_i)
    vulnerability_weight = float(row.get('취약지수', 0.0))
    
    # 인구 가중치 (P_i) - 로그 변환으로 규모의 경제 반영
    population = row.get('총인구', 0)
    population_weight = np.log1p(population) / 10.0
    
    # 효율성 (E_i) - 한계효용 체감
    efficiency = 1.0 / np.sqrt(max(shortage, 1e-6))
    
    # 총 개선효과
    improvement = vulnerability_weight * population_weight * efficiency
    
    return improvement

# =====================================================================
# ILP 최적화
# =====================================================================
def optimize_allocation_ilp(df_scope, resource_type, total_resources):
    """
    정수계획법(Integer Linear Programming) 기반 최적 배분
    
    [수학적 모델]
    목적함수: max Σ(w_i × x_i)
    제약조건:
      1. Σx_i = R (총 자원량)
      2. 0 ≤ x_i ≤ s_i (지역별 부족량 제한)
      3. x_i ∈ ℤ (정수 제약)
    
    여기서:
    - w_i: 지역 i의 단위당 개선효과
    - x_i: 지역 i에 배분할 자원량 (결정변수)
    - R: 총 가용 자원
    - s_i: 지역 i의 현재 부족량
    """
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    col_name = col_map[resource_type]
    
    df_opt = df_scope.copy().reset_index(drop=True)
    
    if col_name not in df_opt.columns:
        st.warning(f"데이터에 '{col_name}' 컬럼이 없습니다.")
        return df_scope.copy()
    
    df_opt['부족량'] = pd.to_numeric(df_opt[col_name], errors='coerce').fillna(0)
    df_opt = df_opt[df_opt['부족량'] > 0].copy()
    
    if df_opt.empty:
        st.warning("배분 가능한 지역이 없습니다.")
        return df_scope.copy()
    
    # 개선효과 계산
    df_opt['개선효과'] = df_opt.apply(
        lambda r: calculate_improvement_per_unit(r, resource_type), 
        axis=1
    )
    
    # PuLP 모델 생성
    model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)
    
    # 결정변수 정의
    x = pulp.LpVariable.dicts("x", df_opt.index, lowBound=0, cat="Integer")
    
    # 목적함수
    model += pulp.lpSum(
        df_opt.loc[i, '개선효과'] * x[i] for i in df_opt.index
    ), "Total_Improvement"
    
    # 제약조건 1: 총 자원량
    model += (
        pulp.lpSum(x[i] for i in df_opt.index) == int(total_resources),
        "Total_Resources"
    )
    
    # 제약조건 2: 지역별 부족량 상한
    for i in df_opt.index:
        model += (
            x[i] <= int(df_opt.loc[i, '부족량']),
            f"Max_Shortage_{i}"
        )
    
    # 최적화 실행
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = model.solve(solver)
    
    # 결과 처리
    df_opt['배분량'] = df_opt.index.map(
        lambda i: int(x[i].value()) if x[i].value() is not None else 0
    )
    
    # 원본 데이터프레임에 병합
    df_result = df_scope.copy()
    df_result['배분량'] = 0
    
    for i in df_opt.index:
        code = df_opt.loc[i, '행정구역코드']
        allocated = int(df_opt.loc[i, '배분량'])
        df_result.loc[df_result['행정구역코드'] == code, '배분량'] = allocated
    
    # 후처리: 배분 후 지표 계산
    df_result['배분_후_부족'] = df_result[col_name] - df_result['배분량']
    df_result['해소율'] = (
        df_result['배분량'] / df_result[col_name] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 취약지수 개선 추정 (선형 근사 모델)
    def calc_vul_improve(row):
        """
        가정: 자원 부족이 취약지수에 30% 기여
        (실제 연구에서는 회귀분석 등으로 정확한 기여도 산출 필요)
        """
        try:
            if row[col_name] > 0:
                resource_contribution = 0.30  # 자원의 취약지수 기여도
                resolution_rate = row['배분량'] / max(row[col_name], 1)
                return row.get('취약지수', 0.0) * resource_contribution * resolution_rate
            else:
                return 0.0
        except:
            return 0.0
    
    df_result['취약지수_개선'] = df_result.apply(calc_vul_improve, axis=1)
    df_result['배분_후_취약지수'] = df_result['취약지수'] - df_result['취약지수_개선']
    df_result['개선율(%)'] = (
        df_result['취약지수_개선'] / df_result['취약지수'] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df_result

# =====================================================================
# 분석 함수
# =====================================================================
def calculate_regional_vulnerability_change(df_result, scope, selected_sido=None):
    """지역 전체 취약지수 변화 계산"""
    if scope == "특정 시도" and selected_sido:
        df_analysis = df_result[df_result['시도명'] == selected_sido].copy()
        region_name = selected_sido
    else:
        df_analysis = df_result.copy()
        region_name = "전국"
    
    total_before = float(df_analysis['취약지수'].sum())
    total_after = float(df_analysis['배분_후_취약지수'].sum())
    improvement = total_before - total_after
    improvement_rate = (improvement / total_before * 100) if total_before > 0 else 0.0
    
    return {
        'region_name': region_name,
        'before': total_before,
        'after': total_after,
        'improvement': improvement,
        'improvement_rate': improvement_rate,
        'num_regions': len(df_analysis),
        'avg_before': float(df_analysis['취약지수'].mean()),
        'avg_after': float(df_analysis['배분_후_취약지수'].mean())
    }

def calculate_sido_vulnerability_changes(df_result):
    """시도별 취약지수 변화"""
    if '시도명' not in df_result.columns:
        return pd.DataFrame()
    
    sido_changes = []
    for sido in df_result['시도명'].unique():
        df_sido = df_result[df_result['시도명'] == sido]
        before = float(df_sido['취약지수'].sum())
        after = float(df_sido['배분_후_취약지수'].sum())
        improvement = before - after
        sido_changes.append({
            '시도': sido,
            '배분전': before,
            '배분후': after,
            '개선효과': improvement,
            '개선율': (improvement / before * 100) if before > 0 else 0.0
        })
    
    return pd.DataFrame(sido_changes).sort_values('개선효과', ascending=False)

# =====================================================================
# 세션 상태 초기화
# =====================================================================
if "ilp_result" not in st.session_state:
    st.session_state["ilp_result"] = None
if "ilp_params" not in st.session_state:
    st.session_state["ilp_params"] = {}

# =====================================================================
# 사이드바
# =====================================================================
st.sidebar.title("🚑 네비게이션")
page = st.sidebar.radio(
    "페이지 선택",
    ["🏠 프로젝트 개요", "📊 현황 분석", "🎯 시나리오 시뮬레이션", "📖 방법론"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 분석 설정")

year_list = sorted(df['연도'].unique()) if '연도' in df.columns else [2025]
selected_year = st.sidebar.select_slider("분석 연도", options=year_list, value=year_list[-1])

st.sidebar.markdown("---")
st.sidebar.info(
    "**프로젝트 정보**\n\n"
    "- 과제명: 응급의료 취약지 분석 및 자원 최적배분\n"
    "- 데이터: 공공데이터포털 (2025~2040 예측)\n"
    "- 알고리즘: Integer Linear Programming\n"
    "- 개발도구: Python, Streamlit, PuLP"
)

# =====================================================================
# 페이지 0: 프로젝트 개요
# =====================================================================
if page == "🏠 프로젝트 개요":
    st.markdown("<h1 style='text-align: center;'>🚑 응급의료 취약지 분석 및 필수자원 예측</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 1.1rem;'>데이터 기반 의사결정 지원 시스템 (Capstone Project)</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 연구 배경
    st.subheader("📌 연구 배경 및 목적")
    st.markdown("""
    <div class='insight-box'>
    <b>Problem Statement</b><br>
    • 한국의 고령화 가속화로 응급의료 수요 급증 전망<br>
    • 지역 간 의료자원 불균형으로 인한 응급의료 사각지대 존재<br>
    • 한정된 예산 내에서 효율적인 자원 배분 전략 필요<br><br>
    
    <b>Research Objective</b><br>
    1. 전국 시군구별 응급의료 취약지수 시각화 및 현황 분석<br>
    2. 정수계획법(ILP)을 활용한 최적 자원 배분 알고리즘 개발<br>
    3. 시나리오 기반 정책 시뮬레이션 도구 제공
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 명세
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 데이터 명세")
        st.markdown("""
        **데이터 출처**
        - 행정안전부: 행정구역 경계 데이터
        - 통계청: 인구 통계 (2025~2040 예측)
        - 보건복지부: 응급의료기관 현황
        
        **주요 변수**
        - 취약지수: 0~1 (높을수록 취약)
        - 추가 필요 자원: 의사, 구급차, 응급시설
        - 인구학적 특성: 총인구, 고령인구 등
        
        **데이터 범위**
        - 공간: 전국 228개 시군구
        - 시간: 2025년~2040년 (연단위)
        """)
    
    with col2:
        st.subheader("🔬 연구 방법론")
        st.markdown("""
        **1단계: 현황 분석**
        - 지역별 취약지수 분포 시각화
        - 자원 부족 현황 통계 분석
        
        **2단계: 최적화 모델링**
        - ILP 기반 자원 배분 최적화
        - 목적함수: 전체 취약지수 개선 최대화
        - 제약조건: 예산 제한, 지역별 수요 상한
        
        **3단계: 시뮬레이션**
        - 다양한 시나리오 분석
        - 정책 대안 효과 비교
        """)
    
    # 주요 기능
    st.markdown("---")
    st.subheader("✨ 시스템 주요 기능")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:20px; border-radius:10px; height:200px;'>
        <h4>📊 현황 분석</h4>
        • 인터랙티브 지도 시각화<br>
        • 지역별 취약도 순위<br>
        • 자원 부족 현황 대시보드<br>
        • 인구-취약성 상관분석
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color:#f3e5f5; padding:20px; border-radius:10px; height:200px;'>
        <h4>🎯 최적화</h4>
        • ILP 기반 수학적 최적해<br>
        • 시나리오별 배분 전략<br>
        • 전국/지역 단위 분석<br>
        • 자원 유형별 시뮬레이션
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; height:200px;'>
        <h4>📈 효과 분석</h4>
        • 배분 전후 비교<br>
        • 취약지수 개선율 계산<br>
        • 시도별 변화 추이<br>
        • CSV 결과 다운로드
        </div>
        """, unsafe_allow_html=True)
    
    # 연구 한계
    st.markdown("---")
    st.subheader("⚠️ 연구 한계 및 향후 과제")
    st.markdown("""
    <div class='warning-box'>
    <b>현재 모델의 한계점</b><br>
    1. <b>취약지수 재계산 간소화</b>: 자원 배분 후 취약지수 변화를 선형 모델로 근사 (실제로는 비선형 관계 가능)<br>
    2. <b>자원 기여도 가정</b>: 자원이 취약지수에 미치는 영향을 30%로 가정 (실증 데이터 필요)<br>
    3. <b>정적 분석</b>: 동적 수요 변화, 지역 간 이동 등 미반영<br>
    4. <b>단일 목적함수</b>: 형평성, 접근성 등 다목적 최적화 미구현<br><br>
    
    <b>향후 개선 방향</b><br>
    • 머신러닝 기반 취약지수 예측 모델 개발<br>
    • 실제 응급의료 데이터를 활용한 모델 검증<br>
    • 다목적 최적화 (Multi-Objective Optimization) 적용<br>
    • 실시간 데이터 연동 및 모니터링 시스템 구축
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
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    total_pop = int(df_year['총인구'].sum()) if '총인구' in df_year.columns else 0
    vul_count = int(df_year['취약지역_여부'].sum()) if '취약지역_여부' in df_year.columns else int((df_year['취약지수'] > 0).sum() if '취약지수' in df_year.columns else 0)
    avg_vul_index = float(df_year['취약지수'].mean()) if '취약지수' in df_year.columns else 0.0
    needed_docs = int(df_year['추가_의사수'].sum()) if '추가_의사수' in df_year.columns else 0
    
    with col1:
        st.metric("👥 총 인구", f"{total_pop:,.0f}명")
    with col2:
        st.metric("🚨 취약지역", f"{vul_count}개 지역")
    with col3:
        st.metric("📉 평균 취약지수", f"{avg_vul_index:.3f}")
    with col4:
        st.metric("👨‍⚕️ 필요 의사", f"{needed_docs:,.0f}명")
    
    st.markdown("---")
    
    # 지도 & 차트
    row1_col1, row1_col2 = st.columns([3, 2])
    
    with row1_col1:
        st.subheader(f"🗺️ 취약지수 분포 지도")
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
                legend_name='취약지수 (0~1)'
            ).add_to(m)
            
            folium.GeoJson(
                merged_gdf,
                name='지역 정보',
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                tooltip=folium.GeoJsonTooltip(
                    fields=['시도명', '시군구명', '총인구', '취약지수', '추가_의사수', '추가_구급차수'],
                    aliases=['시도', '시군구', '인구', '취약지수', '필요 의사', '필요 구급차'],
                    localize=True
                )
            ).add_to(m)
            
            st_folium(m, width=None, height=500)
        else:
            st.warning("지도 데이터가 없습니다.")
    
    with row1_col2:
        st.subheader("📊 주요 지표 분석")
        tab1, tab2, tab3 = st.tabs(["필요 의사 TOP 10", "취약지수 TOP 10", "인구-취약성 분석"])
        
        with tab1:
            if '추가_의사수' in df_year.columns:
                top_docs = df_year.nlargest(10, '추가_의사수')
                fig_doc = px.bar(
                    top_docs, x='추가_의사수', y='시군구명', orientation='h',
                    color='추가_의사수', color_continuous_scale='Reds',
                    text='추가_의사수'
                )
                fig_doc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_doc, use_container_width=True)
        
        with tab2:
            if '취약지수' in df_year.columns:
                top_vul = df_year.nlargest(10, '취약지수')
                fig_vul = px.bar(
                    top_vul, x='취약지수', y='시군구명', orientation='h',
                    color='취약지수', color_continuous_scale='Oranges'
                )
                fig_vul.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_vul, use_container_width=True)
        
        with tab3:
            if '총인구' in df_year.columns and '취약지수' in df_year.columns:
                fig_scatter = px.scatter(
                    df_year, x='총인구', y='취약지수',
                    hover_name='시군구명', color='시도명',
                    size='추가_의사수', size_max=15, opacity=0.7
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("💡 원 크기 = 필요 의사 수. 우상단 지역이 우선순위 높음")
    
    # 상세 데이터
    st.markdown("---")
    st.markdown("### 📋 상세 데이터 테이블")
    with st.expander("클릭하여 전체 데이터 확인"):
        show_cols = [c for c in ['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수'] if c in df_year.columns]
        if show_cols:
            try:
                styled_df = df_year[show_cols].sort_values(by='취약지수', ascending=False).style\
                    .background_gradient(cmap='OrRd', subset=['취약지수'])\
                    .format({'취약지수': '{:.3f}', '총인구': '{:,.0f}'})
                st.dataframe(styled_df, use_container_width=True)
            except:
                st.dataframe(df_year[show_cols].sort_values(by='취약지수', ascending=False), use_container_width=True)

# =====================================================================
# 페이지 2: 시나리오 시뮬레이션
# =====================================================================
elif page == "🎯 시나리오 시뮬레이션":
    st.markdown("<h1 style='text-align: center;'>🎯 응급자원 최적 배분 시뮬레이션</h1>", unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()
    
    # 알고리즘 설명
    st.markdown("""
    <div class='methodology-box'>
    <h4>🔬 최적화 알고리즘 (ILP)</h4>
    <b>목적함수:</b> max Σ(w<sub>i</sub> × x<sub>i</sub>)<br>
    <b>제약조건:</b><br>
    • Σx<sub>i</sub> = R (총 자원량)<br>
    • 0 ≤ x<sub>i</sub> ≤ s<sub>i</sub> (지역별 부족량 제한)<br>
    • x<sub>i</sub> ∈ ℤ (정수 제약)<br><br>
    
    <b>개선효과 계산식:</b> w<sub>i</sub> = V<sub>i</sub> × P<sub>i</sub> × E<sub>i</sub><br>
    • V<sub>i</sub>: 취약도 (현재 취약지수)<br>
    • P<sub>i</sub>: 인구 가중치 = log(인구+1)/10<br>
    • E<sub>i</sub>: 효율성 = 1/√(부족량)<br><br>
    
    <small>※ 실제 정책 결정 시 추가 고려사항: 형평성, 접근성, 지역 특수성 등</small>
    </div>
    """, unsafe_allow_html=True)
    
    # 시나리오 설정
    st.subheader("⚙️ 시나리오 설정")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            scope = st.selectbox("📍 배분 범위", ["전국", "특정 시도"])
        
        with col2:
            selected_sido = None
            if scope == "특정 시도":
                sido_list = sorted(df_year['시도명'].unique()) if '시도명' in df_year.columns else []
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
    
    # Clear 버튼
    if st.button("🧹 결과 초기화"):
        st.session_state["ilp_result"] = None
        st.session_state["ilp_params"] = {}
        st.rerun()
    
    # 실행
    if run_simulation:
        try:
            with st.spinner('ILP Solver 실행 중...'):
                result_df = optimize_allocation_ilp(df_scope, resource_type, resource_amount)
            st.session_state["ilp_result"] = result_df
            st.session_state["ilp_params"] = {
                "scope": scope,
                "selected_sido": selected_sido,
                "resource_type": resource_type,
                "resource_amount": resource_amount,
                "year": selected_year,
                "unit": unit
            }
            st.success("✅ 최적 배분 완료!")
        except Exception as e:
            st.error(f"최적화 실행 중 오류: {e}")
    
    # 결과 표시
    if st.session_state["ilp_result"] is not None:
        df_result = st.session_state["ilp_result"].copy()
        params = st.session_state.get("ilp_params", {})
        unit_str = params.get("unit", "")
        
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        total_improvement = float(df_result['취약지수_개선'].sum())
        avg_before = float(df_result['취약지수'].mean())
        avg_after = float(df_result['배분_후_취약지수'].mean())
        total_allocated = int(df_allocated['배분량'].sum()) if not df_allocated.empty else 0
        
        st.markdown("---")
        st.subheader("📊 최적화 결과")
        
        # 인사이트
        if not df_allocated.empty:
            top_region = df_allocated.loc[df_allocated['배분량'].idxmax()]
            st.markdown(f"""
            <div class='insight-box'>
            <b>📍 주요 결과</b><br>
            • 총 <b>{len(df_allocated)}개 지역</b>에 자원 배분<br>
            • 최다 배분 지역: <b>{top_region['시도명']} {top_region['시군구명']}</b> ({int(top_region['배분량'])}{unit_str})<br>
            • 전체 평균 취약지수: <b>{avg_before:.4f}</b> → <b>{avg_after:.4f}</b>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 배분 지역", f"{len(df_allocated)}개")
        with col2:
            st.metric("✅ 배분량", f"{total_allocated}{unit_str}")
        with col3:
            improvement_rate = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0.0
            st.metric("📈 개선율", f"{improvement_rate:.2f}%")
        with col4:
            st.metric("✨ 목적함수 값", f"{total_improvement:.2f}")
        
        # 지도 & 표
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
            st.markdown("#### 📋 상위 배분 지역 (Top 15)")
            if not df_allocated.empty:
                display_df = df_allocated.nlargest(15, '배분량')[
                    ['시도명', '시군구명', '배분량', '개선율(%)', '해소율']
                ].fillna(0)
                st.dataframe(
                    display_df.style.format({
                        '배분량': '{:.0f}',
                        '개선율(%)': '{:.2f}%',
                        '해소율': '{:.1f}%'
                    }),
                    height=420
                )
        
        # 차트
        if not df_allocated.empty:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 취약지수 변화 (시뮬레이션 효과)")
                regional_info = calculate_regional_vulnerability_change(
                    df_result, params.get('scope', '전국'), params.get('selected_sido')
                )
                
                year = params.get('year', 2025)
                
                # 같은 연도 내에서 "현재 상태" vs "시뮬레이션 적용 시" 비교
                line_data = pd.DataFrame({
                    '시나리오': [f'{year}년 (현재)', f'{year}년 (시뮬레이션 적용)'],
                    '평균 취약지수': [regional_info['avg_before'], regional_info['avg_after']]
                })
                
                fig = go.Figure()
                
                # 막대 그래프로 시각적 차이 강조
                fig.add_trace(go.Bar(
                    x=line_data['시나리오'],
                    y=line_data['평균 취약지수'],
                    text=line_data['평균 취약지수'].apply(lambda x: f'{x:.4f}'),
                    textposition='outside',
                    marker_color=['#e74c3c', '#27ae60'],  # 빨강(나쁨) -> 초록(개선)
                    width=0.5
                ))
                
                fig.update_layout(
                    height=350,
                    yaxis_title='평균 취약지수',
                    yaxis=dict(range=[0, max(regional_info['avg_before'] * 1.2, 0.1)]),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                improvement_pct = regional_info['improvement_rate']
                improvement_abs = regional_info['avg_before'] - regional_info['avg_after']
                
                st.markdown(f"""
                <div style='background-color:#d4edda; padding:15px; border-radius:10px; border-left:4px solid #28a745;'>
                <b>📍 {regional_info['region_name']} ({year}년)</b><br><br>
                • 현재 상태: <b>{regional_info['avg_before']:.4f}</b><br>
                • 시뮬레이션 적용 시: <b>{regional_info['avg_after']:.4f}</b><br>
                • 개선 효과: <b>-{improvement_abs:.4f}</b> (<span style='color:#27ae60; font-size:1.1em;'><b>▼ {improvement_pct:.2f}%</b></span>)
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"💡 {year}년 동일 시점에서 자원 배분 시 즉각적인 취약지수 개선 효과를 보여줍니다.")
            
            with col2:
                st.markdown("#### 📊 배분 효율성 분석")
                top_eff = df_allocated.nlargest(10, '취약지수_개선')
                fig = px.scatter(
                    top_eff, x='배분량', y='취약지수_개선',
                    size='취약지수_개선', color='시군구명',
                    hover_data=['시도명', '해소율']
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 y축이 높을수록 효율적인 배분")
        
        # 다운로드
        with st.expander("📥 결과 데이터 다운로드"):
            st.dataframe(
                df_result[df_result['배분량'] > 0].sort_values('배분량', ascending=False),
                use_container_width=True
            )
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name=f"ILP최적배분_{resource_type}_{selected_year}년.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 시나리오를 설정하고 '최적화 실행' 버튼을 눌러주세요.")

# =====================================================================
# 페이지 3: 방법론
# =====================================================================
elif page == "📖 방법론":
    st.markdown("<h1 style='text-align: center;'>📖 연구 방법론</h1>", unsafe_allow_html=True)
    
    st.subheader("1️⃣ 취약지수 정의")
    st.markdown("""
    <div class='methodology-box'>
    <b>취약지수 (Vulnerability Index)</b>는 다음 요소들을 종합하여 0~1 사이의 값으로 표준화한 지표입니다:<br><br>
    
    • <b>의료자원 접근성</b>: 응급의료기관까지의 거리, 이용 가능한 시설 수<br>
    • <b>인구학적 요인</b>: 고령인구 비율, 인구밀도<br>
    • <b>사회경제적 요인</b>: 재정자립도, 의료보험 가입률<br>
    • <b>지리적 요인</b>: 도서/산간 지역 여부, 교통 인프라<br><br>
    
    <small>※ 본 프로젝트에서는 전처리된 취약지수를 활용하며, 실제 산출 과정은 별도 연구로 진행됨</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("2️⃣ 최적화 수식")
    st.latex(r"""
    \begin{aligned}
    \text{maximize} \quad & \sum_{i=1}^{n} w_i \cdot x_i \\
    \text{subject to} \quad & \sum_{i=1}^{n} x_i = R \\
    & 0 \leq x_i \leq s_i, \quad \forall i \\
    & x_i \in \mathbb{Z}, \quad \forall i
    \end{aligned}
    """)
    
    st.markdown("""
    **변수 설명:**
    - $w_i$: 지역 $i$의 단위당 개선효과
    - $x_i$: 지역 $i$에 배분할 자원량 (결정변수)
    - $R$: 총 가용 자원
    - $s_i$: 지역 $i$의 현재 부족량
    - $n$: 전체 지역 수
    """)
    
    st.subheader("3️⃣ 개선효과 계산식")
    st.latex(r"""
    w_i = V_i \times P_i \times E_i
    """)
    
    st.markdown("""
    where:
    - $V_i$ = 취약도 (현재 취약지수)
    - $P_i$ = 인구 가중치 = $\frac{\log(인구_i + 1)}{10}$
    - $E_i$ = 효율성 = $\frac{1}{\sqrt{부족량_i}}$
    """)
    
    st.info("""
    **이론적 근거:**
    - **취약도 가중**: 취약한 지역일수록 개선의 사회적 가치 증가
    - **인구 가중**: 더 많은 인구가 혜택을 받도록 (로그 변환으로 과도한 편향 방지)
    - **효율성**: 한계효용 체감 법칙 반영 (부족량이 적을수록 단위당 효과 증가)
    """)
    
    st.subheader("4️⃣ 알고리즘 구현")
    st.code("""
# PuLP를 사용한 ILP 모델
model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)

# 결정변수 (정수)
x = pulp.LpVariable.dicts("x", regions, lowBound=0, cat="Integer")

# 목적함수
model += pulp.lpSum(improvement[i] * x[i] for i in regions)

# 제약조건
model += pulp.lpSum(x[i] for i in regions) == total_resources
for i in regions:
    model += x[i] <= shortage[i]

# 최적화 실행
model.solve()
    """, language="python")
    
    st.subheader("5️⃣ 모델 검증")
    st.markdown("""
    <div class='warning-box'>
    <b>⚠️ 모델 한계 및 가정</b><br><br>
    
    1. <b>선형 근사</b>: 자원 배분 후 취약지수 변화를 선형 모델로 근사<br>
    &nbsp;&nbsp;&nbsp;→ 실제로는 비선형 관계일 가능성 존재<br><br>
    
    2. <b>고정 기여도</b>: 자원의 취약지수 기여도를 30%로 가정<br>
    &nbsp;&nbsp;&nbsp;→ 지역별, 자원별로 기여도가 다를 수 있음<br><br>
    
    3. <b>정적 분석</b>: 시간에 따른 변화, 지역 간 상호작용 미반영<br>
    &nbsp;&nbsp;&nbsp;→ 동적 최적화 모델로 확장 필요<br><br>
    
    4. <b>단일 목적</b>: 효율성만 고려, 형평성 등 다른 목표 미반영<br>
    &nbsp;&nbsp;&nbsp;→ 다목적 최적화로 개선 가능
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("6️⃣ 참고문헌")
    st.markdown("""
    - 보건복지부 (2024). 응급의료기관 현황통계
    - 통계청 (2024). 장래인구추계
    - Lee et al. (2023). "Optimization of Emergency Medical Resource Allocation"
    - Kim & Park (2022). "Vulnerability Assessment in Korean Healthcare System"
    """)