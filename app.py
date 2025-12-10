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

# ---------------------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="응급의료 취약지 분석 대시보드",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [DESIGN] 커스텀 CSS 추가: 메트릭 박스, 헤더 스타일링
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #666;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #333;
        font-weight: 700;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Pretendard', sans-serif;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/data.csv')
    # 행정구역코드 형식 보정
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

# ---------------------------------------------------------------------
# 개선효과 계산 함수 (w_i) - 기존 로직 유지
# ---------------------------------------------------------------------
def calculate_improvement_per_unit(row, resource_type):
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
    vulnerability_weight = row.get('취약지수', 0.0)
    population_weight = np.log1p(row.get('총인구', 0)) / 10.0
    efficiency = 1.0 / np.sqrt(max(shortage, 1e-6))
    improvement = float(vulnerability_weight) * float(population_weight) * efficiency
    return improvement

# ---------------------------------------------------------------------
# ILP 최적화 함수 - 기존 로직 유지
# ---------------------------------------------------------------------
def optimize_allocation_ilp(df_scope, resource_type, total_resources):
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

    df_opt['개선효과'] = df_opt.apply(lambda r: calculate_improvement_per_unit(r, resource_type), axis=1)

    # 모델 생성
    model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)

    # 변수 정의
    x = pulp.LpVariable.dicts("x", df_opt.index, lowBound=0, cat="Integer")

    # 목적함수
    model += pulp.lpSum(df_opt.loc[i, '개선효과'] * x[i] for i in df_opt.index), "Total_Improvement"

    # 제약조건: 총량
    model += (pulp.lpSum(x[i] for i in df_opt.index) == int(total_resources), "Total_Resources")

    # 제약조건: 각 지역 상한(부족량)
    for i in df_opt.index:
        model += (x[i] <= int(df_opt.loc[i, '부족량']), f"Max_Shortage_{i}")

    # solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)

    # 결과 정리
    df_opt['배분량'] = df_opt.index.map(lambda i: int(x[i].value()) if x[i].value() is not None else 0)

    df_result = df_scope.copy()
    df_result['배분량'] = 0
    # 매칭: 행정구역코드 기준
    for i in df_opt.index:
        code = df_opt.loc[i, '행정구역코드']
        allocated = int(df_opt.loc[i, '배분량'])
        df_result.loc[df_result['행정구역코드'] == code, '배분량'] = allocated

    # 후처리
    df_result['배분_후_부족'] = df_result[col_name] - df_result['배분량']
    df_result['해소율'] = (df_result['배분량'] / df_result[col_name] * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # 취약지수 개선(간단 모델)
    def calc_vul_improve(row):
        try:
            if row[col_name] > 0:
                return row.get('취약지수', 0.0) * 0.3 * (row['배분량'] / max(row[col_name], 1))
            else:
                return 0.0
        except:
            return 0.0

    df_result['취약지수_개선'] = df_result.apply(calc_vul_improve, axis=1)
    df_result['배분_후_취약지수'] = df_result['취약지수'] - df_result['취약지수_개선']
    
    # 개선율(%) 계산: 원래 취약지수 대비 몇 % 개선되었는지
    df_result['개선율(%)'] = (df_result['취약지수_개선'] / df_result['취약지수'] * 100).replace([np.inf, -np.inf], 0).fillna(0)

    return df_result

# ---------------------------------------------------------------------
# 지역 취약지수 변화 계산 함수
# ---------------------------------------------------------------------
def calculate_regional_vulnerability_change(df_result, scope, selected_sido=None):
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

# ---------------------------------------------------------------------
# 시도별 취약지수 변화 계산 함수
# ---------------------------------------------------------------------
def calculate_sido_vulnerability_changes(df_result):
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

# ---------------------------------------------------------------------
# session_state 초기화 유틸
# ---------------------------------------------------------------------
if "ilp_result" not in st.session_state:
    st.session_state["ilp_result"] = None
if "ilp_params" not in st.session_state:
    st.session_state["ilp_params"] = {}

# ---------------------------------------------------------------------
# 사이드바 (공통)
# ---------------------------------------------------------------------
st.sidebar.title("🚑 메뉴")
page = st.sidebar.radio("페이지 선택", ["📊 현황 분석", "🎯 시나리오 시뮬레이션"])
st.sidebar.markdown("---")
st.sidebar.header("🔍 분석 옵션")

year_list = sorted(df['연도'].unique()) if '연도' in df.columns else [2025]
selected_year = st.sidebar.select_slider("분석 연도", options=year_list, value=year_list[-1])

st.sidebar.markdown("---")
st.sidebar.info(
    "**사용 가이드**\n\n"
    "1. **현황 분석**: 현재 응급의료 취약지 및 부족 자원 현황을 파악합니다.\n"
    "2. **시뮬레이션**: 한정된 자원(의사, 구급차 등)을 최적으로 배분했을 때의 효과를 예측합니다."
)

# ---------------------------------------------------------------------
# 페이지 1: 현황 분석
# ---------------------------------------------------------------------
if page == "📊 현황 분석":
    st.markdown("<h1 style='text-align: center;'>🚑 응급의료 취약지 분석 대시보드</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>{selected_year}년도 기준 데이터 분석 현황입니다.</p>", unsafe_allow_html=True)
    
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()
    merged_gdf = gdf.merge(df_year, on='행정구역코드', how='inner')

    # [UPDATE] KPI 메트릭을 좀 더 깔끔하게 배치
    col1, col2, col3, col4 = st.columns(4)
    total_pop = int(df_year['총인구'].sum()) if '총인구' in df_year.columns else 0
    vul_count = int(df_year['취약지역_여부'].sum()) if '취약지역_여부' in df_year.columns else int((df_year['취약지수'] > 0).sum() if '취약지수' in df_year.columns else 0)
    avg_vul_index = float(df_year['취약지수'].mean()) if '취약지수' in df_year.columns else 0.0
    needed_docs = int(df_year['추가_의사수'].sum()) if '추가_의사수' in df_year.columns else 0

    with col1:
        st.metric("👥 총 인구 수", f"{total_pop:,.0f}명")
    with col2:
        st.metric("🚨 취약지역 수", f"{vul_count}개")
    with col3:
        st.metric("📉 평균 취약지수", f"{avg_vul_index:.3f}")
    with col4:
        st.metric("👨‍⚕️ 총 필요 의사", f"{needed_docs:,.0f}명")

    st.markdown("---")

    # 지도와 차트 레이아웃
    row1_col1, row1_col2 = st.columns([3, 2])
    
    with row1_col1:
        st.subheader(f"🗺️ {selected_year}년 응급의료 취약지수 지도")
        if not merged_gdf.empty:
            center = [merged_gdf.geometry.centroid.y.mean(), merged_gdf.geometry.centroid.x.mean()]
            m = folium.Map(location=center, zoom_start=7, tiles='cartodbpositron')
            
            # [UPDATE] 툴팁 필드에 인구수 추가
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
                    fields=['시도명', '시군구명', '총인구', '취약지수', '추가_의사수', '추가_구급차수'],
                    aliases=['시도', '시군구', '인구(명)', '취약지수', '필요 의사', '필요 구급차'],
                    localize=True
                )
            ).add_to(m)
            st_folium(m, width=None, height=500)
        else:
            st.warning("지도 표시를 위한 지오데이터가 비어있습니다.")

    with row1_col2:
        st.subheader("📊 주요 부족 자원 현황")
        tab1, tab2, tab3 = st.tabs(["필요 의사 TOP 10", "취약지수 TOP 10", "인구 vs 취약성"])
        
        with tab1:
            if '추가_의사수' in df_year.columns:
                top_docs = df_year.nlargest(10, '추가_의사수')
                if not top_docs.empty:
                    fig_doc = px.bar(top_docs, x='추가_의사수', y='시군구명', orientation='h', 
                                     color='추가_의사수', color_continuous_scale='Reds',
                                     text='추가_의사수')
                    fig_doc.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_doc, use_container_width=True)
            else:
                st.info("필요 의사 수 데이터가 없습니다.")
        
        with tab2:
            if '취약지수' in df_year.columns:
                top_vul = df_year.nlargest(10, '취약지수')
                fig_vul = px.bar(top_vul, x='취약지수', y='시군구명', orientation='h', 
                                 color='취약지수', color_continuous_scale='Oranges',
                                 text_auto='.3f')
                fig_vul.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_vul, use_container_width=True)
            else:
                st.info("취약지수 데이터가 없습니다.")
        
        # [NEW FEATURE] 상관관계 분석 차트 추가
        with tab3:
            st.markdown("**인구수와 취약지수의 상관관계**")
            if '총인구' in df_year.columns and '취약지수' in df_year.columns:
                fig_scatter = px.scatter(
                    df_year, x='총인구', y='취약지수', 
                    hover_name='시군구명', color='시도명', size='추가_의사수',
                    size_max=15, opacity=0.7
                )
                fig_scatter.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("💡 원의 크기는 부족한 의사 수를 나타냅니다. 인구가 많은데 취약지수가 높은(우상단) 지역이 관리 우선순위가 높을 수 있습니다.")

    st.markdown("### 📋 데이터 상세 보기")
    with st.expander("클릭하여 전체 데이터 테이블 확인"):
        show_cols = [c for c in ['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수'] if c in df_year.columns]
        if not show_cols:
            st.write(df_year.head(10))
        else:
            try:
                styled_df = df_year[show_cols].sort_values(by='취약지수', ascending=False).style\
                    .background_gradient(cmap='OrRd', subset=['취약지수'])\
                    .bar(subset=['추가_의사수'], color='#FFA07A')\
                    .format({'취약지수': '{:.3f}', '총인구': '{:,.0f}'})
                st.dataframe(styled_df, use_container_width=True)
            except:
                st.dataframe(df_year[show_cols].sort_values(by='취약지수', ascending=False), use_container_width=True)

# ---------------------------------------------------------------------
# 페이지 2: 시나리오 시뮬레이션
# ---------------------------------------------------------------------
elif page == "🎯 시나리오 시뮬레이션":
    st.markdown("<h1 style='text-align: center;'>🎯 응급자원 최적 배분 시뮬레이션</h1>", unsafe_allow_html=True)
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()

    st.markdown("""
    <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;'>
    <b>💡 알고리즘 설명 (ILP)</b><br>
    한정된 예산(자원) 내에서 <b>전체 취약지수 개선 총량을 최대화</b>하는 최적의 배분 조합을 수학적으로 계산합니다.
    단순히 부족한 곳에 채우는 것이 아니라, <b>'투입 대비 개선 효과'</b>가 가장 큰 지역을 우선 선정합니다.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ 시나리오 설정")
    
    # [DESIGN] 입력 폼을 컨테이너로 감싸서 구분감 부여
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
            resource_amount = st.slider(f"추가 가능한 {resource_type} 수량", min_value=1, max_value=max_val, value=min(30, max_val))
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_simulation = st.button("🚀 최적화 실행", type="primary", use_container_width=True, key="run_ilp")

    # Clear 버튼
    if st.button("🧹 결과 초기화", key="clear_ilp"):
        st.session_state["ilp_result"] = None
        st.session_state["ilp_params"] = {}
        st.rerun()

    # 실행 로직
    if run_simulation:
        try:
            with st.spinner(f'{resource_type} {resource_amount}{unit}에 대한 최적 배분 계산 중...'):
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

    # 결과 화면
    if st.session_state["ilp_result"] is not None:
        df_result = st.session_state["ilp_result"].copy()
        params = st.session_state.get("ilp_params", {})
        unit_str = params.get("unit", "")
        
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        total_improvement = float(df_result['취약지수_개선'].sum()) if '취약지수_개선' in df_result.columns else 0.0
        avg_before = float(df_result['취약지수'].mean()) if '취약지수' in df_result.columns else 0.0
        avg_after = float(df_result['배분_후_취약지수'].mean()) if '배분_후_취약지수' in df_result.columns else 0.0
        total_allocated = int(df_allocated['배분량'].sum()) if not df_allocated.empty else 0

        st.markdown("---")
        st.subheader("📊 최적화 결과 리포트")

        # [NEW FEATURE] 자동 생성 인사이트 메시지
        if not df_allocated.empty:
            top_alloc_region = df_allocated.loc[df_allocated['배분량'].idxmax()]
            top_alloc_name = top_alloc_region['시군구명']
            top_alloc_val = int(top_alloc_region['배분량'])
            
            insight_msg = f"""
            <div class='insight-box'>
            <b>💡 Analysis Insight</b><br>
            시뮬레이션 결과, 총 <b>{len(df_allocated)}개 지역</b>에 자원이 배분되었습니다.<br>
            가장 많은 자원이 투입된 지역은 <b>{top_alloc_region['시도명']} {top_alloc_name}</b>이며, 
            단일 지역에 <b>{top_alloc_val}{unit_str}</b>가 배정되었습니다. 
            이를 통해 전체 취약지수 평균이 <b>{avg_before:.3f}</b>에서 <b>{avg_after:.3f}</b>로 개선되었습니다.
            </div>
            """
            st.markdown(insight_msg, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("🎯 배분 지역 수", f"{len(df_allocated)}개")
        with k2:
            st.metric("✅ 실제 배분량", f"{total_allocated}{unit_str}")
        with k3:
            improvement_rate = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0.0
            st.metric("📈 취약성 개선율", f"{improvement_rate:.1f}%")
        with k4:
            st.metric("✨ 총 효용(Objective)", f"{total_improvement:.2f}")

        col_map_for_merge = ['행정구역코드', '배분량', '취약지수_개선']
        if '행정구역코드' in gdf.columns and set(col_map_for_merge).issubset(df_result.columns):
            gdf_result = gdf.merge(df_result[col_map_for_merge], on='행정구역코드', how='inner')
        else:
            gdf_result = gdf.copy()

        # 지도 및 표
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### 🗺️ 최적 배분 결과 지도")
            if not gdf_result.empty and '배분량' in gdf_result.columns:
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
                
                merged_for_tooltip = gdf_result.merge(df_result[['행정구역코드', '시도명', '시군구명']], on='행정구역코드', how='left') if '시도명' in df_result.columns else gdf_result
                folium.GeoJson(
                    merged_for_tooltip,
                    name='배분 정보',
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['시군구명', '배분량', '취약지수_개선'] if '시군구명' in merged_for_tooltip.columns else ['배분량', '취약지수_개선'],
                        aliases=['지역', f'{resource_type} 배분', '개선 효과'],
                        localize=True
                    )
                ).add_to(m)
                st_folium(m, width=None, height=420)
            else:
                st.warning("지도에 표시할 배분 결과가 없습니다.")

        with col2:
            st.markdown("#### 📋 배분 상위 지역 (Top 15)")
            if not df_allocated.empty:
                display_df = df_allocated.nlargest(15, '배분량')[['시도명', '시군구명', '배분량', '개선율(%)', '해소율']].fillna(0)
                st.dataframe(
                    display_df.style.background_gradient(cmap='Greens', subset=['배분량'])
                    .format({'배분량': '{:.0f}', '개선율(%)': '{:.1f}%', '해소율': '{:.1f}%'}), 
                    height=420,
                    use_container_width=True
                )
            else:
                st.info("배분된 지역이 없습니다.")

        # 개선 효과 차트
        if not df_allocated.empty and '배분_후_취약지수' in df_result.columns:
            st.markdown("---")
            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.markdown("#### 📊 지역 전체 취약지수 변화")
                regional_info = calculate_regional_vulnerability_change(
                    df_result, 
                    params.get('scope', '전국'),
                    params.get('selected_sido')
                )
                
                # 꺾은선 그래프용 데이터: 현재 -> 시뮬레이션 적용 후
                current_year = params.get('year', 2024)
                next_year = current_year + 1
                
                # 평균 취약지수 사용 (0.xx 형태)
                avg_before = regional_info['avg_before']
                avg_after = regional_info['avg_after']
                improvement_pct = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0.0
                
                line_data = pd.DataFrame({
                    '연도': [f'{current_year}년 현재', f'{next_year}년 (시뮬레이션 적용)'],
                    '평균 취약지수': [avg_before, avg_after]
                })
                
                fig_regional = go.Figure()
                fig_regional.add_trace(go.Scatter(
                    x=line_data['연도'],
                    y=line_data['평균 취약지수'],
                    mode='lines+markers+text',
                    line=dict(color='#636EFA', width=3),
                    marker=dict(size=12, color=['#EF553B', '#00CC96']),
                    text=[f'{avg_before:.4f}', f'{avg_after:.4f}'],
                    textposition='top center',
                    textfont=dict(size=14, color='black'),
                    hovertemplate='%{x}<br>취약지수: %{y:.4f}<extra></extra>'
                ))
                
                fig_regional.update_layout(
                    height=300,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='평균 취약지수',
                    xaxis_title='',
                    yaxis=dict(range=[0, max(avg_before * 1.2, 0.1)])
                )
                st.plotly_chart(fig_regional, use_container_width=True)
                
                # 개선효과를 퍼센트로 명확하게 표시
                scope_name = regional_info['region_name']
                st.markdown(f"""
                <div style='background-color:#e8f4ea; padding:15px; border-radius:10px; border-left:4px solid #00CC96;'>
                    <b>📍 {scope_name}</b> 시뮬레이션 결과<br><br>
                    • 현재 평균 취약지수: <b>{avg_before:.4f}</b><br>
                    • 적용 후 평균 취약지수: <b>{avg_after:.4f}</b><br>
                    • <span style='color:#00CC96; font-size:1.2em;'><b>▼ {improvement_pct:.2f}% 개선</b></span>
                </div>
                """, unsafe_allow_html=True)
                
                # 시도별 변화 (전국 범위일 때만)
                if params.get('scope') == '전국':
                    with st.expander("시도별 개선 현황 보기"):
                        sido_changes = calculate_sido_vulnerability_changes(df_result)
                        if not sido_changes.empty:
                            fig_sido = px.bar(sido_changes, x='시도', y='개선율', color='개선율', 
                                            color_continuous_scale='Teal',
                                            text=sido_changes['개선율'].apply(lambda x: f'{x:.1f}%'))
                            fig_sido.update_traces(textposition='outside')
                            fig_sido.update_layout(yaxis_title='개선율 (%)')
                            st.plotly_chart(fig_sido, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📊 배분 효과성 (개선 효율 Top 10)")
                # [NEW FEATURE] 단순 배분량이 아니라, 개선 효율이 높은 곳을 시각화
                if '취약지수_개선' in df_allocated.columns:
                    top_eff = df_allocated.nlargest(10, '취약지수_개선')
                    fig_eff = px.scatter(
                        top_eff, x='배분량', y='취약지수_개선', size='취약지수_개선', color='시군구명',
                        hover_data=['시도명', '해소율'],
                        labels={'취약지수_개선': '총 개선효과', '배분량': '자원 투입량'}
                    )
                    fig_eff.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_eff, use_container_width=True)
                    st.caption("💡 y축이 높을수록 적은 자원으로도 큰 효과를 본 지역입니다.")

        # 다운로드 섹션
        with st.expander("📥 결과 데이터 다운로드"):
            display_full = df_result[df_result['배분량'] > 0] if len(df_result[df_result['배분량'] > 0]) > 0 else df_result.head(20)
            st.dataframe(display_full.sort_values('배분량', ascending=False), use_container_width=True)
            csv = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="CSV 다운로드", data=csv, file_name=f"ILP_최적배분_{resource_type}_{selected_year}년.csv", mime="text/csv")

    else:
        st.info("👈 왼쪽 사이드바에서 시나리오를 설정하고 '최적화 실행' 버튼을 눌러주세요.")