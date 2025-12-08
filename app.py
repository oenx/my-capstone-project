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
# 개선효과 계산 함수 (w_i)
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
# ILP 최적화 함수
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

    return df_result

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

# ---------------------------------------------------------------------
# 페이지 1: 현황 분석
# ---------------------------------------------------------------------
if page == "📊 현황 분석":
    st.markdown("<h1 style='text-align: center;'>🚑 응급의료 취약지 분석 대시보드</h1>", unsafe_allow_html=True)
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()
    merged_gdf = gdf.merge(df_year, on='행정구역코드', how='inner')

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
        st.metric("👨‍⚕️ 필요 의사", f"{needed_docs:,.0f}명")

    st.markdown("---")

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
        else:
            st.warning("지도 표시를 위한 지오데이터가 비어있습니다.")

    with row1_col2:
        st.subheader("📊 자원 부족 상위 지역 (Top 10)")
        tab1, tab2 = st.tabs(["필요 의사 수", "취약지수 순위"])
        with tab1:
            if '추가_의사수' in df_year.columns:
                top_docs = df_year.nlargest(10, '추가_의사수')
                if not top_docs.empty:
                    fig_doc = px.bar(top_docs, x='추가_의사수', y='시군구명', orientation='h', color='추가_의사수')
                    fig_doc.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_doc, use_container_width=True)
            else:
                st.info("필요 의사 수 데이터가 없습니다.")
        with tab2:
            if '취약지수' in df_year.columns:
                top_vul = df_year.nlargest(10, '취약지수')
                fig_vul = px.bar(top_vul, x='취약지수', y='시군구명', orientation='h', color='취약지수')
                fig_vul.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_vul, use_container_width=True)
            else:
                st.info("취약지수 데이터가 없습니다.")

    st.markdown("### 📋 상세 데이터")
    with st.expander("클릭하여 전체 데이터 확인"):
        show_cols = [c for c in ['시도명', '시군구명', '총인구', '고령인구_65세이상', '취약지수', '추가_의사수', '추가_구급차수', '추가_응급시설수'] if c in df_year.columns]
        if not show_cols:
            st.write(df_year.head(10))
        else:
            try:
                styled_df = df_year[show_cols].sort_values(by='취약지수', ascending=False).style.background_gradient(cmap='OrRd', subset=['취약지수']).format({'취약지수': '{:.3f}', '총인구': '{:,.0f}'})
                st.dataframe(styled_df)
            except:
                st.dataframe(df_year[show_cols].sort_values(by='취약지수', ascending=False))

# ---------------------------------------------------------------------
# 페이지 2: 시나리오 시뮬레이션
# ---------------------------------------------------------------------
elif page == "🎯 시나리오 시뮬레이션":
    st.markdown("<h1 style='text-align: center;'>🎯 응급자원 최적 배분 시뮬레이션</h1>", unsafe_allow_html=True)
    df_year = df[df['연도'] == selected_year] if '연도' in df.columns else df.copy()

    st.info("💡 정수계획법(ILP)을 사용해 전체 취약지수 개선량을 최대화하는 자원 배분 계산")

    st.subheader("⚙️ 시나리오 설정")
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
    clear_sim = st.button("🧹 결과 초기화", key="clear_ilp")
    if clear_sim:
        st.session_state["ilp_result"] = None
        st.session_state["ilp_params"] = {}

    # 버튼 클릭시 실행하고 결과를 session_state에 저장
    if run_simulation:
        try:
            with st.spinner('정수계획법(ILP)으로 최적해 계산 중...'):
                result_df = optimize_allocation_ilp(df_scope, resource_type, resource_amount)
            st.session_state["ilp_result"] = result_df
            st.session_state["ilp_params"] = {
                "scope": scope,
                "selected_sido": selected_sido,
                "resource_type": resource_type,
                "resource_amount": resource_amount,
                "year": selected_year
            }
            st.success("✅ 최적 배분 완료!")
        except Exception as e:
            st.error(f"최적화 실행 중 오류: {e}")

    # 화면에는 session_state의 결과를 사용 (rerun 방지)
    if st.session_state["ilp_result"] is not None:
        df_result = st.session_state["ilp_result"].copy()
        params = st.session_state.get("ilp_params", {})
        df_allocated = df_result[df_result['배분량'] > 0].copy()
        total_improvement = float(df_result['취약지수_개선'].sum()) if '취약지수_개선' in df_result.columns else 0.0
        avg_before = float(df_result['취약지수'].mean()) if '취약지수' in df_result.columns else 0.0
        avg_after = float(df_result['배분_후_취약지수'].mean()) if '배분_후_취약지수' in df_result.columns else 0.0
        total_allocated = int(df_allocated['배분량'].sum()) if not df_allocated.empty else 0

        st.markdown("---")
        st.subheader("📊 최적화 결과")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("🎯 배분 지역", f"{len(df_allocated)}개")
        with k2:
            st.metric("✅ 배분 완료", f"{total_allocated}{unit}")
        with k3:
            improvement_rate = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0.0
            st.metric("📈 평균 개선율", f"{improvement_rate:.1f}%")
        with k4:
            st.metric("✨ 총 개선 효과", f"{total_improvement:.4f}")

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
            st.markdown("#### 📋 배분 상세 (Top 15)")
            if not df_allocated.empty:
                display_df = df_allocated.nlargest(15, '배분량')[['시도명', '시군구명', '배분량', '취약지수_개선', '해소율']].fillna(0)
                st.dataframe(display_df.style.format({'배분량': '{:.0f}', '취약지수_개선': '{:.4f}', '해소율': '{:.1f}%'}), height=420)
            else:
                st.info("배분된 지역이 없습니다.")

        # 개선 효과 차트
        if not df_allocated.empty and '배분_후_취약지수' in df_result.columns:
            st.markdown("#### 📊 취약지수 개선 효과 (Top 10)")
            top10 = df_allocated.nlargest(10, '배분량')
            fig = go.Figure()
            fig.add_trace(go.Bar(y=top10['시군구명'], x=top10['취약지수'], name='배분 전', orientation='h'))
            fig.add_trace(go.Bar(y=top10['시군구명'], x=top10['배분_후_취약지수'], name='배분 후', orientation='h'))
            fig.update_layout(barmode='group', yaxis={'categoryorder':'total ascending'}, height=420, xaxis_title='취약지수')
            st.plotly_chart(fig, use_container_width=True)

        # 전체 결과 및 다운로드
        with st.expander("📋 전체 지역 배분 결과 보기"):
            display_full = df_result[df_result['배분량'] > 0] if len(df_result[df_result['배분량'] > 0]) > 0 else df_result.head(20)
            cols_to_show = [c for c in ['시도명', '시군구명', '취약지수', '배분량', '배분_후_취약지수', '취약지수_개선', '해소율'] if c in display_full.columns]
            st.dataframe(display_full[cols_to_show].sort_values('배분량', ascending=False).style.format({
                '취약지수': '{:.4f}', '배분량': '{:.0f}', '배분_후_취약지수': '{:.4f}', '취약지수_개선': '{:.4f}', '해소율': '{:.1f}%'
            }))

        csv = df_result.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(label="📥 최적화 결과 다운로드 (CSV)", data=csv, file_name=f"ILP_최적배분_{resource_type}_{selected_year}년.csv", mime="text/csv")

    else:
        st.info("ILP 최적 배분을 실행하려면 오른쪽 상단의 '🚀 최적화 실행' 버튼을 누르세요.")

# ---------------------------------------------------------------------
# 끝
# ---------------------------------------------------------------------
