import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pulp

st.set_page_config(
    page_title="응급의료 취약지 분석 대시보드",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 데이터 로드
# =============================================================================
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
    df, gdf = load_data()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# =============================================================================
# 취약지수 개선 효과 계산 함수
# =============================================================================
def calculate_improvement_per_unit(row, resource_type):
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }

    col_name = col_map[resource_type]
    shortage = row[col_name]

    if shortage <= 0:
        return 0

    vulnerability_weight = row['취약지수']
    population_weight = np.log1p(row['총인구']) / 10
    efficiency = 1.0 / np.sqrt(shortage)

    return vulnerability_weight * population_weight * efficiency


# =============================================================================
# ILP 최적화
# =============================================================================
def optimize_allocation_ilp(df_scope, resource_type, total_resources):
    col_map = {
        "구급차": "추가_구급차수",
        "의사": "추가_의사수",
        "응급시설": "추가_응급시설수"
    }
    col_name = col_map[resource_type]

    df_opt = df_scope.copy().reset_index(drop=True)
    df_opt['부족량'] = df_opt[col_name]
    df_opt = df_opt[df_opt['부족량'] > 0].copy()

    if df_opt.empty:
        st.warning("배분 가능한 지역 없음")
        return df_scope

    df_opt['개선효과'] = df_opt.apply(
        lambda row: calculate_improvement_per_unit(row, resource_type),
        axis=1
    )

    model = pulp.LpProblem("Emergency_Resource_Allocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", df_opt.index, lowBound=0, cat="Integer")

    model += pulp.lpSum(df_opt.loc[i, '개선효과'] * x[i] for i in df_opt.index)

    model += pulp.lpSum(x[i] for i in df_opt.index) == total_resources
    for i in df_opt.index:
        model += x[i] <= df_opt.loc[i, '부족량']

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    df_opt['배분량'] = [x[i].value() for i in df_opt.index]

    df_result = df_scope.copy()
    df_result['배분량'] = 0

    for i in df_opt.index:
        code = df_opt.loc[i, '행정구역코드']
        df_result.loc[df_result['행정구역코드'] == code, '배분량'] = df_opt.loc[i, '배분량']

    df_result['배분_후_부족'] = df_result[col_name] - df_result['배분량']
    df_result['해소율'] = (df_result['배분량'] / df_result[col_name] * 100).fillna(0)

    df_result['취약지수_개선'] = df_result.apply(
        lambda r: r['취약지수'] * 0.3 * (r['배분량'] / max(r[col_name], 1)) if r[col_name] > 0 else 0,
        axis=1
    )
    df_result['배분_후_취약지수'] = df_result['취약지수'] - df_result['취약지수_개선']

    return df_result


# =============================================================================
# 사이드바
# =============================================================================
st.sidebar.title("🚑 메뉴")
page = st.sidebar.radio("페이지", ["📊 현황 분석", "🎯 시나리오 시뮬레이션"])

year_list = sorted(df['연도'].unique())
selected_year = st.sidebar.select_slider("분석 연도", options=year_list, value=2025)

# =============================================================================
# 페이지 1: 현황 분석
# =============================================================================
if page == "📊 현황 분석":
    st.header("🚑 응급의료 취약지 현황 분석")

    df_year = df[df['연도'] == selected_year]

    merged = gdf.merge(df_year, on="행정구역코드", how="left")

    st.subheader("🗺 취약지수 지도")
    st.dataframe(df_year)

# =============================================================================
# 페이지 2: 시나리오 시뮬레이션
# =============================================================================
else:
    st.header("🎯 시나리오 기반 자원 배분 최적화")

    df_year = df[df["연도"] == selected_year].copy()

    st.subheader("지역 선택")
    area_option = st.selectbox("분석 단위 선택", ["전국"] + sorted(df_year["시도"].unique()))

    if area_option == "전국":
        df_scope = df_year.copy()
    else:
        df_scope = df_year[df_year["시도"] == area_option].copy()

    resource_type = st.selectbox("자원 종류", ["구급차", "의사", "응급시설"])
    total_resources = st.number_input("사용 가능한 자원 수", min_value=1, max_value=200, value=30)

    # 실행 버튼
    if st.button("최적 배분 실행"):
        st.session_state["ilp_result"] = optimize_allocation_ilp(df_scope, resource_type, total_resources)

    # -------------------------------------------------------------------------
    # 결과 시각화 (여기 전체가 새로 추가된 부분)
    # -------------------------------------------------------------------------
    if "ilp_result" in st.session_state and st.session_state["ilp_result"] is not None:
        df_res = st.session_state["ilp_result"]

        original_total = df_res["취약지수"].sum()
        improved_total = df_res["배분_후_취약지수"].sum()

        left, right = st.columns([1, 1])

        # ---- 왼쪽: 전체 취약지수 변화 (전국 또는 선택 시도)
        with left:
            st.subheader("📈 총 취약지수 변화")

            fig_line = go.Figure()
            fig_line.add_trace(go.Bar(
                x=["배분 전"], y=[original_total], name="배분 전"
            ))
            fig_line.add_trace(go.Bar(
                x=["배분 후"], y=[improved_total], name="배분 후"
            ))

            fig_line.update_layout(
                title="전체 취약지수 변화",
                yaxis_title="취약지수 합계",
                barmode="group"
            )

            st.plotly_chart(fig_line, use_container_width=True)

        # ---- 오른쪽: 기존 Top10 바차트
        with right:
            st.subheader("📊 취약지수 개선 효과 Top10")

            top10 = df_res.nlargest(10, "취약지수")

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=top10["시군구"],
                x=top10["취약지수"],
                name="배분 전",
                orientation="h"
            ))
            fig_bar.add_trace(go.Bar(
                y=top10["시군구"],
                x=top10["배분_후_취약지수"],
                name="배분 후",
                orientation="h"
            ))

            fig_bar.update_layout(
                xaxis_title="취약지수",
                yaxis_title="지역",
                barmode="group",
                height=600
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("📍 ILP 배분 결과 데이터")
        st.dataframe(df_res)

