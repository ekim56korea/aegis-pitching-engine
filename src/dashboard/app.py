"""
Aegis Pitching Engine - Interactive 3D Strategy Room
====================================================

Streamlit 기반 대시보드로 실시간 투구 전략을 시각화하고 시뮬레이션합니다.

Features:
- 실시간 게임 상황 입력
- AI 전략 추천
- 3D 투구 궤적 시각화 (Plotly)
- 물리 지표 데이터 테이블
- 성능 최적화 (caching)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.common.config import StrategyConfig
from src.data_pipeline.data_loader import AegisDataLoader
from src.game_theory.engine import AegisStrategyEngine, DecisionResult


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Aegis Strategy Room",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Resource Caching (Performance Optimization)
# ============================================================================
@st.cache_resource
def load_data_loader():
    """AegisDataLoader를 캐싱하여 재사용"""
    try:
        loader = AegisDataLoader()
        return loader
    except FileNotFoundError:
        st.warning("⚠️ DuckDB 파일을 찾을 수 없습니다. 기본 데이터 사용.")
        return None


@st.cache_resource
def load_strategy_engine():
    """AegisStrategyEngine을 캐싱하여 재사용"""
    engine = AegisStrategyEngine(device='cpu')
    return engine


@st.cache_data
def load_pitcher_data(pitcher_id: int, year: int = 2024):
    """투수 데이터를 캐싱"""
    loader = load_data_loader()
    if loader is None:
        return get_default_pitcher_stats()

    try:
        df = loader.load_pitcher_data(pitcher_id)
        if df.empty:
            return get_default_pitcher_stats()

        # 연도 필터링
        if 'game_year' in df.columns:
            df = df[df['game_year'] == year]
            if df.empty:
                df = loader.load_pitcher_data(pitcher_id)

        # 구종별 구사율
        pitch_usage_stats = {}
        if 'pitch_type' in df.columns:
            pitch_counts = df['pitch_type'].value_counts()
            total_pitches = len(df)
            pitch_usage_stats = {
                pitch: count / total_pitches
                for pitch, count in pitch_counts.items()
            }

        # 샘플 수
        sample_sizes = dict(pitch_counts) if 'pitch_type' in df.columns else {}

        # Stuff+ 추정
        stuff_plus = {}
        if 'pitch_type' in df.columns and 'release_speed' in df.columns:
            for pitch_type in pitch_usage_stats.keys():
                pitch_df = df[df['pitch_type'] == pitch_type]
                avg_velo = pitch_df['release_speed'].mean()
                stuff_plus[pitch_type] = 100 + (avg_velo - 90.0) * 2.0

        # 존별 제구
        zone_command = {}
        for pitch_type in pitch_usage_stats.keys():
            zone_command[pitch_type] = {
                'chase_low': 0.65,
                'chase_high': 0.70,
                'shadow_in_mid': 0.75
            }

        return {
            'pitch_usage_stats': pitch_usage_stats,
            'stuff_plus': stuff_plus,
            'sample_sizes': sample_sizes,
            'zone_command': zone_command
        }

    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return get_default_pitcher_stats()


def get_default_pitcher_stats() -> Dict:
    """기본 투수 통계"""
    return {
        'pitch_usage_stats': {
            'FF': 0.55,
            'SL': 0.28,
            'CU': 0.10,
            'CH': 0.07
        },
        'stuff_plus': {
            'FF': 108.0,
            'SL': 115.0,
            'CU': 105.0,
            'CH': 98.0
        },
        'sample_sizes': {
            'FF': 165,
            'SL': 84,
            'CU': 30,
            'CH': 21
        },
        'zone_command': {
            'FF': {'chase_low': 0.70, 'chase_high': 0.72},
            'SL': {'chase_low': 0.68, 'chase_out': 0.72},
            'CU': {'chase_low': 0.65},
            'CH': {'chase_low': 0.63}
        }
    }


# ============================================================================
# Sidebar - Control Tower
# ============================================================================
def render_sidebar():
    """사이드바 UI 렌더링 및 입력 받기"""
    st.sidebar.markdown("## ⚙️ Control Tower")
    st.sidebar.markdown("---")

    # 게임 상황
    st.sidebar.markdown("### 🏟️ Game State")
    inning = st.sidebar.slider("Inning", 1, 9, 9)
    score_diff = st.sidebar.slider("Score Difference", -5, 5, 1)
    outs = st.sidebar.selectbox("Outs", [0, 1, 2], index=2)

    # 주자 상황
    st.sidebar.markdown("#### Runners")
    col1, col2, col3 = st.sidebar.columns(3)
    runner_1st = col1.checkbox("1st", value=True)
    runner_2nd = col2.checkbox("2nd", value=True)
    runner_3rd = col3.checkbox("3rd", value=True)

    runners = [
        1 if runner_1st else 0,
        1 if runner_2nd else 0,
        1 if runner_3rd else 0
    ]

    st.sidebar.markdown("---")

    # 볼카운트
    st.sidebar.markdown("### ⚾ Count")
    col1, col2 = st.sidebar.columns(2)
    balls = col1.selectbox("Balls", [0, 1, 2, 3], index=3)
    strikes = col2.selectbox("Strikes", [0, 1, 2], index=2)
    count = f"{balls}-{strikes}"

    st.sidebar.markdown("---")

    # 투수 상태
    st.sidebar.markdown("### 🎯 Pitcher")
    pitch_count = st.sidebar.slider("Pitch Count", 0, 120, 98)
    pitcher_role = st.sidebar.selectbox("Role", ["SP", "RP"], index=0)
    pitcher_hand = st.sidebar.selectbox("Hand", ["R", "L"], index=0)
    prev_pitch = st.sidebar.selectbox(
        "Previous Pitch",
        ["FF", "SI", "FC", "SL", "CU", "CH", "KC", "ST"],
        index=0
    )
    prev_velo = st.sidebar.number_input("Previous Velocity (mph)", 85.0, 105.0, 97.0, 0.5)

    st.sidebar.markdown("---")

    # 매치업
    st.sidebar.markdown("### 🎯 Matchup")
    batter_hand = st.sidebar.selectbox("Batter Hand", ["R", "L"], index=1)
    chase_rate = st.sidebar.slider("Chase Rate", 0.0, 0.5, 0.32, 0.01)
    whiff_rate = st.sidebar.slider("Whiff Rate", 0.0, 0.5, 0.28, 0.01)
    iso = st.sidebar.slider("ISO (Power)", 0.0, 0.500, 0.350, 0.010)
    ops = st.sidebar.slider("OPS", 0.5, 1.5, 1.05, 0.01)

    st.sidebar.markdown("---")

    # 투수 선택
    st.sidebar.markdown("### 📊 Pitcher Selection")
    pitcher_presets = {
        "Walker Buehler (621111)": 621111,
        "Default Stats": None
    }
    pitcher_choice = st.sidebar.selectbox(
        "Select Pitcher",
        list(pitcher_presets.keys()),
        index=0
    )
    pitcher_id = pitcher_presets[pitcher_choice]

    # 시뮬레이션 버튼
    st.sidebar.markdown("---")
    simulate_button = st.sidebar.button(
        "🚀 Simulate Strategy",
        type="primary",
        use_container_width=True
    )

    # 상태 반환
    game_state = {
        'outs': outs,
        'count': count,
        'runners': runners,
        'score_diff': score_diff,
        'inning': inning
    }

    pitcher_state = {
        'hand': pitcher_hand,
        'role': pitcher_role,
        'pitch_count': pitch_count,
        'entropy': 0.62,  # 기본값
        'prev_pitch': prev_pitch,
        'prev_velo': prev_velo
    }

    matchup_state = {
        'batter_hand': batter_hand,
        'times_faced': 2,
        'chase_rate': chase_rate,
        'whiff_rate': whiff_rate,
        'iso': iso,
        'gb_fb_ratio': 0.8,
        'ops': ops
    }

    return (
        game_state,
        pitcher_state,
        matchup_state,
        pitcher_id,
        simulate_button
    )


# ============================================================================
# 3D Trajectory Visualization
# ============================================================================
def plot_3d_trajectory(
    prev_pitch_type: str,
    selected_pitch_type: str,
    prev_velo: float,
    selected_velo: float
) -> go.Figure:
    """
    3D 투구 궤적 시각화

    Args:
        prev_pitch_type: 직전 투구 타입
        selected_pitch_type: 선택된 투구 타입
        prev_velo: 직전 투구 속도
        selected_velo: 선택된 투구 속도

    Returns:
        Plotly Figure 객체
    """
    # 구종별 색상
    pitch_colors = {
        'FF': '#FF0000',  # Red
        'SI': '#FF8C00',  # Orange
        'FC': '#FFD700',  # Gold
        'SL': '#FFFF00',  # Yellow
        'CU': '#00FF00',  # Green
        'CH': '#00CED1',  # Cyan
        'KC': '#0000FF',  # Blue
        'ST': '#9370DB'   # Purple
    }

    # 투구 궤적 시뮬레이션 (간단한 포물선 모델)
    def simulate_trajectory(pitch_type: str, velo: float):
        """간단한 포물선 투구 궤적 생성"""
        t = np.linspace(0, 0.45, 100)  # 0 ~ 0.45초 (투구 시간)

        # 거리 (y축): 마운드(60.5ft)에서 홈플레이트(0ft)까지
        y = 60.5 - (velo * 1.467) * t  # mph to ft/s

        # 수평 이동 (x축): 구종에 따른 무브먼트
        movement_x = {
            'FF': 0.3, 'SI': -0.8, 'FC': 0.5, 'SL': 0.8,
            'CU': 0.2, 'CH': -0.4, 'KC': 0.6, 'ST': 1.0
        }
        x = movement_x.get(pitch_type, 0.0) * np.sin(t * 3)

        # 수직 이동 (z축): 드롭
        drop = {
            'FF': -1.2, 'SI': -1.8, 'FC': -1.5, 'SL': -1.0,
            'CU': -2.5, 'CH': -2.0, 'KC': -2.2, 'ST': -1.3
        }
        z = 6.0 + (drop.get(pitch_type, -1.5) * (t / 0.45))

        return x, y, z, t

    # 궤적 생성
    prev_x, prev_y, prev_z, prev_t = simulate_trajectory(prev_pitch_type, prev_velo)
    sel_x, sel_y, sel_z, sel_t = simulate_trajectory(selected_pitch_type, selected_velo)

    # 터널링 포인트 (0.167초)
    tunnel_idx = int(0.167 / 0.45 * 100)

    # Figure 생성
    fig = go.Figure()

    # 홈플레이트
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='white', symbol='square'),
        name='Home Plate',
        showlegend=True
    ))

    # 마운드
    fig.add_trace(go.Scatter3d(
        x=[0], y=[60.5], z=[0],
        mode='markers',
        marker=dict(size=8, color='brown', symbol='circle'),
        name='Mound',
        showlegend=True
    ))

    # 직전 투구 (회색 점선)
    fig.add_trace(go.Scatter3d(
        x=prev_x, y=prev_y, z=prev_z,
        mode='lines',
        line=dict(color='gray', width=4, dash='dash'),
        name=f'Previous: {prev_pitch_type}',
        showlegend=True
    ))

    # 선택된 투구 (색상 실선)
    fig.add_trace(go.Scatter3d(
        x=sel_x, y=sel_y, z=sel_z,
        mode='lines',
        line=dict(color=pitch_colors.get(selected_pitch_type, 'red'), width=6),
        name=f'Selected: {selected_pitch_type}',
        showlegend=True
    ))

    # 터널링 포인트 마커
    fig.add_trace(go.Scatter3d(
        x=[prev_x[tunnel_idx], sel_x[tunnel_idx]],
        y=[prev_y[tunnel_idx], sel_y[tunnel_idx]],
        z=[prev_z[tunnel_idx], sel_z[tunnel_idx]],
        mode='markers',
        marker=dict(size=8, color=['gray', pitch_colors.get(selected_pitch_type, 'red')]),
        name='Tunnel Point (0.167s)',
        showlegend=True
    ))

    # 스트라이크 존 (간단한 사각형)
    strike_zone_x = [-0.71, 0.71, 0.71, -0.71, -0.71]
    strike_zone_z = [1.5, 1.5, 3.5, 3.5, 1.5]
    strike_zone_y = [0, 0, 0, 0, 0]

    fig.add_trace(go.Scatter3d(
        x=strike_zone_x, y=strike_zone_y, z=strike_zone_z,
        mode='lines',
        line=dict(color='green', width=3),
        name='Strike Zone',
        showlegend=True
    ))

    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': '3D Pitch Trajectory Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        scene=dict(
            xaxis=dict(title='Horizontal (ft)', range=[-2, 2]),
            yaxis=dict(title='Distance (ft)', range=[0, 65]),
            zaxis=dict(title='Height (ft)', range=[0, 8]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=3, z=0.5)
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


# ============================================================================
# Main Layout
# ============================================================================
def main():
    """메인 대시보드 로직"""

    # 헤더
    st.markdown('<div class="main-header">⚾ Aegis Strategy Room - 3D Interactive Dashboard</div>', unsafe_allow_html=True)

    # 사이드바 렌더링 및 입력 받기
    (
        game_state,
        pitcher_state,
        matchup_state,
        pitcher_id,
        simulate_button
    ) = render_sidebar()

    # 시뮬레이션 실행
    if simulate_button:
        with st.spinner('🔄 Running AI Strategy Simulation...'):
            try:
                # 엔진 로드
                engine = load_strategy_engine()

                # 투수 데이터 로드
                if pitcher_id:
                    pitcher_stats = load_pitcher_data(pitcher_id, year=2024)
                else:
                    pitcher_stats = get_default_pitcher_stats()

                pitch_usage_stats = pitcher_stats['pitch_usage_stats']

                # 의사결정 실행
                result: DecisionResult = engine.decide_pitch(
                    game_state=game_state,
                    pitcher_state=pitcher_state,
                    matchup_state=matchup_state,
                    pitch_usage_stats=pitch_usage_stats,
                    pitcher_stats=pitcher_stats
                )

                # 세션에 저장
                st.session_state['result'] = result
                st.session_state['pitcher_state'] = pitcher_state
                st.session_state['game_state'] = game_state
                st.session_state['matchup_state'] = matchup_state

                st.success("✅ Simulation completed!")

            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 결과 표시
    if 'result' in st.session_state:
        result = st.session_state['result']
        pitcher_state = st.session_state['pitcher_state']
        game_state = st.session_state['game_state']
        matchup_state = st.session_state['matchup_state']

        # ====================================================================
        # Top Row: Situation & Recommendation
        # ====================================================================
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📋 Situation Summary")
            st.markdown(f"""
            <div class="metric-card">
            <b>🏟️ Game:</b> Inning {game_state['inning']}, {game_state['outs']} Outs<br>
            <b>⚾ Count:</b> {game_state['count']}<br>
            <b>🏃 Runners:</b> {game_state['runners']}<br>
            <b>📊 Score:</b> {'+' if game_state['score_diff'] > 0 else ''}{game_state['score_diff']}<br>
            <b>🎚️ Leverage:</b> <span style="color: red;">{result.leverage_level.upper()}</span><br>
            <b>🔢 Pitch Count:</b> {pitcher_state['pitch_count']}<br>
            <b>📈 Entropy:</b> {result.entropy_status}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎯 Batter Profile")
            st.markdown(f"""
            <div class="metric-card">
            <b>Hand:</b> {matchup_state['batter_hand']}<br>
            <b>Chase Rate:</b> {matchup_state['chase_rate']:.1%}<br>
            <b>Whiff Rate:</b> {matchup_state['whiff_rate']:.1%}<br>
            <b>ISO:</b> {matchup_state['iso']:.3f}<br>
            <b>OPS:</b> {matchup_state['ops']:.3f}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🤖 AI Recommendation")

            # 추천 구종 (큰 텍스트)
            pitch_type = result.selected_action.pitch_type
            zone = result.selected_action.zone

            st.markdown(f"""
            <div class="recommendation-box">
            {pitch_type} @ {zone}
            </div>
            """, unsafe_allow_html=True)

            # 위치 정보
            st.markdown(f"""
            <div class="metric-card">
            <b>📍 Location:</b> ({result.selected_action.plate_x:.2f}, {result.selected_action.plate_z:.2f})<br>
            <b>📊 Confidence:</b> {list(result.action_probs.values())[0]:.1%}
            </div>
            """, unsafe_allow_html=True)

            # Top 3 확률
            st.markdown("#### 📊 Top 3 Probabilities")
            top_3 = list(result.action_probs.items())[:3]

            for i, (action_key, prob) in enumerate(top_3, 1):
                st.progress(prob, text=f"{i}. {action_key}: {prob:.1%}")

        # ====================================================================
        # Rationale
        # ====================================================================
        st.markdown("---")
        st.info(f"**📝 Strategic Rationale:**\n\n{result.rationale}")

        # ====================================================================
        # Middle Row: 3D Trajectory Visualization
        # ====================================================================
        st.markdown("---")
        st.markdown("### 🎬 3D Pitch Trajectory")

        # 선택된 구종의 예상 속도 계산 (간단한 추정)
        pitch_velos = {
            'FF': 95.0, 'SI': 93.0, 'FC': 92.0, 'SL': 85.0,
            'CU': 78.0, 'CH': 84.0, 'KC': 80.0, 'ST': 82.0
        }
        selected_velo = pitch_velos.get(pitch_type, 90.0)

        fig = plot_3d_trajectory(
            prev_pitch_type=pitcher_state['prev_pitch'],
            selected_pitch_type=pitch_type,
            prev_velo=pitcher_state['prev_velo'],
            selected_velo=selected_velo
        )

        st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        # Bottom Row: Data Table
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📊 Expected Physics Metrics")

        # 물리 지표 데이터프레임 생성
        metrics_data = {
            'Metric': [
                'Pitch Type',
                'Zone',
                'Plate Location (X, Z)',
                'Estimated Velocity',
                'Tunnel Score',
                'EV Delta',
                'Command Risk',
                'Stuff Quality',
                'Chase Score',
                'Entropy Bonus',
                'Data Quality'
            ],
            'Value': [
                pitch_type,
                zone,
                f"({result.selected_action.plate_x:.2f}, {result.selected_action.plate_z:.2f})",
                f"{selected_velo:.1f} mph",
                f"{result.q_values.get(f'{pitch_type}_{zone}', 0.0):.3f}",
                "N/A",  # EV Delta는 result에 직접 없음
                "N/A",  # Command Risk
                "N/A",  # Stuff Quality
                "N/A",  # Chase Score
                "N/A",  # Entropy Bonus
                "N/A"   # Data Quality
            ]
        }

        df_metrics = pd.DataFrame(metrics_data)

        st.dataframe(
            df_metrics,
            use_container_width=True,
            hide_index=True
        )

        # 필터링 정보
        if result.noise_pitches:
            st.warning(f"⚠️ **Filtered Ghost Pitches:** {', '.join(result.noise_pitches)}")

        st.success(f"✅ **Available Pitches:** {', '.join(result.filtered_pitches.keys())}")

    else:
        # 초기 화면
        st.info("👈 **Setup game parameters in the sidebar and click 'Simulate Strategy' to begin.**")

        st.markdown("---")
        st.markdown("### 🎯 Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### ⚙️ Interactive Controls
            - Adjustable game state
            - Real-time pitcher selection
            - Custom matchup profiles
            """)

        with col2:
            st.markdown("""
            #### 🤖 AI Strategy
            - Multi-metric evaluation
            - Data noise filtering
            - Probabilistic selection
            """)

        with col3:
            st.markdown("""
            #### 📊 Visualization
            - 3D trajectory plotting
            - Tunneling analysis
            - Physics metrics table
            """)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
