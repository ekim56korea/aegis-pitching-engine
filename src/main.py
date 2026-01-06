"""
Aegis Pitching Engine - Entry Point (Main)
==========================================

프로젝트의 진입점(Entry Point)으로, 지금까지 만든 모든 모듈을 통합하여
**단일 타석 시뮬레이션(One At-Bat Simulation)**을 수행합니다.

Scenario: Walker Buehler vs. Shohei Ohtani (9회말 만루 위기 상황)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import traceback

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 임포트
from src.common.config import StrategyConfig
from src.data_pipeline.data_loader import AegisDataLoader
from src.game_theory.engine import AegisStrategyEngine, DecisionResult


# ============================================================================
# 로깅 설정
# ============================================================================
def setup_logging():
    """로깅 시스템 초기화 (INFO 레벨)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('aegis_simulation.log', mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🚀 Aegis Pitching Engine - Main Entry Point")
    logger.info("=" * 80)
    return logger


# ============================================================================
# 데이터 로딩 함수
# ============================================================================
def load_pitcher_stats(
    loader: AegisDataLoader,
    pitcher_id: int,
    year: int = 2024
) -> Dict:
    """
    투수의 실제 통계를 데이터베이스에서 로드

    Args:
        loader: AegisDataLoader 인스턴스
        pitcher_id: 투수 ID (예: 621111 = Walker Buehler)
        year: 조회할 시즌 연도

    Returns:
        pitcher_stats: 투수 통계 딕셔너리
            - pitch_usage_stats: 구종별 구사율
            - stuff_plus: 구종별 Stuff+ 점수
            - sample_sizes: 구종별 샘플 수
            - zone_command: 존별 제구 성공률
    """
    logger = logging.getLogger(__name__)
    logger.info(f"📊 투수 데이터 로딩: ID={pitcher_id}, Year={year}")

    try:
        # 투수의 전체 투구 데이터 로드
        df = loader.load_pitcher_data(pitcher_id)

        if df.empty:
            logger.warning(f"⚠️  투수 {pitcher_id}의 데이터가 없습니다. 기본값 사용.")
            return get_default_pitcher_stats()

        # 연도 필터링 (컬럼이 있는 경우)
        if 'game_year' in df.columns:
            df = df[df['game_year'] == year]
            if df.empty:
                logger.warning(f"⚠️  {year}년 데이터가 없습니다. 전체 연도 사용.")
                df = loader.load_pitcher_data(pitcher_id)

        # 1. 구종별 구사율 계산
        pitch_usage_stats = {}
        if 'pitch_type' in df.columns:
            pitch_counts = df['pitch_type'].value_counts()
            total_pitches = len(df)
            pitch_usage_stats = {
                pitch: count / total_pitches
                for pitch, count in pitch_counts.items()
            }
            logger.info(f"✅ 구종 분포: {pitch_usage_stats}")

        # 2. 구종별 샘플 수
        sample_sizes = dict(pitch_counts) if 'pitch_type' in df.columns else {}

        # 3. Stuff+ 추정 (release_speed 기반 휴리스틱)
        stuff_plus = {}
        if 'pitch_type' in df.columns and 'release_speed' in df.columns:
            for pitch_type in pitch_usage_stats.keys():
                pitch_df = df[df['pitch_type'] == pitch_type]
                avg_velo = pitch_df['release_speed'].mean()

                # 간단한 Stuff+ 추정 (실제로는 더 복잡한 모델 필요)
                # 평균 속도 대비 점수 (90mph = 100, +1mph = +2점)
                stuff_plus[pitch_type] = 100 + (avg_velo - 90.0) * 2.0

        # 4. 존별 제구 성공률 (간단한 버전)
        zone_command = {}
        for pitch_type in pitch_usage_stats.keys():
            zone_command[pitch_type] = {
                'chase_low': 0.65,
                'chase_high': 0.70,
                'shadow_in_mid': 0.75
            }

        pitcher_stats = {
            'pitch_usage_stats': pitch_usage_stats,
            'stuff_plus': stuff_plus,
            'sample_sizes': sample_sizes,
            'zone_command': zone_command
        }

        logger.info(f"✅ 투수 통계 로딩 완료: {len(pitch_usage_stats)}개 구종")
        return pitcher_stats

    except Exception as e:
        logger.error(f"❌ 투수 데이터 로딩 실패: {e}")
        logger.error(traceback.format_exc())
        return get_default_pitcher_stats()


def get_default_pitcher_stats() -> Dict:
    """데이터 로딩 실패 시 사용할 기본 투수 통계"""
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
            'FF': {'chase_low': 0.70, 'chase_high': 0.72, 'shadow_in_mid': 0.75},
            'SL': {'chase_low': 0.68, 'chase_out': 0.72, 'shadow_out_mid': 0.70},
            'CU': {'chase_low': 0.65, 'shadow_in_low': 0.68},
            'CH': {'chase_low': 0.63, 'chase_out': 0.65}
        }
    }


def create_ohtani_matchup() -> Dict:
    """
    Shohei Ohtani의 매치업 데이터 생성 (가상의 위협적인 타자)

    Returns:
        matchup_state: 타자 매치업 정보
    """
    return {
        'batter_hand': 'L',       # 좌타자
        'times_faced': 2,         # 이번 게임에서 2번째 대면
        'chase_rate': 0.32,       # Chase Rate 32% (높은 선구안)
        'whiff_rate': 0.28,       # Whiff Rate 28% (강한 컨택 능력)
        'iso': 0.350,             # ISO .350 (매우 위험한 장타력)
        'gb_fb_ratio': 0.8,       # GB/FB 0.8 (플라이볼 히터)
        'ops': 1.050,             # OPS 1.050 (슈퍼스타급)
        'prev_result': 'whiff'    # 직전 타석은 헛스윙 (심리적 요인)
    }


# ============================================================================
# 상황 출력 함수
# ============================================================================
def print_situation_report(
    game_state: Dict,
    pitcher_state: Dict,
    matchup_state: Dict,
    pitcher_name: str = "Walker Buehler",
    batter_name: str = "Shohei Ohtani"
):
    """현재 상황을 보고서 형식으로 출력"""
    print("\n" + "=" * 80)
    print("📋 SITUATION REPORT - The War Room")
    print("=" * 80)
    print(f"\n🏟️  Scenario: {pitcher_name} vs. {batter_name}")
    print(f"   Inning: Bottom 9th")
    print(f"   Outs: {game_state['outs']}")
    print(f"   Count: {game_state['count']}")
    print(f"   Runners: Bases Loaded (1st, 2nd, 3rd)")
    print(f"   Score: Leading by {game_state['score_diff']} run(s)")
    print(f"   Leverage: 🔴 CRITICAL - High Leverage Situation")

    print(f"\n⚾ Pitcher Status:")
    print(f"   Hand: {pitcher_state['hand']}")
    print(f"   Pitch Count: {pitcher_state['pitch_count']} (Fatigue Critical)")
    print(f"   Entropy: {pitcher_state['entropy']:.2f}")
    print(f"   Previous Pitch: {pitcher_state['prev_pitch']} @ {pitcher_state['prev_velo']:.1f} mph")

    print(f"\n🎯 Batter Profile:")
    print(f"   Hand: {matchup_state['batter_hand']}")
    print(f"   Chase Rate: {matchup_state['chase_rate']:.1%}")
    print(f"   Whiff Rate: {matchup_state['whiff_rate']:.1%}")
    print(f"   ISO: {matchup_state['iso']:.3f} (⚠️  HIGH POWER)")
    print(f"   OPS: {matchup_state['ops']:.3f}")
    print(f"   GB/FB: {matchup_state['gb_fb_ratio']:.2f}")

    print("\n" + "=" * 80)


def print_ai_recommendation(result: DecisionResult):
    """AI 추천 결과를 출력"""
    print("\n" + "=" * 80)
    print("🤖 AI RECOMMENDATION")
    print("=" * 80)

    action = result.selected_action
    print(f"\n✅ Recommended Pitch:")
    print(f"   Type: {action.pitch_type}")
    print(f"   Zone: {action.zone}")
    print(f"   Location: ({action.plate_x:.2f}, {action.plate_z:.2f})")

    print(f"\n📊 Top 3 Action Probabilities:")
    for i, (action_key, prob) in enumerate(list(result.action_probs.items())[:3], 1):
        print(f"   {i}. {action_key}: {prob:.1%}")

    print(f"\n🎚️  Decision Context:")
    print(f"   Leverage: {result.leverage_level}")
    print(f"   Entropy Status: {result.entropy_status}")

    print(f"\n🔍 Noise Filtering:")
    print(f"   Filtered Pitches: {list(result.filtered_pitches.keys())}")
    if result.noise_pitches:
        print(f"   Removed (Noise): {result.noise_pitches}")
    else:
        print(f"   Removed (Noise): None")

    print("\n" + "=" * 80)


def print_strategic_rationale(rationale: str):
    """전략적 근거를 출력"""
    print("\n" + "=" * 80)
    print("📝 STRATEGIC RATIONALE")
    print("=" * 80)
    print(f"\n{rationale}")
    print("\n" + "=" * 80)


def save_visualization_placeholder():
    """
    물리 엔진 시각화 플레이스홀더
    (실제 구현 시 TunnelingAnalyzer와 matplotlib 사용)
    """
    logger = logging.getLogger(__name__)
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 사용
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5, 0.5,
            'Trajectory Visualization\n(Placeholder)\n\n'
            'Full implementation requires:\n'
            '- TunnelingAnalyzer.simulate_trajectory()\n'
            '- Physics engine integration\n'
            '- 3D trajectory plotting',
            ha='center', va='center',
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        output_path = project_root / 'simulation_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ 시각화 저장됨: {output_path}")
        print(f"\n💾 Physics Visualization: {output_path}")

    except ImportError:
        logger.warning("⚠️  matplotlib이 설치되지 않아 시각화를 생략합니다.")
        print("\n⚠️  Visualization skipped (matplotlib not installed)")
    except Exception as e:
        logger.error(f"❌ 시각화 생성 실패: {e}")


# ============================================================================
# 메인 실행 함수
# ============================================================================
def main():
    """
    메인 실행 흐름:
    1. Setup & Config
    2. Scenario Definition
    3. Data Loading
    4. Engine Execution
    5. Results Display
    """
    logger = setup_logging()

    try:
        # ====================================================================
        # Step 1: Setup & Config
        # ====================================================================
        logger.info("Step 1: Setup & Configuration")
        config = StrategyConfig()
        logger.info("✅ StrategyConfig 로딩 완료")

        # ====================================================================
        # Step 2: Scenario Definition - "The War Room"
        # ====================================================================
        logger.info("\nStep 2: Scenario Definition")

        # 게임 상황 (9회말 만루 위기)
        game_state = {
            'outs': 2,              # 2아웃
            'count': '3-2',         # 풀카운트
            'runners': [1, 1, 1],   # 만루
            'score_diff': 1,        # 1점 리드 (High Leverage)
            'inning': 9             # 9회말
        }

        # 투수 상태 (Walker Buehler)
        pitcher_state = {
            'hand': 'R',            # 우투수
            'role': 'SP',           # 선발 투수
            'pitch_count': 98,      # 98개 투구 (Fatigue Critical)
            'entropy': 0.62,        # 중간 엔트로피
            'prev_pitch': 'FF',     # 직전 투구: 패스트볼
            'prev_velo': 97.0       # 97mph 하이 패스트볼
        }

        # 매치업 상태 (Shohei Ohtani)
        matchup_state = create_ohtani_matchup()

        # 상황 출력
        print_situation_report(
            game_state, pitcher_state, matchup_state,
            pitcher_name="Walker Buehler",
            batter_name="Shohei Ohtani"
        )

        # ====================================================================
        # Step 3: Data Loading
        # ====================================================================
        logger.info("\nStep 3: Loading Pitcher Data from Database")

        pitcher_id = 621111  # Walker Buehler
        pitcher_stats = None

        try:
            with AegisDataLoader() as loader:
                # 스키마 검증
                if not loader.check_schema():
                    logger.warning("⚠️  스키마 검증 실패. 기본값 사용.")
                    pitcher_stats = get_default_pitcher_stats()
                else:
                    # 실제 데이터 로드
                    pitcher_stats = load_pitcher_stats(loader, pitcher_id, year=2024)

        except FileNotFoundError:
            logger.warning("⚠️  DuckDB 파일을 찾을 수 없습니다. 기본값 사용.")
            pitcher_stats = get_default_pitcher_stats()

        except Exception as e:
            logger.error(f"❌ 데이터 로딩 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            pitcher_stats = get_default_pitcher_stats()

        # pitcher_stats 구조 확인
        pitch_usage_stats = pitcher_stats['pitch_usage_stats']

        logger.info(f"✅ 투수 데이터 준비 완료: {len(pitch_usage_stats)}개 구종")

        # ====================================================================
        # Step 4: Engine Execution
        # ====================================================================
        logger.info("\nStep 4: Executing AegisStrategyEngine")

        # 엔진 초기화
        engine = AegisStrategyEngine(device='cpu')
        logger.info("✅ AegisStrategyEngine 초기화 완료")

        # 의사결정 실행 (모든 서브 모듈이 작동)
        result = engine.decide_pitch(
            game_state=game_state,
            pitcher_state=pitcher_state,
            matchup_state=matchup_state,
            pitch_usage_stats=pitch_usage_stats,
            pitcher_stats=pitcher_stats
        )

        logger.info("✅ 의사결정 완료")

        # ====================================================================
        # Step 5: Results Display
        # ====================================================================
        logger.info("\nStep 5: Displaying Results")

        # AI 추천 출력
        print_ai_recommendation(result)

        # 전략적 근거 출력
        print_strategic_rationale(result.rationale)

        # 시각화 저장
        save_visualization_placeholder()

        # ====================================================================
        # Final Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("✅ ONE AT-BAT SIMULATION COMPLETED")
        print("=" * 80)
        print(f"\n🎯 Final Decision: {result.selected_action.pitch_type} @ {result.selected_action.zone}")
        print(f"📊 Confidence: {list(result.action_probs.values())[0]:.1%}")
        print(f"🔧 Leverage: {result.leverage_level}")
        print(f"📝 Log: aegis_simulation.log")
        print("\n" + "=" * 80)

        logger.info("=" * 80)
        logger.info("✅ Simulation Completed Successfully")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ CRITICAL ERROR IN SIMULATION")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

        print("\n" + "=" * 80)
        print("❌ SIMULATION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("See aegis_simulation.log for details.")
        print("=" * 80)

        sys.exit(1)


if __name__ == "__main__":
    main()
