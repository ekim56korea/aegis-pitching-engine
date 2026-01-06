"""
AegisDataLoader: Baseball Savant 데이터 로딩 및 관리 클래스
DuckDB를 사용한 효율적인 데이터 액세스 제공
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List
import warnings

from src.common.config import DB_PATH, REQUIRED_COLUMNS


class AegisDataLoader:
    """
    Baseball Savant 데이터를 DuckDB에서 로드하는 클래스

    Features:
        - 투수별 데이터 조회
        - 연도별 데이터 샘플링
        - 스키마 검증
        - Read-only 연결로 데이터 안정성 보장
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        AegisDataLoader 초기화

        Args:
            db_path: DuckDB 파일 경로 (기본값: config.DB_PATH)
        """
        self.db_path = db_path or DB_PATH

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DuckDB 파일을 찾을 수 없습니다: {self.db_path}"
            )

        # Read-only 연결
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        print(f"✅ DuckDB 연결 성공: {self.db_path}")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()

    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            print("🔒 DuckDB 연결 종료")

    def check_schema(self) -> bool:
        """
        pitches 테이블의 스키마를 검증하고 REQUIRED_COLUMNS 존재 여부 확인

        Returns:
            bool: 모든 필수 컬럼이 존재하면 True, 아니면 False
        """
        try:
            # 테이블 존재 확인
            tables_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'pitches'
            """
            tables = self.conn.execute(tables_query).fetchall()

            if not tables:
                warnings.warn("⚠️  'pitches' 테이블이 존재하지 않습니다.")
                return False

            # 테이블의 모든 컬럼 조회
            columns_query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'pitches'
            """
            existing_columns = [
                row[0] for row in self.conn.execute(columns_query).fetchall()
            ]

            # 누락된 컬럼 확인
            missing_columns = [
                col for col in REQUIRED_COLUMNS if col not in existing_columns
            ]

            if missing_columns:
                warnings.warn(
                    f"⚠️  다음 필수 컬럼이 'pitches' 테이블에 없습니다:\n"
                    f"   {', '.join(missing_columns)}"
                )
                return False

            print(f"✅ 스키마 검증 완료: 모든 필수 컬럼({len(REQUIRED_COLUMNS)}개) 존재")
            return True

        except Exception as e:
            warnings.warn(f"⚠️  스키마 검증 중 오류 발생: {e}")
            return False

    def load_pitcher_data(self, pitcher_id: int) -> pd.DataFrame:
        """
        특정 투수의 모든 투구 데이터를 로드

        Args:
            pitcher_id: 투수 ID

        Returns:
            pd.DataFrame: 투수의 투구 데이터
        """
        # REQUIRED_COLUMNS를 쿼리에 사용
        columns_str = ", ".join(REQUIRED_COLUMNS)

        query = f"""
            SELECT {columns_str}
            FROM pitches
            WHERE pitcher = ?
        """

        try:
            # numpy 타입을 Python 기본 타입으로 변환
            pitcher_id = int(pitcher_id)
            df = self.conn.execute(query, [pitcher_id]).df()
            print(f"📊 투수 {pitcher_id}: {len(df):,}개 투구 데이터 로드")
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def load_data_by_year(
        self,
        year: int,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        특정 연도의 데이터를 샘플링하여 로드

        Args:
            year: 조회할 연도
            limit: 반환할 최대 행 수

        Returns:
            pd.DataFrame: 샘플링된 투구 데이터
        """
        columns_str = ", ".join(REQUIRED_COLUMNS)

        query = f"""
            SELECT {columns_str}
            FROM pitches
            WHERE game_year = ?
            LIMIT ?
        """

        try:
            # numpy 타입을 Python 기본 타입으로 변환
            year = int(year)
            limit = int(limit)
            df = self.conn.execute(query, [year, limit]).df()
            print(f"📊 {year}년 데이터: {len(df):,}개 투구 샘플 로드")
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def get_table_info(self) -> dict:
        """
        데이터베이스 테이블 정보 조회

        Returns:
            dict: 테이블 정보 (테이블명, 행 개수 등)
        """
        try:
            # 테이블 목록
            tables_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
            """
            tables = [row[0] for row in self.conn.execute(tables_query).fetchall()]

            info = {"tables": {}}

            for table in tables:
                row_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                info["tables"][table] = {"row_count": row_count}

            return info

        except Exception as e:
            print(f"❌ 테이블 정보 조회 실패: {e}")
            return {}


def main():
    """사용 예시"""
    print("=" * 80)
    print("🚀 AegisDataLoader 테스트")
    print("=" * 80 + "\n")

    try:
        # Context manager 사용
        with AegisDataLoader() as loader:
            # 1. 테이블 정보 확인
            print("📋 데이터베이스 정보:")
            info = loader.get_table_info()
            for table_name, table_info in info.get("tables", {}).items():
                print(f"   - {table_name}: {table_info['row_count']:,} rows")
            print()

            # 2. 스키마 검증
            print("🔍 스키마 검증:")
            loader.check_schema()
            print()

            # 3. 2024년 데이터 샘플 로드
            print("📊 2024년 데이터 샘플 (5개):")
            df_2024 = loader.load_data_by_year(year=2024, limit=5)

            if not df_2024.empty:
                print(df_2024.to_string())
                print(f"\n✅ 컬럼 개수: {len(df_2024.columns)}")
                print(f"✅ 행 개수: {len(df_2024)}")
            else:
                print("⚠️  2024년 데이터가 없습니다.")
            print()

            # 4. 특정 투수 데이터 로드 (첫 번째 투수 ID 사용)
            if not df_2024.empty and 'pitcher' in df_2024.columns:
                pitcher_id = df_2024['pitcher'].iloc[0]
                print(f"📊 투수 {pitcher_id} 데이터 샘플 (5개):")
                df_pitcher = loader.load_pitcher_data(pitcher_id)

                if not df_pitcher.empty:
                    print(df_pitcher.head().to_string())
                    print(f"\n✅ 해당 투수의 총 투구 수: {len(df_pitcher):,}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Tip: DuckDB 파일에 'pitches' 테이블을 먼저 생성해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

    print("\n" + "=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
