"""
Savant DuckDB 데이터베이스 탐색 스크립트
DuckDB 파일의 테이블 구조와 데이터를 조회합니다.
"""

import duckdb
from pathlib import Path
from typing import List, Tuple


def explore_database(db_path: Path) -> None:
    """
    DuckDB 데이터베이스를 탐색하고 구조 정보를 출력합니다.

    Args:
        db_path: DuckDB 파일 경로
    """
    try:
        # 데이터베이스 파일 존재 확인
        if not db_path.exists():
            print(f"❌ 오류: 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
            return

        print(f"{'='*80}")
        print(f"📊 DuckDB 데이터베이스 탐색")
        print(f"{'='*80}")
        print(f"파일 경로: {db_path}")
        print(f"파일 크기: {db_path.stat().st_size / (1024*1024):.2f} MB\n")

        # 데이터베이스 연결
        conn = duckdb.connect(str(db_path), read_only=True)

        # 모든 테이블 목록 조회
        tables_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """
        tables = conn.execute(tables_query).fetchall()

        if not tables:
            print("⚠️  데이터베이스에 테이블이 없습니다.")
            conn.close()
            return

        print(f"{'='*80}")
        print(f"📋 테이블 목록 (총 {len(tables)}개)")
        print(f"{'='*80}")
        for idx, (table_name,) in enumerate(tables, 1):
            print(f"  {idx}. {table_name}")
        print()

        # 각 테이블에 대한 상세 정보 출력
        for table_name, in tables:
            print(f"\n{'='*80}")
            print(f"📊 테이블: {table_name}")
            print(f"{'='*80}\n")

            # 행 개수 조회
            row_count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = conn.execute(row_count_query).fetchone()[0]
            print(f"📈 전체 행 개수: {row_count:,}\n")

            # 컬럼 정보 조회 (상위 5개)
            columns_query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                LIMIT 5
            """
            columns = conn.execute(columns_query).fetchall()

            print("🔍 컬럼 정보 (상위 5개):")
            print(f"{'─'*80}")
            print(f"{'컬럼명':<40} {'데이터 타입':<30}")
            print(f"{'─'*80}")

            for col_name, data_type in columns:
                print(f"{col_name:<40} {data_type:<30}")

            # 전체 컬럼 개수 확인
            total_columns_query = f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
            """
            total_columns = conn.execute(total_columns_query).fetchone()[0]

            if total_columns > 5:
                print(f"{'─'*80}")
                print(f"... 외 {total_columns - 5}개 컬럼 더 있음 (총 {total_columns}개)")

        # 연결 종료
        conn.close()

        print(f"\n{'='*80}")
        print("✅ 탐색 완료!")
        print(f"{'='*80}\n")

    except duckdb.Error as e:
        print(f"❌ DuckDB 오류 발생: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 디렉토리 기준으로 데이터베이스 경로 설정
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "01_raw" / "savant.duckdb"

    explore_database(db_path)


if __name__ == "__main__":
    main()
