import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def clean_text(text: str) -> str:
    """
    HTML 태그를 제거하고 불필요한 공백/줄바꿈을 정리합니다.
    """
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    t = soup.get_text(separator=" ")
    return " ".join(t.split()).lower()

def preprocess(records: list[dict]) -> pd.DataFrame:
    """
    수집된 레코드를 통합 스키마에 맞게 전처리합니다.
    - HTML 태그 제거 및 텍스트 정리
    - name + url 기준 중복 제거
    - 필수 필드 (name, description, url) 누락 체크
    """
    df = pd.DataFrame(records)

    # 스키마 통일 및 필요한 필드 확인
    required_fields = ["name", "description", "url", "category", "tags", "source", "collected_at"]
    for field in required_fields:
        if field not in df.columns:
            df[field] = None # 없는 필드는 None으로 채움

    # 텍스트 클리닝
    df["description"] = df["description"].apply(clean_text)

    # 중복 제거 (name + url 기준)
    df.drop_duplicates(subset=["name", "url"], inplace=True)

    # 필수 필드 누락 행 제거
    df = df.dropna(subset=["name", "description", "url"])

    # collected_at 필드가 datetime 객체가 아니면 문자열로 변환 (FAISS 저장 시 직렬화 문제 방지)
    # 또는 저장 시점에 문자열로 변환하는 것을 고려할 수 있습니다.
    # 여기서는 JSON 직렬화에 문제가 없도록 문자열을 그대로 사용합니다.

    return df

if __name__ == "__main__":
    # 테스트를 위한 더미 데이터
    dummy_records = [
        {
            "name": "Test Tool 1",
            "description": "<p>A tool for <b>testing</b> purposes. <br/>It's great!</p>",
            "url": "http://test.com/tool1",
            "category": "Testing",
            "tags": ["test", "dev"],
            "source": "Dummy",
            "collected_at": "2024-05-20T11:00:00Z"
        },
        {
            "name": "Test Tool 2",
            "description": "Another tool for tests. <p>Very useful.</p>",
            "url": "http://test.com/tool2",
            "category": "Utilities",
            "tags": ["utility"],
            "source": "Dummy",
            "collected_at": "2024-05-20T11:05:00Z"
        },
        {
            "name": "Test Tool 1", # Duplicate name+url
            "description": "Duplicate entry for testing.",
            "url": "http://test.com/tool1",
            "category": "Testing",
            "tags": ["duplicate"],
            "source": "Dummy",
            "collected_at": "2024-05-20T11:10:00Z"
        },
        {
            "name": "Missing Desc Tool",
            "description": None, # Missing description
            "url": "http://test.com/missing",
            "category": "Missing",
            "tags": [],
            "source": "Dummy",
            "collected_at": "2024-05-20T11:15:00Z"
        }
    ]

    processed_df = preprocess(dummy_records)
    print("--- 전처리된 데이터 ---")
    print(processed_df.to_dict(orient="records")) 