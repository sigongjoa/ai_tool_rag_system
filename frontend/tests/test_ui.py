import pytest
from playwright.sync_api import Page, expect

# 프론트엔드 애플리케이션이 실행되는 URL을 여기에 지정합니다.
# 개발 서버 (예: http-server, Live Server 등)를 사용하여 프론트엔드를 실행해야 합니다.
BASE_URL = "http://localhost:8000" # 실제 포트에 맞게 변경해주세요

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    # 이 fixture는 브라우저 컨텍스트에 추가 인수를 제공하는 데 사용됩니다.
    # 예를 들어, 네트워크 모킹 또는 스토리지 상태를 처리할 수 있습니다.
    return {
        **browser_context_args,
        "base_url": BASE_URL,
    }

def test_homepage_loads(page: Page):
    """홈페이지가 올바르게 로드되고 제목이 예상과 일치하는지 테스트합니다."""
    page.goto("/")
    # index.html의 <title> 태그 내용을 확인합니다.
    # 프로젝트에 따라 이 제목을 실제 제목에 맞게 변경해주세요.
    expect(page).to_have_title("AI Tools RAG System") 
    print("INFO: Homepage loaded successfully and title checked.")

def test_search_functionality(page: Page):
    """검색 기능이 올바르게 작동하는지 테스트합니다."""
    page.goto("/")

    # 검색 입력란과 검색 버튼의 ID를 가정합니다.
    # 실제 프론트엔드 HTML에서 사용되는 ID에 맞게 조정해야 합니다.
    search_input_selector = "#search-input" # 예: <input id="search-input" ...>
    search_button_selector = "#search-button" # 예: <button id="search-button" ...>
    search_results_selector = "#search-results" # 예: <div id="search-results">...</div>

    print(f"INFO: Entering 'image generation' into {search_input_selector}")
    search_input = page.locator(search_input_selector)
    search_input.fill("image generation")

    print(f"INFO: Clicking {search_button_selector}")
    search_button = page.locator(search_button_selector)
    search_button.click()

    # 검색 결과가 나타날 때까지 기다리고 내용을 검증합니다.
    # 여기서는 최소한 하나 이상의 결과가 나타나는지, 그리고 특정 텍스트가 포함되는지 확인합니다.
    # 실제 예상되는 검색 결과에 따라 내용을 변경해주세요.
    print(f"INFO: Waiting for {search_results_selector} to contain text.")
    search_results = page.locator(search_results_selector)
    expect(search_results).not_to_be_empty()
    # 예시: 검색 결과에 'Midjourney'라는 텍스트가 포함되어 있는지 확인
    expect(search_results).to_contain_text("Midjourney") 
    print("INFO: Search functionality tested successfully.")

def test_filter_functionality(page: Page):
    """필터 기능이 올바르게 작동하는지 테스트합니다."""
    page.goto("/")

    # 필터 UI 요소의 ID를 가정합니다. 실제 HTML에 맞게 조정해야 합니다.
    # 이 예시는 'AI & ML' 카테고리 체크박스를 가정합니다.
    filter_checkbox_selector = "#category-ai-ml" # 예: <input type="checkbox" id="category-ai-ml" ...>
    search_results_selector = "#search-results"

    print(f"INFO: Checking filter {filter_checkbox_selector}")
    ai_category_checkbox = page.locator(filter_checkbox_selector)
    if ai_category_checkbox.is_enabled(): # 체크박스가 활성화되어 있다면
        ai_category_checkbox.check()

    # 필터 적용 후 결과가 업데이트될 때까지 기다립니다.
    # (대부분의 SPA는 필터 적용 시 자동으로 재렌더링되므로 명시적인 버튼 클릭이 필요 없을 수 있습니다.)

    # 필터링된 결과가 나타날 때까지 기다리고 내용을 검증합니다.
    print(f"INFO: Waiting for {search_results_selector} to reflect filter changes.")
    search_results = page.locator(search_results_selector)
    expect(search_results).not_to_be_empty()
    # 예시: 'Hugging Face'와 같이 필터링된 카테고리에 속하는 도구가 포함되어 있는지 확인
    expect(search_results).to_contain_text("Hugging Face")
    # 예시: 'VS Code'와 같이 필터링된 카테고리에 속하지 않는 도구가 포함되어 있지 않은지 확인
    expect(search_results).not_to_contain_text("VS Code")
    print("INFO: Filter functionality tested successfully.")

# ==================================================================================
# 테스트 실행 방법:
# 1. 프론트엔드 애플리케이션을 실행합니다 (예: `cd frontend` 후 `npx http-server` 또는 Live Server 확장 사용).
#    프론트엔드가 실행되는 URL이 `BASE_URL` 변수에 올바르게 설정되어 있는지 확인하세요.
# 2. `pytest-playwright` 설치:
#    `pip install pytest-playwright`
# 3. Playwright 브라우저 설치 (최초 1회 실행):
#    `playwright install`
# 4. 프로젝트 루트 디렉토리에서 테스트 실행:
#    `pytest frontend/tests/test_ui.py`
#    또는 `pytest frontend/tests/`
# 5. `print` 문은 테스트 실행 시 터미널에 정보를 출력하여 디버깅에 도움을 줍니다.
# ================================================================================== 