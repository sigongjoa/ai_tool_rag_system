import asyncio
import yaml
from playwright.async_api import async_playwright
import os

async def scrape_site(config):
    print(f"스크래핑 시작: {config['site_name']} ({config['url']})")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(config['url'], timeout=60000)
            # 페이지 로딩 대기 (필요에 따라 더 구체적인 대기 조건 추가 가능)
            await page.wait_for_load_state("networkidle")

            # HTML 콘텐츠 추출
            # 여기서는 전체 페이지의 HTML을 가져오지만,
            # config['target_selector']를 사용하여 특정 요소만 추출하도록 확장 가능
            html_content = await page.content()

            # 원본 HTML 저장 디렉토리 확인 및 생성
            raw_html_dir = "raw_html_storage"
            os.makedirs(raw_html_dir, exist_ok=True)

            # 파일명은 사이트 이름과 현재 시간으로 지정
            filename = f"{config['site_name'].replace(' ', '_')}_{asyncio.get_event_loop().time()}.html"
            filepath = os.path.join(raw_html_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML 저장 완료: {filepath}")

        except Exception as e:
            print(f"스크래핑 중 오류 발생 {config['site_name']}: {e}")
        finally:
            await browser.close()

async def main():
    config_path = os.path.join(os.path.dirname(__file__), 'scraper_config.yaml')
    if not os.path.exists(config_path):
        print(f"오류: scraper_config.yaml 파일을 찾을 수 없습니다: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)

    tasks = [scrape_site(config) for config in configs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main()) 