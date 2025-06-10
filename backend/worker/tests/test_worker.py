import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import os
import asyncio
import yaml

# Ensure these imports are correct based on your project structure
from backend.worker.worker import scrape_site, main

@pytest.mark.asyncio
@patch('playwright.async_api.async_playwright')
@patch('os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
@patch('os.path.join', side_effect=os.path.join) # Keep original join behavior
async def test_scrape_site_success(mock_join, mock_open, mock_makedirs, mock_async_playwright):
    # Setup mocks for Playwright
    mock_p = AsyncMock()
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    mock_async_playwright.return_value.__aenter__.return_value = mock_p
    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_page.return_value = mock_page
    mock_page.goto.return_value = None
    mock_page.content.return_value = "<html><body>Test Content</body></html>"

    config = {"site_name": "Test Site", "url": "http://test.com"}

    asyncio.get_event_loop().time = MagicMock(return_value=123456789.0) # Mock time for filename

    await scrape_site(config)

    mock_async_playwright.assert_called_once()
    mock_p.chromium.launch.assert_called_once()
    mock_browser.new_page.assert_called_once()
    mock_page.goto.assert_called_once_with(config['url'], timeout=60000)
    mock_page.wait_for_load_state.assert_called_once_with("networkidle")
    mock_page.content.assert_called_once()
    
    mock_makedirs.assert_called_once_with("raw_html_storage", exist_ok=True)
    mock_open.assert_called_once()
    mock_open.return_value.__enter__.return_value.write.assert_called_once_with("<html><body>Test Content</body></html>")
    mock_browser.close.assert_called_once()

@pytest.mark.asyncio
@patch('playwright.async_api.async_playwright')
@patch('builtins.print')
async def test_scrape_site_error_handling(mock_print, mock_async_playwright):
    mock_p = AsyncMock()
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    mock_async_playwright.return_value.__aenter__.return_value = mock_p
    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_page.return_value = mock_page
    mock_page.goto.side_effect = Exception("Network error") # Simulate error

    config = {"site_name": "Error Site", "url": "http://error.com"}

    await scrape_site(config)

    mock_browser.close.assert_called_once()
    mock_print.assert_any_call('스크래핑 중 오류 발생 Error Site: Network error')

@pytest.mark.asyncio
@patch('os.path.exists')
@patch('builtins.open', new_callable=MagicMock)
@patch('yaml.safe_load')
@patch('asyncio.gather', new_callable=AsyncMock) # Mock asyncio.gather
@patch('backend.worker.worker.scrape_site', new_callable=AsyncMock) # Mock scrape_site
async def test_main_success(mock_scrape_site, mock_gather, mock_yaml_load, mock_open, mock_exists):
    mock_exists.return_value = True
    mock_yaml_load.return_value = [
        {"site_name": "Site1", "url": "http://site1.com"},
        {"site_name": "Site2", "url": "http://site2.com"}
    ]

    await main()

    mock_open.assert_called_once()
    mock_yaml_load.assert_called_once()
    assert mock_scrape_site.call_count == 2 # Called for each config
    # Ensure gather was called with the correct tasks
    mock_gather.assert_called_once_with(
        mock_scrape_site.return_value, mock_scrape_site.return_value
    )

@pytest.mark.asyncio
@patch('os.path.exists')
@patch('builtins.print')
async def test_main_no_config_file(mock_print, mock_exists):
    mock_exists.return_value = False
    await main()
    mock_print.assert_any_call('오류: scraper_config.yaml 파일을 찾을 수 없습니다: ' + os.path.join(os.path.dirname(__file__), 'scraper_config.yaml')) 