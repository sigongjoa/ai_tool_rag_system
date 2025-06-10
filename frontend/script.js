document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchForm = document.getElementById('searchForm');
    const resultsContainer = document.getElementById('results');
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    const API_BASE_URL = document.querySelector('meta[name="api-base-url"]').getAttribute('content');
    const API_SEARCH_URL = `${API_BASE_URL}/api/v1/search`;
    const API_PROCESS_URL = `${API_BASE_URL}/api/v1/process_url`;

    let currentVersion = 'ai'; // 기본 버전

    const updatePlaceholder = () => {
        const activeTab = document.querySelector('.tab-button.active');
        let placeholderText = 'AI 도구 검색...';
        if (activeTab) {
            const versionKey = activeTab.dataset.version;
            switch (versionKey) {
                case 'ai':
                    placeholderText = 'AI 전용 도구 검색...';
                    break;
                case 'dev':
                    placeholderText = '개발자용 도구 검색...';
                    break;
                case 'office':
                    placeholderText = '일반 사무용 도구 검색...';
                    break;
                case 'url_add':
                    placeholderText = 'URL 추가 기능';
                    break;
            }
        }
        searchInput.placeholder = placeholderText;
    };

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            currentVersion = button.dataset.version;
            updatePlaceholder();

            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`${currentVersion}_content`).classList.add('active');

            if (currentVersion === 'ai' || currentVersion === 'dev' || currentVersion === 'office') {
                searchForm.style.display = 'flex';
            } else {
                searchForm.style.display = 'none';
            }
            resultsContainer.innerHTML = '';
        });
    });

    document.getElementById('ai_content').classList.add('active');
    searchForm.style.display = 'flex';

    updatePlaceholder();

    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        performSearch();
    });

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            resultsContainer.innerHTML = '<p>검색어를 입력해주세요.</p>';
            return;
        }

        resultsContainer.innerHTML = '<p>검색 중입니다...</p>';

        try {
            const response = await fetch(API_SEARCH_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    version: currentVersion
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP 오류! 상태: ${response.status}`);
            }

            const data = await response.json();

            resultsContainer.innerHTML = '';

            if (data.data && data.data.content) {
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'summary-content';
                summaryDiv.innerHTML = `<h2>요약 답변:</h2><p>${data.data.content}</p>`;
                resultsContainer.appendChild(summaryDiv);
            }

            if (data.retrieved_sources && data.retrieved_sources.length > 0) {
                const sourcesHeader = document.createElement('h2');
                sourcesHeader.textContent = '관련 도구';
                resultsContainer.appendChild(sourcesHeader);

                data.retrieved_sources.forEach(source => {
                    const toolDiv = document.createElement('div');
                    toolDiv.className = 'tool-item';
                    toolDiv.innerHTML = `
                        <h3>${source.name || '제목 없음'}</h3>
                        <p><strong>URL:</strong> <a href="${source.url}" target="_blank">${source.url}</a></p>
                        <p>${source.description || '설명 없음'}</p>
                    `;
                    resultsContainer.appendChild(toolDiv);
                });
            } else {
                resultsContainer.innerHTML = '<p>검색된 AI 도구가 없습니다.</p>';
            }

        } catch (error) {
            console.error('검색 중 오류 발생:', error);
            resultsContainer.innerHTML = `<p>오류 발생: ${error.message}. 백엔드 서버가 실행 중인지 확인해주세요.</p>`;
        }
    }

    const urlInputProcess = document.getElementById('urlInputProcess');
    const processUrlButtonProcess = document.getElementById('processUrlButtonProcess');
    const urlProcessResultDisplay = document.getElementById('urlProcessResultDisplay');

    processUrlButtonProcess.addEventListener('click', async () => {
        const url = urlInputProcess.value.trim();
        if (!url) {
            urlProcessResultDisplay.textContent = '처리할 URL을 입력해주세요.';
            urlProcessResultDisplay.style.color = 'orange';
            return;
        }

        urlProcessResultDisplay.textContent = 'URL 처리 중... 잠시 기다려 주세요.';
        urlProcessResultDisplay.style.color = 'blue';

        try {
            const response = await fetch(API_PROCESS_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url }),
            });

            const data = await response.json();

            if (response.ok) {
                urlProcessResultDisplay.textContent = `성공: ${data.message || 'URL이 성공적으로 처리되었습니다.'}`;
                urlProcessResultDisplay.style.color = 'green';
                urlInputProcess.value = '';
            } else {
                urlProcessResultDisplay.textContent = `오류: ${data.detail || 'URL 처리 중 알 수 없는 오류 발생'}`;
                urlProcessResultDisplay.style.color = 'red';
            }
        } catch (error) {
            console.error('URL process fetch error:', error);
            urlProcessResultDisplay.textContent = '네트워크 오류 또는 백엔드에 연결할 수 없습니다.';
            urlProcessResultDisplay.style.color = 'red';
        }
    });
}); 