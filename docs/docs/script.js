document.addEventListener('DOMContentLoaded', () => {
    console.debug('DOM fully loaded and parsed');

    // const popularToolsSection = document.querySelector('.popular-tools');
    // if (popularToolsSection) {
    //     console.debug('Popular tools section found, will be hidden initially.');
    //     popularToolsSection.style.display = 'none'; // Hide popular tools section initially
    // }

    let allAIData = []; // To store original AI tools data for filtering

    // Fetch data from individual JSON files
    Promise.all([
        fetch('data/ai_tools.json').then(response => response.json()),
        fetch('data/dev_websites.json').then(response => response.json()),
        fetch('data/productivity_websites.json').then(response => response.json())
    ])
        .then(([aiToolsData, devWebsitesData, productivityWebsitesData]) => {
            console.debug('Data loaded successfully from individual files.');
            const data = {
                ai_tools: aiToolsData,
                dev_websites: devWebsitesData,
                productivity_websites: productivityWebsitesData
            };
            allAIData = data.ai_tools; // Store original AI tools data

            const populateTable = (tableId, items, headers) => {
                console.debug(`Populating table: ${tableId}`);
                const tableHead = document.querySelector(`#${tableId} thead tr`);
                const tbody = document.querySelector(`#${tableId} tbody`);
                
                if (!tableHead || !tbody) {
                    console.error(`Table head or body not found for ID: ${tableId}`);
                    return;
                }

                // Clear existing headers and rows
                tableHead.innerHTML = '';
                tbody.innerHTML = '';

                // Populate headers
                headers.forEach(header => {
                    console.debug(`Adding header to ${tableId}: ${header}`);
                    const th = document.createElement('th');
                    th.textContent = header;
                    tableHead.appendChild(th);
                });

                // Populate rows
                items.forEach(item => {
                    console.debug(`Adding item to ${tableId}:`, item);
                    const row = tbody.insertRow();
                    // Add click event listener to the row
                    row.style.cursor = 'pointer'; // Indicate that the row is clickable
                    row.addEventListener('click', () => {
                        console.debug(`Row clicked for item: ${item.name}`);
                        if (item.website) {
                            window.open(item.website, '_blank');
                            console.debug(`Opening website: ${item.website}`);
                        } else {
                            console.warn(`No website URL found for item: ${item.name}`);
                        }
                    });
                    headers.forEach(headerText => {
                        const cell = row.insertCell();
                        const key = {
                            '이름': 'name',
                            '설명': 'description',
                            '카테고리': 'category',
                            '가격': 'pricing',
                            '지원 언어': 'languages_supported',
                            '최종 업데이트': 'last_updated',
                            '태그': 'tags',
                            '플랫폼': 'platforms'
                        }[headerText];

                        if (key && item[key]) {
                            if (Array.isArray(item[key])) {
                                cell.textContent = item[key].join(', ');
                            } else {
                                cell.textContent = item[key];
                            }
                        } else {
                            cell.textContent = 'N/A';
                        }

                        // Add category tag style if it's the category cell
                        if (headerText === '카테고리') {
                            const categorySpan = document.createElement('span');
                            categorySpan.className = 'category-tag';
                            categorySpan.textContent = item.category;
                            cell.innerHTML = ''; // Clear previous textContent
                            cell.appendChild(categorySpan);
                        }
                    });
                });
                console.debug(`Finished populating table: ${tableId}`);
            };

            // Define headers for each table based on the schema
            const aiToolsHeaders = ['이름', '설명', '카테고리', '가격'];
            const devWebsitesHeaders = ['이름', '설명', '카테고리', '태그'];
            const productivityWebsitesHeaders = ['이름', '설명', '카테고리', '플랫폼'];

            populateTable('ai-tools-table', data.ai_tools, aiToolsHeaders);
            populateTable('dev-websites-table', data.dev_websites, devWebsitesHeaders);
            populateTable('productivity-websites-table', data.productivity_websites, productivityWebsitesHeaders);

            // Tab navigation logic
            const navItems = document.querySelectorAll('.nav-item');
            const tabContents = document.querySelectorAll('.tab-content');

            const showTab = (targetId) => {
                console.debug(`Showing tab: ${targetId}`);
                tabContents.forEach(content => {
                    console.debug(`Hiding content: ${content.id}`);
                    content.classList.remove('active');
                    content.style.display = 'none';
                });
                navItems.forEach(item => {
                    console.debug(`Removing active class from nav item: ${item.dataset.target}`);
                    item.classList.remove('active');
                });

                const activeContent = document.getElementById(targetId);
                const activeNavItem = document.querySelector(`.nav-item[data-target="${targetId}"]`);

                if (activeContent) {
                    console.debug(`Showing active content: ${activeContent.id}`);
                    activeContent.classList.add('active');
                    activeContent.style.display = 'block';
                }
                if (activeNavItem) {
                    console.debug(`Adding active class to nav item: ${activeNavItem.dataset.target}`);
                    activeNavItem.classList.add('active');
                }
                console.debug(`Finished showing tab: ${targetId}`);
            };

            navItems.forEach(item => {
                console.debug(`Adding click listener to nav item: ${item.dataset.target}`);
                item.addEventListener('click', (e) => {
                    console.debug(`Nav item clicked: ${e.target.dataset.target}`);
                    e.preventDefault();
                    showTab(e.target.dataset.target);
                });
            });

            // Show the page intro tab by default
            showTab('page-intro');

            // Search functionality
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                console.debug('Search input found, adding event listener.');
                searchInput.addEventListener('input', (e) => {
                    const searchTerm = e.target.value.toLowerCase();
                    console.debug(`Search term: ${searchTerm}`);

                    const filteredAIData = allAIData.filter(tool => {
                        return tool.name.toLowerCase().includes(searchTerm) ||
                               tool.description.toLowerCase().includes(searchTerm) ||
                               tool.category.toLowerCase().includes(searchTerm) ||
                               (Array.isArray(tool.tags) && tool.tags.some(tag => tag.toLowerCase().includes(searchTerm)));
                    });
                    console.debug('Filtered AI data:', filteredAIData);

                    // Show AI Tools tab and populate with filtered data
                    showTab('ai-tools');
                    populateTable('ai-tools-table', filteredAIData, aiToolsHeaders);
                });
            } else {
                console.error('Search input not found.');
            }

            console.debug('Script execution finished.');
        })
        .catch(error => {
            console.error('Error loading data:', error);
            // logger.debug(`Error loading data: ${error.message}`); // Removed logger.debug
        });
}); 