// Enhanced FOS Performance Dashboard JavaScript
class EnhancedPerformanceDashboard {
    constructor() {
        this.data = null;
        this.charts = {};
        this.filteredData = null;
        this.historicalReports = [];
        this.isLoadingHistorical = false;
        this.filters = {
            search: '',
            grade: '',
            device: '',
            dateRange: 'all',
            performanceRange: 'all'
        };
        this.refreshInterval = null;
        this.animationQueue = [];

        // Chart configurations
        this.chartColors = {
            primary: '#2563eb',
            secondary: '#64748b',
            success: '#10b981',
            warning: '#f59e0b',
            error: '#ef4444',
            desktop: '#4285F4',
            mobile: '#EA4335',
            lcp: '#FF6B6B',
            fcp: '#4ECDC4',
            cls: '#45B7D1',
            inp: '#96CEB4',
            gradients: {
                primary: 'linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)',
                success: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                warning: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                error: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
            }
        };

        this.init();
    }

    async init() {
        try {
            this.showLoadingState();
            await this.loadData();
            await this.loadHistoricalReports();
            this.renderDashboard();
            this.initializeCharts();
            this.setupEventListeners();
            this.setupRealtimeUpdates();
            this.animateElements();
            this.hideLoadingState();
            console.log('✅ Enhanced Dashboard initialized successfully');
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to load dashboard data: ' + error.message);
        }
    }

    async loadData() {
        // Use embedded data if available
        if (window.DASHBOARD_DATA) {
            this.data = window.DASHBOARD_DATA;
            this.filteredData = this.data;
            console.log('✅ Loaded embedded dashboard data');
            return;
        }

        // Fallback to fetch from API
        try {
            const response = await fetch('data/dashboard_data.json');
            if (!response.ok) throw new Error('Failed to fetch dashboard data');
            this.data = await response.json();
            this.filteredData = this.data;
            console.log('✅ Loaded dashboard data from API');
        } catch (error) {
            console.warn('⚠️ Using fallback data due to fetch error:', error);
            this.data = this.getFallbackData();
            this.filteredData = this.data;
        }
    }

    async loadHistoricalReports() {
        this.isLoadingHistorical = true;
        this.showLoadingIndicator('Loading historical reports...');

        try {
            // Use embedded historical reports data
            if (this.data && this.data.historical_reports) {
                this.historicalReports = this.data.historical_reports
                    .filter(report => report.has_reports)
                    .sort((a, b) => new Date(b.date) - new Date(a.date));

                console.log(`✅ Loaded ${this.historicalReports.length} historical reports`);
            } else {
                console.warn('⚠️ No historical reports data found');
                this.historicalReports = [];
            }
        } catch (error) {
            console.error('❌ Error loading historical reports:', error);
            this.historicalReports = [];
        } finally {
            this.isLoadingHistorical = false;
            this.hideLoadingIndicator();
        }
    }

    renderDashboard() {
        this.updateSummaryStats();
        this.renderSitesTable();
        this.renderRecentTests();
        this.renderInsights();
        this.renderHistoricalReports();
        this.updateLastUpdatedTime();
    }

    updateSummaryStats() {
        const summary = this.data.summary || {};

        // Update stat values with animation
        this.animateCounter('total-sites', summary.total_sites || 0);
        this.animateCounter('total-tests', summary.total_tests || 0);
        this.animateCounter('avg-performance', Math.round(summary.avg_performance_score || 0));
        this.animateCounter('avg-cwv', Math.round(summary.avg_cwv_score || 0));

        // Update trend indicators
        this.updateTrendIndicator('performance-trend', summary.avg_performance_score);
        this.updateTrendIndicator('cwv-trend', summary.avg_cwv_score);

        // Add performance insights to stats
        this.updatePerformanceInsights(summary);
    }

    animateCounter(elementId, targetValue, duration = 1000) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const startValue = 0;
        const increment = targetValue / (duration / 16);
        let currentValue = startValue;

        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= targetValue) {
                currentValue = targetValue;
                clearInterval(timer);
            }
            element.textContent = Math.round(currentValue);
        }, 16);
    }

    updateTrendIndicator(elementId, score) {
        const element = document.getElementById(elementId);
        if (!element) return;

        let trendClass = 'neutral';
        let trendIcon = '📊';
        let trendText = 'Stable';
        let trendDescription = '';

        if (score >= 90) {
            trendClass = 'positive';
            trendIcon = '🟢';
            trendText = 'Excellent';
            trendDescription = 'Outstanding performance';
        } else if (score >= 80) {
            trendClass = 'positive';
            trendIcon = '🟢';
            trendText = 'Very Good';
            trendDescription = 'Above average performance';
        } else if (score >= 70) {
            trendClass = 'positive';
            trendIcon = '🟡';
            trendText = 'Good';
            trendDescription = 'Satisfactory performance';
        } else if (score >= 60) {
            trendClass = 'warning';
            trendIcon = '🟡';
            trendText = 'Fair';
            trendDescription = 'Room for improvement';
        } else if (score >= 40) {
            trendClass = 'warning';
            trendIcon = '🟠';
            trendText = 'Needs Work';
            trendDescription = 'Requires optimization';
        } else {
            trendClass = 'negative';
            trendIcon = '🔴';
            trendText = 'Poor';
            trendDescription = 'Critical performance issues';
        }

        element.className = `stat-change ${trendClass}`;
        element.innerHTML = `<span>${trendIcon}</span> ${trendText}`;
        element.title = trendDescription;
    }

    updatePerformanceInsights(summary) {
        // Add dynamic insights based on current data
        const insights = [];

        if (summary.avg_performance_score < 60) {
            insights.push({
                type: 'error',
                title: 'Critical Performance Issues',
                description: `Average performance score is ${Math.round(summary.avg_performance_score)}%, well below recommended thresholds.`,
                action: 'Immediate optimization required for Core Web Vitals and loading performance.'
            });
        }

        if (summary.avg_cwv_score > 80) {
            insights.push({
                type: 'success',
                title: 'Excellent Core Web Vitals',
                description: `Core Web Vitals score of ${Math.round(summary.avg_cwv_score)}% indicates great user experience.`,
                action: 'Maintain current optimization strategies and monitor for regressions.'
            });
        }

        // Store dynamic insights for rendering
        this.dynamicInsights = insights;
    }

    renderSitesTable() {
        const tbody = document.getElementById('sites-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';
        const sites = this.applyFilters(this.data.sites || []);

        if (sites.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="10" style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    <div style="font-size: 1.2rem; margin-bottom: 10px;">📊</div>
                    <div>No sites match the current filters</div>
                    <button onclick="dashboard.clearFilters()" style="margin-top: 10px; padding: 8px 16px; background: var(--primary-color); color: white; border: none; border-radius: 6px; cursor: pointer;">Clear Filters</button>
                </td>
            `;
            tbody.appendChild(row);
            return;
        }

        sites.forEach((site, index) => {
            const row = document.createElement('tr');
            row.style.animationDelay = `${index * 0.1}s`;
            row.className = 'table-row-animate';

            row.innerHTML = `
                <td>
                    <div class="site-info">
                        <div class="site-name">${this.extractDomainName(site.name || site.url)}</div>
                        <div class="site-url">${this.truncateUrl(site.url || '', 40)}</div>
                    </div>
                </td>
                <td><span class="grade-badge grade-${site.latest_grade || 'N-A'}" title="Core Web Vitals Grade">${site.latest_grade || 'N/A'}</span></td>
                <td><span class="score-value" title="Core Web Vitals Score">${Math.round(site.latest_score || 0)}</span></td>
                <td><span class="score-value" title="Lighthouse Performance Score">${Math.round(site.performance_score || 0)}</span></td>
                <td><span class="metric-value" title="Largest Contentful Paint">${this.formatMetric(site.lcp, 's')}</span></td>
                <td><span class="metric-value" title="First Contentful Paint">${this.formatMetric(site.fcp, 's')}</span></td>
                <td><span class="metric-value" title="Cumulative Layout Shift">${this.formatMetric(site.cls, '')}</span></td>
                <td><span class="metric-value" title="Interaction to Next Paint">${this.formatMetric(site.inp, 'ms')}</span></td>
                <td><span class="test-count" title="Total Tests Performed">${site.total_tests || 0}</span></td>
                <td>
                    <span class="trend-indicator ${this.getTrendClass(site.trend)}" title="Performance Trend">
                        ${this.getTrendIcon(site.trend)} ${this.formatTrend(site.trend)}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        });

        // Add table interaction events
        this.setupTableInteractions();
    }

    setupTableInteractions() {
        const rows = document.querySelectorAll('#sites-table-body tr');
        rows.forEach(row => {
            row.addEventListener('click', (e) => {
                if (e.target.closest('.grade-badge') || e.target.closest('.trend-indicator')) {
                    this.showSiteDetails(row);
                }
            });
        });
    }

    showSiteDetails(row) {
        const siteName = row.querySelector('.site-name')?.textContent;
        if (!siteName) return;

        const siteData = this.data.sites?.find(s =>
            this.extractDomainName(s.name || s.url) === siteName
        );

        if (siteData) {
            this.displaySiteModal(siteData);
        }
    }

    displaySiteModal(siteData) {
        const modal = document.createElement('div');
        modal.className = 'site-modal-overlay';
        modal.innerHTML = `
            <div class="site-modal">
                <div class="modal-header">
                    <h3>${this.extractDomainName(siteData.name || siteData.url)}</h3>
                    <button class="modal-close" onclick="this.closest('.site-modal-overlay').remove()">×</button>
                </div>
                <div class="modal-content">
                    <div class="modal-section">
                        <h4>Performance Overview</h4>
                        <div class="metric-grid">
                            <div class="metric-item">
                                <span class="metric-label">Performance Score</span>
                                <span class="metric-value">${Math.round(siteData.performance_score || 0)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">CWV Score</span>
                                <span class="metric-value">${Math.round(siteData.latest_score || 0)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Total Tests</span>
                                <span class="metric-value">${siteData.total_tests || 0}</span>
                            </div>
                        </div>
                    </div>
                    <div class="modal-section">
                        <h4>Core Web Vitals</h4>
                        <div class="cwv-details">
                            <div class="cwv-metric">
                                <span class="cwv-label">LCP</span>
                                <span class="cwv-value">${this.formatMetric(siteData.lcp, 's')}</span>
                                <span class="cwv-status ${this.getCWVStatus(siteData.lcp, 'lcp')}">${this.getCWVStatus(siteData.lcp, 'lcp')}</span>
                            </div>
                            <div class="cwv-metric">
                                <span class="cwv-label">FCP</span>
                                <span class="cwv-value">${this.formatMetric(siteData.fcp, 's')}</span>
                                <span class="cwv-status ${this.getCWVStatus(siteData.fcp, 'fcp')}">${this.getCWVStatus(siteData.fcp, 'fcp')}</span>
                            </div>
                            <div class="cwv-metric">
                                <span class="cwv-label">CLS</span>
                                <span class="cwv-value">${this.formatMetric(siteData.cls, '')}</span>
                                <span class="cwv-status ${this.getCWVStatus(siteData.cls, 'cls')}">${this.getCWVStatus(siteData.cls, 'cls')}</span>
                            </div>
                            <div class="cwv-metric">
                                <span class="cwv-label">INP</span>
                                <span class="cwv-value">${this.formatMetric(siteData.inp, 'ms')}</span>
                                <span class="cwv-status ${this.getCWVStatus(siteData.inp, 'inp')}">${this.getCWVStatus(siteData.inp, 'inp')}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal styles
        if (!document.getElementById('modal-styles')) {
            const styles = document.createElement('style');
            styles.id = 'modal-styles';
            styles.textContent = `
                .site-modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                    backdrop-filter: blur(5px);
                    animation: fadeIn 0.3s ease;
                }
                .site-modal {
                    background: white;
                    border-radius: 16px;
                    max-width: 600px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                    animation: scaleIn 0.3s ease;
                }
                .modal-header {
                    padding: 20px 25px;
                    border-bottom: 1px solid var(--border-color);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                }
                .modal-close {
                    background: none;
                    border: none;
                    font-size: 1.5rem;
                    cursor: pointer;
                    color: var(--text-secondary);
                    transition: color 0.2s ease;
                }
                .modal-close:hover { color: var(--error-color); }
                .modal-content { padding: 25px; }
                .modal-section { margin-bottom: 25px; }
                .modal-section h4 { margin-bottom: 15px; font-weight: 600; }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                }
                .metric-item {
                    text-align: center;
                    padding: 15px;
                    background: #f8fafc;
                    border-radius: 8px;
                }
                .metric-label {
                    display: block;
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                    margin-bottom: 5px;
                }
                .cwv-details { display: flex; flex-direction: column; gap: 12px; }
                .cwv-metric {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px;
                    background: #f8fafc;
                    border-radius: 8px;
                }
                .cwv-status.good { color: var(--success-color); font-weight: 600; }
                .cwv-status.needs-improvement { color: var(--warning-color); font-weight: 600; }
                .cwv-status.poor { color: var(--error-color); font-weight: 600; }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(modal);

        // Close modal on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    getCWVStatus(value, metric) {
        if (value === 'N/A' || value === null || value === undefined) return 'N/A';

        const numValue = parseFloat(value);
        const thresholds = {
            lcp: { good: 2.5, poor: 4.0 },
            fcp: { good: 1.8, poor: 3.0 },
            cls: { good: 0.1, poor: 0.25 },
            inp: { good: 200, poor: 500 }
        };

        const threshold = thresholds[metric];
        if (!threshold) return 'N/A';

        if (numValue <= threshold.good) return 'Good';
        if (numValue <= threshold.poor) return 'Needs Improvement';
        return 'Poor';
    }

    renderRecentTests() {
        const container = document.getElementById('recent-tests');
        if (!container) return;

        container.innerHTML = '';
        const recentTests = this.data.recent_tests || [];

        if (recentTests.length === 0) {
            container.innerHTML = `
                <div class="no-data">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🧪</div>
                    <div>No recent tests available</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">Run some performance tests to see results here</div>
                </div>
            `;
            return;
        }

        const testsList = document.createElement('div');
        testsList.className = 'recent-tests-list';

        recentTests.slice(0, 8).forEach((test, index) => {
            const testElement = document.createElement('div');
            testElement.className = 'recent-test-item';
            testElement.style.animationDelay = `${index * 0.1}s`;
            testElement.innerHTML = `
                <div class="test-info">
                    <div class="test-site">${this.extractDomainName(test.name)}</div>
                    <div class="test-meta">
                        <span class="device-mode">${test.device_mode || 'desktop'}</span>
                        <span class="test-date">${this.formatDate(test.date)}</span>
                        <span class="test-time">${this.getTimeAgo(test.date)}</span>
                    </div>
                </div>
                <div class="test-results">
                    <span class="grade-badge grade-${test.cwv_grade || 'N-A'}" title="Core Web Vitals Grade">${test.cwv_grade || 'N/A'}</span>
                    <div class="performance-score" title="Performance Score">${Math.round(test.performance_score || 0)}%</div>
                </div>
            `;
            testsList.appendChild(testElement);
        });

        container.appendChild(testsList);
    }

    renderInsights() {
        const container = document.getElementById('insights-container');
        if (!container) return;

        container.innerHTML = '';

        // Combine static and dynamic insights
        const staticInsights = this.data.performance_insights || [];
        const allInsights = [...staticInsights, ...(this.dynamicInsights || [])];

        if (allInsights.length === 0) {
            container.innerHTML = `
                <div class="no-insights">
                    <div style="font-size: 2rem; margin-bottom: 10px;">💡</div>
                    <div>No performance insights available</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">Insights will appear as more data is collected</div>
                </div>
            `;
            return;
        }

        allInsights.forEach((insight, index) => {
            const insightElement = document.createElement('div');
            insightElement.className = `insight insight-${insight.type}`;
            insightElement.style.animationDelay = `${index * 0.2}s`;
            insightElement.innerHTML = `
                <div class="insight-header">
                    <span class="insight-icon">${this.getInsightIcon(insight.type)}</span>
                    <h4 class="insight-title">${insight.title}</h4>
                </div>
                <p class="insight-description">${insight.description}</p>
                <div class="insight-action">${insight.action}</div>
            `;
            container.appendChild(insightElement);
        });
    }

    renderHistoricalReports() {
        const container = document.getElementById('historical-reports');
        if (!container) return;

        if (this.historicalReports.length === 0) {
            container.innerHTML = `
                <div class="no-historical-data">
                    <div class="no-data-icon">📊</div>
                    <h3>No Historical Reports Found</h3>
                    <p>Historical performance reports will appear here once performance tests are completed with proper data saving.</p>
                    <p>Recent tests may not have generated the required JSON data files.</p>
                    <button class="refresh-btn" onclick="dashboard.loadHistoricalReports()">
                        🔄 Refresh Historical Reports
                    </button>
                </div>
            `;
            return;
        }

        // Create historical reports grid
        const reportsGrid = document.createElement('div');
        reportsGrid.className = 'historical-reports-grid';

        this.historicalReports.slice(0, 12).forEach((report, index) => {
            const reportCard = document.createElement('div');
            reportCard.className = 'historical-report-card';
            reportCard.style.animationDelay = `${index * 0.1}s`;
            reportCard.innerHTML = `
                <div class="report-header">
                    <div class="report-date">
                        <span class="date-main">${this.formatDate(report.date)}</span>
                        <span class="date-sub">${this.getTimeAgo(report.date)}</span>
                    </div>
                    <div class="report-status ${report.has_reports ? 'active' : 'inactive'}" title="${report.has_reports ? 'Reports Available' : 'No Reports'}">
                        ${report.has_reports ? '✅' : '❌'}
                    </div>
                </div>
                <div class="report-stats">
                    <div class="stat-row">
                        <span class="stat-label">Sites Tested</span>
                        <span class="stat-value">${report.sites || 0}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Tests</span>
                        <span class="stat-value">${report.total_tests || 0}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Performance</span>
                        <span class="stat-value performance-score">${Math.round(report.avg_performance || 0)}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg CWV Score</span>
                        <span class="stat-value cwv-score">${Math.round(report.avg_cwv || 0)}%</span>
                    </div>
                </div>
                <div class="report-actions">
                    <button class="view-report-btn" onclick="dashboard.viewReportDetails('${report.directory}')" ${!report.has_reports ? 'disabled' : ''}>
                        📋 View Details
                    </button>
                </div>
            `;
            reportsGrid.appendChild(reportCard);
        });

        container.innerHTML = '';
        container.appendChild(reportsGrid);
    }

    initializeCharts() {
        // Check if Chart.js is available
        if (typeof Chart !== 'undefined') {
            this.initPerformanceChart();
            this.initCWVChart();
            this.initGradeChart();
        } else {
            console.warn('Chart.js not available - charts will show static images');
            this.setupStaticCharts();
        }
    }

    setupStaticCharts() {
        // Ensure chart images are properly displayed
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            const img = container.querySelector('img');
            if (img) {
                img.addEventListener('error', () => {
                    img.style.display = 'none';
                    container.innerHTML = `
                        <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                            <div style="font-size: 3rem; margin-bottom: 15px;">📈</div>
                            <div>Chart not available</div>
                            <div style="font-size: 0.8rem; margin-top: 5px;">Generate reports to see performance charts</div>
                        </div>
                    `;
                });
            }
        });
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('site-search');
        if (searchInput) {
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.filters.search = e.target.value.toLowerCase();
                    this.renderSitesTable();
                }, 300);
            });
        }

        // Filter controls
        const gradeFilter = document.getElementById('grade-filter');
        if (gradeFilter) {
            gradeFilter.addEventListener('change', (e) => {
                this.filters.grade = e.target.value;
                this.renderSitesTable();
            });
        }

        const deviceFilter = document.getElementById('device-filter');
        if (deviceFilter) {
            deviceFilter.addEventListener('change', (e) => {
                this.filters.device = e.target.value;
                this.renderSitesTable();
            });
        }

        // Performance range filter
        const performanceFilter = document.getElementById('performance-filter');
        if (performanceFilter) {
            performanceFilter.addEventListener('change', (e) => {
                this.filters.performanceRange = e.target.value;
                this.renderSitesTable();
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'k':
                        e.preventDefault();
                        searchInput?.focus();
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshDashboard();
                        break;
                }
            }
        });

        // Theme toggle (if implemented)
        this.setupThemeToggle();
    }

    setupThemeToggle() {
        // Add theme toggle functionality
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('dark-theme');
                localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
            });

            // Load saved theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {
                document.body.classList.add('dark-theme');
            }
        }
    }

    setupRealtimeUpdates() {
        // Set up periodic data refresh
        this.refreshInterval = setInterval(() => {
            this.softRefreshData();
        }, 30000); // Refresh every 30 seconds

        // Listen for visibility changes to pause/resume updates
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                clearInterval(this.refreshInterval);
            } else {
                this.setupRealtimeUpdates();
            }
        });
    }

    async softRefreshData() {
        try {
            const response = await fetch('data/dashboard_data.json');
            if (response.ok) {
                const newData = await response.json();

                // Check if data has changed
                if (JSON.stringify(newData) !== JSON.stringify(this.data)) {
                    this.data = newData;
                    this.updateSummaryStats();
                    this.showUpdateNotification();
                }
            }
        } catch (error) {
            console.warn('Failed to refresh data:', error);
        }
    }

    showUpdateNotification() {
        const notification = document.createElement('div');
        notification.className = 'update-notification';
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px; background: var(--success-color); color: white; padding: 12px 20px; border-radius: 8px; margin: 10px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
                <span>🔄</span>
                <span>Dashboard updated with latest data</span>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    animateElements() {
        // Animate elements on load
        const elements = document.querySelectorAll('.stat-card, .card, .recent-test-item, .insight');
        elements.forEach((element, index) => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(20px)';

            setTimeout(() => {
                element.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    applyFilters(sites) {
        return sites.filter(site => {
            // Search filter
            if (this.filters.search) {
                const searchTerm = this.filters.search.toLowerCase();
                const siteName = (site.name || '').toLowerCase();
                const siteUrl = (site.url || '').toLowerCase();
                if (!siteName.includes(searchTerm) && !siteUrl.includes(searchTerm)) {
                    return false;
                }
            }

            // Grade filter
            if (this.filters.grade && this.filters.grade !== 'all') {
                if ((site.latest_grade || 'N/A') !== this.filters.grade) {
                    return false;
                }
            }

            // Performance range filter
            if (this.filters.performanceRange && this.filters.performanceRange !== 'all') {
                const score = site.performance_score || 0;
                switch (this.filters.performanceRange) {
                    case 'excellent':
                        if (score < 90) return false;
                        break;
                    case 'good':
                        if (score < 70 || score >= 90) return false;
                        break;
                    case 'fair':
                        if (score < 50 || score >= 70) return false;
                        break;
                    case 'poor':
                        if (score >= 50) return false;
                        break;
                }
            }

            return true;
        });
    }

    clearFilters() {
        this.filters = {
            search: '',
            grade: '',
            device: '',
            dateRange: 'all',
            performanceRange: 'all'
        };

        // Reset form controls
        const searchInput = document.getElementById('site-search');
        if (searchInput) searchInput.value = '';

        const filterSelects = document.querySelectorAll('.filter-select');
        filterSelects.forEach(select => select.value = 'all');

        this.renderSitesTable();
    }

    async refreshDashboard() {
        this.showLoadingIndicator('Refreshing dashboard...');

        try {
            await this.loadData();
            await this.loadHistoricalReports();
            this.renderDashboard();
            this.showSuccessMessage('Dashboard refreshed successfully');
        } catch (error) {
            this.showError('Failed to refresh dashboard: ' + error.message);
        } finally {
            this.hideLoadingIndicator();
        }
    }

    updateLastUpdatedTime() {
        const element = document.getElementById('last-updated');
        if (element) {
            element.textContent = new Date().toLocaleString();
        }
    }

    // Utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    }

    extractDomainName(url) {
        if (!url) return 'Unknown';
        try {
            return new URL(url).hostname.replace('www.', '');
        } catch {
            return url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0];
        }
    }

    truncateUrl(url, maxLength) {
        if (!url || url.length <= maxLength) return url;
        return url.substring(0, maxLength) + '...';
    }

    formatMetric(value, unit) {
        if (value === null || value === undefined || isNaN(value)) return 'N/A';

        if (unit === 's') {
            return value < 1 ? `${Math.round(value * 1000)}ms` : `${value.toFixed(2)}s`;
        } else if (unit === 'ms') {
            return `${Math.round(value)}ms`;
        } else {
            return parseFloat(value).toFixed(3);
        }
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            return new Date(dateString).toLocaleDateString();
        } catch {
            return dateString;
        }
    }

    getTimeAgo(dateString) {
        if (!dateString) return '';
        try {
            const date = new Date(dateString);
            const now = new Date();
            const diffTime = Math.abs(now - date);
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

            if (diffDays === 1) return '1 day ago';
            if (diffDays < 7) return `${diffDays} days ago`;
            if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
            return `${Math.floor(diffDays / 30)} months ago`;
        } catch {
            return '';
        }
    }

    getTrendIcon(trend) {
        switch (trend) {
            case 'improving': return '📈';
            case 'declining': return '📉';
            default: return '➡️';
        }
    }

    getTrendClass(trend) {
        switch (trend) {
            case 'improving': return 'positive';
            case 'declining': return 'negative';
            default: return 'neutral';
        }
    }

    formatTrend(trend) {
        return trend ? trend.charAt(0).toUpperCase() + trend.slice(1) : 'Neutral';
    }

    getInsightIcon(type) {
        switch (type) {
            case 'success': return '✅';
            case 'warning': return '⚠️';
            case 'error': return '❌';
            case 'info': return 'ℹ️';
            default: return '💡';
        }
    }

    renderNoDataChart(ctx, message) {
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = '#6b7280';
        ctx.font = '16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2);
    }

    showLoadingState() {
        const indicators = document.querySelectorAll('.stat-value');
        indicators.forEach(indicator => {
            indicator.textContent = '...';
        });
    }

    hideLoadingState() {
        // Loading state is automatically hidden when data is rendered
    }

    showLoadingIndicator(message) {
        const indicator = document.createElement('div');
        indicator.id = 'loading-indicator';
        indicator.className = 'loading-overlay';
        indicator.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
            </div>
        `;
        document.body.appendChild(indicator);
    }

    hideLoadingIndicator() {
        const indicator = document.getElementById('loading-indicator');
        if (indicator) indicator.remove();
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">❌</span>
                <span class="error-message">${message}</span>
                <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        document.body.insertBefore(errorDiv, document.body.firstChild);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) errorDiv.remove();
        }, 10000);
    }

    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-notification';
        successDiv.innerHTML = `
            <div class="success-content">
                <span class="success-icon">✅</span>
                <span class="success-message">${message}</span>
                <button class="success-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        document.body.insertBefore(successDiv, document.body.firstChild);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.style.animation = 'slideOut 0.3s ease forwards';
                setTimeout(() => successDiv.remove(), 300);
            }
        }, 5000);
    }

    initPerformanceChart() {
        const canvas = document.getElementById('performance-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const trends = this.data.trends?.performance_trend || [];

        if (trends.length === 0) {
            this.renderNoDataChart(ctx, 'No performance trend data available');
            return;
        }

        // Process data for Chart.js
        const desktopData = trends.filter(t => t.device_mode === 'desktop');
        const mobileData = trends.filter(t => t.device_mode === 'mobile');

        const labels = [...new Set(trends.map(t => this.formatDate(t.date)))];

        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Desktop Performance',
                        data: desktopData.map(d => d.performance_score),
                        borderColor: this.chartColors.desktop,
                        backgroundColor: this.chartColors.desktop + '20',
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: this.chartColors.desktop,
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    },
                    {
                        label: 'Mobile Performance',
                        data: mobileData.map(d => d.performance_score),
                        borderColor: this.chartColors.mobile,
                        backgroundColor: this.chartColors.mobile + '20',
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: this.chartColors.mobile,
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Score Trends',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Performance Score',
                            font: {
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date',
                            font: {
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }
            }
        });
    }

    initCWVChart() {
        const canvas = document.getElementById('cwv-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const cwvTrends = this.data.trends?.cwv_trend || {};

        if (Object.keys(cwvTrends).length === 0) {
            this.renderNoDataChart(ctx, 'No Core Web Vitals data available');
            return;
        }

        const datasets = [];

        // LCP Dataset
        if (cwvTrends.lcp && cwvTrends.lcp.length > 0) {
            const lcpData = cwvTrends.lcp.filter(d => d.device_mode === 'desktop');
            datasets.push({
                label: 'LCP (Desktop)',
                data: lcpData.map(d => d.lcp),
                borderColor: this.chartColors.lcp,
                backgroundColor: this.chartColors.lcp + '20',
                fill: false,
                tension: 0.4,
                pointBackgroundColor: this.chartColors.lcp,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5
            });
        }

        // FCP Dataset
        if (cwvTrends.fcp && cwvTrends.fcp.length > 0) {
            const fcpData = cwvTrends.fcp.filter(d => d.device_mode === 'desktop');
            datasets.push({
                label: 'FCP (Desktop)',
                data: fcpData.map(d => d.fcp),
                borderColor: this.chartColors.fcp,
                backgroundColor: this.chartColors.fcp + '20',
                fill: false,
                tension: 0.4,
                pointBackgroundColor: this.chartColors.fcp,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5
            });
        }

        const labels = [...new Set(cwvTrends.lcp?.map(t => this.formatDate(t.date)) || [])];

        this.charts.cwv = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Core Web Vitals Trends',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}s`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Seconds',
                            font: {
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date',
                            font: {
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                }
            }
        });
    }

    initGradeChart() {
        const canvas = document.getElementById('grade-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const gradeDistribution = this.data.summary?.grade_distribution || {};

        if (Object.keys(gradeDistribution).length === 0) {
            this.renderNoDataChart(ctx, 'No grade distribution data available');
            return;
        }

        const grades = Object.keys(gradeDistribution);
        const counts = Object.values(gradeDistribution);

        const gradeColors = {
            'A': '#10b981',
            'B': '#3b82f6',
            'C': '#f59e0b',
            'D': '#ef4444',
            'F': '#991b1b'
        };

        this.charts.grade = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: grades.map(grade => `Grade ${grade}`),
                datasets: [{
                    data: counts,
                    backgroundColor: grades.map(grade => gradeColors[grade] || '#6b7280'),
                    borderWidth: 3,
                    borderColor: '#ffffff',
                    hoverBackgroundColor: grades.map(grade => gradeColors[grade] || '#6b7280'),
                    hoverBorderWidth: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Grade Distribution',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true,
                    duration: 1000
                }
            }
        });
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new EnhancedPerformanceDashboard();
});

// Export for global access
window.dashboard = dashboard;
