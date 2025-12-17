/**
 * FOS Performance Dashboard JavaScript
 * This file contains the code for interactive charts and dashboard functionality.
 */

// Dashboard state
const dashboardState = {
    timeRange: '3m', // Default to 3 months
    selectedBrands: [],
    selectedMode: 'all', // 'desktop', 'mobile', or 'all'
    data: null,
    filteredData: null,
    filtersVisible: true // Track filter visibility state
};

// Load and initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initDashboard();
});

// Initialize the dashboard
async function initDashboard() {
    try {
        // Load dashboard data
        const response = await fetch('data/dashboard_data.json');
        dashboardState.data = await response.json();

        // Initialize filters
        initializeFilters(dashboardState.data);

        // Setup toggle filters button
        setupToggleFiltersButton();

        // Apply initial filters
        applyFilters();

        // Display loading state
        document.getElementById('loading-indicator').style.display = 'none';
        document.getElementById('dashboard-content').style.display = 'flex';

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        document.getElementById('loading-indicator').style.display = 'none';
        document.getElementById('error-message').style.display = 'block';
        document.getElementById('error-details').textContent = error.message;
    }
}

// Setup toggle filters button
function setupToggleFiltersButton() {
    const toggleButton = document.getElementById('toggle-filters');
    const filtersSection = document.getElementById('filters-section');

    if (toggleButton && filtersSection) {
        toggleButton.addEventListener('click', function() {
            dashboardState.filtersVisible = !dashboardState.filtersVisible;

            if (dashboardState.filtersVisible) {
                filtersSection.style.display = 'grid';
                toggleButton.textContent = 'Hide Filters';
            } else {
                filtersSection.style.display = 'none';
                toggleButton.textContent = 'Show Filters';
            }
        });
    }
}

// Initialize filters with data
function initializeFilters(data) {
    // Time range selector with expanded options
    const timeRangeSelect = document.getElementById('time-range');
    if (timeRangeSelect) {
        // Clear existing options
        timeRangeSelect.innerHTML = '';

        // Add expanded time range options
        const timeRangeOptions = [
            { value: '1h', label: 'Last Hour' },
            { value: '6h', label: 'Last 6 Hours' },
            { value: '24h', label: 'Last 24 Hours' },
            { value: '7d', label: 'Last 7 Days' },
            { value: '14d', label: 'Last 14 Days' },
            { value: '1m', label: 'Last Month' },
            { value: '3m', label: 'Last 3 Months' },
            { value: '6m', label: 'Last 6 Months' },
            { value: '1y', label: 'Last Year' },
            { value: 'all', label: 'All Time' }
        ];

        // Add specific report period options if we have unique_dates
        if (data.unique_dates && data.unique_dates.length > 0) {
            // Add a report-specific separator
            const separator = document.createElement('optgroup');
            separator.label = "Specific Report Periods";
            timeRangeSelect.appendChild(separator);

            // Sort dates chronologically (newest first)
            const reportDates = [...data.unique_dates].sort((a, b) => new Date(b) - new Date(a));

            // Add "All Reports" option first
            const allReportsOption = document.createElement('option');
            allReportsOption.value = 'all_reports';
            allReportsOption.textContent = `All Reports (${reportDates.length})`;
            timeRangeSelect.appendChild(allReportsOption);

            // Add options for each report date
            reportDates.forEach((dateStr, index) => {
                try {
                    const date = new Date(dateStr);
                    if (!isNaN(date)) {
                        const option = document.createElement('option');
                        option.value = `report_${index}`;

                        // Format date for display
                        const displayDate = date.toLocaleString('default', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                        });

                        option.textContent = `Report: ${displayDate}`;
                        option.dataset.timestamp = date.getTime();
                        option.dataset.dateStr = dateStr;
                        timeRangeSelect.appendChild(option);
                    }
                } catch (e) {
                    console.warn("Error creating option for date:", dateStr, e);
                }
            });

            // Add comparison options if we have at least 2 reports
            if (reportDates.length >= 2) {
                const comparisonSeparator = document.createElement('optgroup');
                comparisonSeparator.label = "Report Comparisons";
                timeRangeSelect.appendChild(comparisonSeparator);

                // Last 2 reports
                const last2Option = document.createElement('option');
                last2Option.value = 'last_2_reports';
                last2Option.textContent = 'Last 2 Reports';
                timeRangeSelect.appendChild(last2Option);

                // Last 3 reports
                if (reportDates.length >= 3) {
                    const last3Option = document.createElement('option');
                    last3Option.value = 'last_3_reports';
                    last3Option.textContent = 'Last 3 Reports';
                    timeRangeSelect.appendChild(last3Option);
                }

                // Last 4 reports
                if (reportDates.length >= 4) {
                    const last4Option = document.createElement('option');
                    last4Option.value = 'last_4_reports';
                    last4Option.textContent = 'Last 4 Reports';
                    timeRangeSelect.appendChild(last4Option);
                }
            }
        }

        // Add standard time range options under a separator
        const standardSeparator = document.createElement('optgroup');
        standardSeparator.label = "Time Ranges";
        timeRangeSelect.appendChild(standardSeparator);

        timeRangeOptions.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option.value;
            optionElement.textContent = option.label;
            if (option.value === dashboardState.timeRange) {
                optionElement.selected = true;
            }
            timeRangeSelect.appendChild(optionElement);
        });

        timeRangeSelect.addEventListener('change', function() {
            dashboardState.timeRange = this.value;

            // Store selected date string if it's a report-specific selection
            if (this.value.startsWith('report_')) {
                const selected = this.options[this.selectedIndex];
                dashboardState.selectedReportDate = selected.dataset.dateStr;
                dashboardState.selectedReportTimestamp = parseInt(selected.dataset.timestamp);
            } else if (this.value.includes('reports')) {
                // Handle report comparison options
                dashboardState.reportComparisonMode = this.value;
                dashboardState.selectedReportDate = null;
                dashboardState.selectedReportTimestamp = null;
            } else {
                dashboardState.reportComparisonMode = null;
                dashboardState.selectedReportDate = null;
                dashboardState.selectedReportTimestamp = null;
            }

            applyFilters();
        });
    }

    // Brand filter checkboxes
    const brandFilter = document.getElementById('brand-filter');
    if (brandFilter && data.unique_brands && data.unique_brands.length > 0) {
        // Clear existing filter options
        brandFilter.innerHTML = '';

        // Add "All Brands" option
        const allBrandsDiv = document.createElement('div');
        allBrandsDiv.className = 'filter-option';

        const allBrandsCheckbox = document.createElement('input');
        allBrandsCheckbox.type = 'checkbox';
        allBrandsCheckbox.id = 'brand-all';
        allBrandsCheckbox.checked = true;
        allBrandsCheckbox.addEventListener('change', function() {
            const brandCheckboxes = document.querySelectorAll('.brand-checkbox');
            brandCheckboxes.forEach(cb => {
                cb.checked = this.checked;
            });
            updateSelectedBrands();
        });

        const allBrandsLabel = document.createElement('label');
        allBrandsLabel.htmlFor = 'brand-all';
        allBrandsLabel.textContent = 'All Brands';

        allBrandsDiv.appendChild(allBrandsCheckbox);
        allBrandsDiv.appendChild(allBrandsLabel);
        brandFilter.appendChild(allBrandsDiv);

        // Add individual brand options - ensure all brands from the data are included
        data.unique_brands.forEach(brand => {
            if (!brand) return; // Skip empty brand names

            const brandDiv = document.createElement('div');
            brandDiv.className = 'filter-option';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `brand-${brand.toLowerCase().replace(/\s+/g, '-')}`;
            checkbox.value = brand;
            checkbox.className = 'brand-checkbox';
            checkbox.checked = true;
            checkbox.addEventListener('change', updateSelectedBrands);

            const label = document.createElement('label');
            label.htmlFor = checkbox.id;
            label.textContent = brand;

            brandDiv.appendChild(checkbox);
            brandDiv.appendChild(label);
            brandFilter.appendChild(brandDiv);
        });
    }

    // Device mode selector
    document.getElementById('device-mode').addEventListener('change', function() {
        dashboardState.selectedMode = this.value;
        applyFilters();
    });

    // Initialize selected brands
    updateSelectedBrands();
}

// Update selected brands based on checkboxes
function updateSelectedBrands() {
    const checkboxes = document.querySelectorAll('.brand-checkbox:checked');
    dashboardState.selectedBrands = Array.from(checkboxes).map(cb => cb.value);

    // Update "All Brands" checkbox state
    const allBrandsCheckbox = document.getElementById('brand-all');
    const brandCheckboxes = document.querySelectorAll('.brand-checkbox');

    if (dashboardState.selectedBrands.length === brandCheckboxes.length) {
        allBrandsCheckbox.checked = true;
        allBrandsCheckbox.indeterminate = false;
    } else if (dashboardState.selectedBrands.length === 0) {
        allBrandsCheckbox.checked = false;
        allBrandsCheckbox.indeterminate = false;
    } else {
        allBrandsCheckbox.indeterminate = true;
    }

    applyFilters();
}

// Apply filters to data and update visualizations
function applyFilters() {
    if (!dashboardState.data) return;

    // Filter time series data based on selected time range, brands, and mode
    let filteredData = filterTimeSeriesData(
        dashboardState.data.time_series,
        dashboardState.timeRange,
        dashboardState.selectedBrands,
        dashboardState.selectedMode
    );

    dashboardState.filteredData = filteredData;

    // Update all visualizations with filtered data
    updateScoreChart(filteredData);
    updateWebVitalsChart(filteredData);
    updatePerformanceTable(filteredData);
    updateKPIs(filteredData); // Ensure KPIs are updated with filtered data
    updateDiagnosticsSummaryChart(filteredData); // Update diagnostics summary chart
    updateBusinessInsights();
}

// Filter time series data based on selected filters
function filterTimeSeriesData(data, timeRange, brands, mode) {
    if (!data || !Array.isArray(data)) return [];

    // First, convert date strings to date objects for proper comparison
    const dataWithParsedDates = data.map(item => ({
        ...item,
        dateObj: new Date(item.date)
    }));

    // Handle report-specific filters
    if (timeRange.startsWith('report_')) {
        // Get specific report by index from unique_dates array
        const reportIndex = parseInt(timeRange.replace('report_', ''));
        if (!dashboardState.data || !dashboardState.data.unique_dates || reportIndex >= dashboardState.data.unique_dates.length) {
            return [];
        }

        const reportDate = dashboardState.data.unique_dates[reportIndex];
        const targetDate = new Date(reportDate);

        return dataWithParsedDates.filter(item => {
            // Filter by specific report date, using just the date part (ignoring time)
            const sameDate = item.dateObj.toDateString() === targetDate.toDateString();

            // Filter by brand
            const brandMatch = brands.length === 0 || brands.includes(item.brand);

            // Filter by mode
            const modeMatch = mode === 'all' || item.mode === mode;

            return sameDate && brandMatch && modeMatch;
        });
    }

    // Handle report comparison modes
    if (timeRange === 'all_reports') {
        // Include all reports
        return dataWithParsedDates.filter(item => {
            // Only filter by brand and mode
            const brandMatch = brands.length === 0 || brands.includes(item.brand);
            const modeMatch = mode === 'all' || item.mode === mode;
            return brandMatch && modeMatch;
        });
    }

    if (timeRange === 'last_2_reports' || timeRange === 'last_3_reports' || timeRange === 'last_4_reports') {
        // Get the last N reports
        const n = parseInt(timeRange.split('_')[1]);

        if (!dashboardState.data || !dashboardState.data.unique_dates || dashboardState.data.unique_dates.length < n) {
            return [];
        }

        // Get the dates of the last N reports (sorted newest first)
        const reportDates = [...dashboardState.data.unique_dates]
            .sort((a, b) => new Date(b) - new Date(a))
            .slice(0, n)
            .map(date => new Date(date).toDateString());

        return dataWithParsedDates.filter(item => {
            // Filter by report dates
            const dateMatch = reportDates.includes(item.dateObj.toDateString());

            // Filter by brand
            const brandMatch = brands.length === 0 || brands.includes(item.brand);

            // Filter by mode
            const modeMatch = mode === 'all' || item.mode === mode;

            return dateMatch && brandMatch && modeMatch;
        });
    }

    // Handle regular time ranges
    const now = new Date();
    let cutoffDate;

    switch(timeRange) {
        case '1h':
            cutoffDate = new Date(now.getTime() - (1 * 60 * 60 * 1000)); // 1 hour ago
            break;
        case '6h':
            cutoffDate = new Date(now.getTime() - (6 * 60 * 60 * 1000)); // 6 hours ago
            break;
        case '24h':
            cutoffDate = new Date(now.getTime() - (24 * 60 * 60 * 1000)); // 24 hours ago
            break;
        case '7d':
            cutoffDate = new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000)); // 7 days ago
            break;
        case '14d':
            cutoffDate = new Date(now.getTime() - (14 * 24 * 60 * 60 * 1000)); // 14 days ago
            break;
        case '1m':
            cutoffDate = new Date(now.setMonth(now.getMonth() - 1));
            break;
        case '3m':
            cutoffDate = new Date(now.setMonth(now.getMonth() - 3));
            break;
        case '6m':
            cutoffDate = new Date(now.setMonth(now.getMonth() - 6));
            break;
        case '1y':
            cutoffDate = new Date(now.setFullYear(now.getFullYear() - 1));
            break;
        case 'all':
            cutoffDate = new Date(0); // Beginning of time
            break;
        default:
            cutoffDate = new Date(now);
            cutoffDate.setMonth(now.getMonth() - 3); // Default to 3 months
    }

    return dataWithParsedDates.filter(item => {
        // Filter by date
        const dateMatch = item.dateObj >= cutoffDate;

        // Filter by brand
        const brandMatch = brands.length === 0 || brands.includes(item.brand);

        // Filter by mode
        const modeMatch = mode === 'all' || item.mode === mode;

        return dateMatch && brandMatch && modeMatch;
    });
}

// Update performance score chart
function updateScoreChart(filteredData) {
    const ctx = document.getElementById('performance-scores-chart').getContext('2d');
    const viewTypeSelector = document.getElementById('score-chart-view-type');
    const viewType = viewTypeSelector ? viewTypeSelector.value : 'daily';

    // Get if hourly data is available from dashboardState
    const hasHourlyData = dashboardState.data && dashboardState.data.has_hourly_data === true;

    // Display message if hourly view selected but no hourly data available
    const messageContainer = document.getElementById('hourly-data-message-container');
    if (!messageContainer) {
        const container = document.createElement('div');
        container.id = 'hourly-data-message-container';
        container.style.textAlign = 'center';
        container.style.marginTop = '10px';
        ctx.canvas.parentNode.appendChild(container);
    }

    // For hourly view, check if we have time information in the data
    const hasTimeInformation = filteredData.some(item => {
        const dateStr = item.date;
        // Check for full timestamp format (YYYY-MM-DD HH:MM:SS)
        return dateStr && (dateStr.includes('T') || dateStr.includes(':'));
    });

    // Use daily view if hourly is selected but no time information is available
    let effectiveViewType = viewType;
    if (viewType === 'hourly' && !hasTimeInformation && !hasHourlyData) {
        effectiveViewType = 'daily';
    }

    // Show or hide message about hourly data
    if (viewType === 'hourly' && !hasTimeInformation && !hasHourlyData) {
        const messageContainer = document.getElementById('hourly-data-message-container');
        messageContainer.innerHTML = '<div style="color: #d9534f; padding: 5px; border-radius: 3px; font-weight: bold;">Hourly data is not available. Showing daily view instead.</div>';
        messageContainer.style.display = 'block';
    } else {
        const messageContainer = document.getElementById('hourly-data-message-container');
        if (messageContainer) {
            messageContainer.style.display = 'none';
        }
    }

    // Check if we're dealing with report-specific data
    const isReportSpecific = dashboardState.timeRange &&
        (dashboardState.timeRange.startsWith('report_') ||
         dashboardState.timeRange.includes('reports'));

    // Prepare data for chart with appropriate time grouping
    let timeLabels = [];
    let groupedData = {};

    // Special handling for report-specific views
    if (isReportSpecific && dashboardState.reportComparisonMode) {
        // For report comparisons (last_2_reports, last_3_reports, etc.)
        filteredData.forEach(item => {
            // Use date as the key for grouping
            const dateKey = item.date;

            if (!groupedData[dateKey]) {
                groupedData[dateKey] = {};
                timeLabels.push(dateKey);
            }

            if (!groupedData[dateKey][item.brand]) {
                groupedData[dateKey][item.brand] = [];
            }

            groupedData[dateKey][item.brand].push(item);
        });

        // Sort chronologically (newest first for report comparisons)
        timeLabels.sort((a, b) => new Date(b) - new Date(a));

        // Instead of limiting to a specific number of reports by slicing,
        // we'll keep ALL reports but prioritize the most recent ones at the top
        // This ensures we don't lose intermediate data points

        // The timeLabels array now contains ALL unique dates from the filtered data
        console.log(`Using all ${timeLabels.length} unique report dates for chart`);

        // Sort chronologically (oldest to newest for display)
        timeLabels.sort((a, b) => new Date(a) - new Date(b));
    } else if ((effectiveViewType === 'hourly' && (hasTimeInformation || hasHourlyData))) {
        // Extract date parts and hour parts from the timestamps
        filteredData.forEach(item => {
            let dateStr, hourPart, hourKey;

            // Check for different formats of date/time
            if (item.date.includes('T')) {
                // ISO format: 2025-06-14T15:30:00
                const [datePart, timePart] = item.date.split('T');
                dateStr = datePart;
                hourPart = timePart.split(':')[0] + ':00';
                hourKey = `${datePart} ${hourPart}`;
            } else if (item.date.includes(':')) {
                // Format with space: 2025-06-14 15:30:00
                const parts = item.date.split(' ');
                dateStr = parts[0];
                if (parts.length > 1) {
                    const timeParts = parts[1].split(':');
                    hourPart = timeParts[0] + ':00';
                    hourKey = `${dateStr} ${hourPart}`;
                } else {
                    // Fallback if time part is missing
                    hourKey = `${dateStr} 00:00`;
                }
            } else {
                // Plain date without time: use 00:00
                dateStr = item.date;
                hourKey = `${dateStr} 00:00`;
            }

            if (!groupedData[hourKey]) {
                groupedData[hourKey] = {};
                timeLabels.push(hourKey);
            }

            if (!groupedData[hourKey][item.brand]) {
                groupedData[hourKey][item.brand] = [];
            }

            groupedData[hourKey][item.brand].push(item);
        });

        // Sort chronologically
        timeLabels.sort();
    } else if (effectiveViewType === 'daily') {
        // Daily view - extract just the date part if timestamps are present
        filteredData.forEach(item => {
            // Extract just the date part if the date contains time information
            let datePart = item.date;
            if (datePart.includes('T')) {
                datePart = datePart.split('T')[0];
            } else if (datePart.includes(' ')) {
                datePart = datePart.split(' ')[0];
            }

            if (!groupedData[datePart]) {
                groupedData[datePart] = {};
                if (!timeLabels.includes(datePart)) {
                    timeLabels.push(datePart);
                }
            }

            if (!groupedData[datePart][item.brand]) {
                groupedData[datePart][item.brand] = [];
            }

            groupedData[datePart][item.brand].push(item);
        });

        // Sort dates chronologically
        timeLabels.sort();
    } else if (effectiveViewType === 'weekly') {
        // Weekly view - group by week
        filteredData.forEach(item => {
            let dateObj = new Date(item.date);
            // Get the start of the week (Sunday)
            const startOfWeek = new Date(dateObj);
            startOfWeek.setDate(dateObj.getDate() - dateObj.getDay());
            const weekKey = startOfWeek.toISOString().split('T')[0];

            if (!groupedData[weekKey]) {
                groupedData[weekKey] = {};
                if (!timeLabels.includes(weekKey)) {
                    timeLabels.push(weekKey);
                }
            }

            if (!groupedData[weekKey][item.brand]) {
                groupedData[weekKey][item.brand] = [];
            }

            groupedData[weekKey][item.brand].push(item);
        });

        // Sort dates chronologically
        timeLabels.sort();
    } else if (effectiveViewType === 'monthly') {
        // Monthly view - group by month
        filteredData.forEach(item => {
            let dateObj = new Date(item.date);
            const monthKey = `${dateObj.getFullYear()}-${String(dateObj.getMonth() + 1).padStart(2, '0')}`;

            if (!groupedData[monthKey]) {
                groupedData[monthKey] = {};
                if (!timeLabels.includes(monthKey)) {
                    timeLabels.push(monthKey);
                }
            }

            if (!groupedData[monthKey][item.brand]) {
                groupedData[monthKey][item.brand] = [];
            }

            groupedData[monthKey][item.brand].push(item);
        });

        // Sort dates chronologically
        timeLabels.sort();
    } else if (effectiveViewType === 'yearly') {
        // Yearly view - group by year
        filteredData.forEach(item => {
            let dateObj = new Date(item.date);
            const yearKey = `${dateObj.getFullYear()}`;

            if (!groupedData[yearKey]) {
                groupedData[yearKey] = {};
                if (!timeLabels.includes(yearKey)) {
                    timeLabels.push(yearKey);
                }
            }

            if (!groupedData[yearKey][item.brand]) {
                groupedData[yearKey][item.brand] = [];
            }

            groupedData[yearKey][item.brand].push(item);
        });

        // Sort dates chronologically
        timeLabels.sort();
    }

    const brands = [...new Set(filteredData.map(item => item.brand))];

    // Create datasets
    const datasets = brands.map((brand, index) => {
        // Calculate average performance score for each time point
        const scores = timeLabels.map(timeLabel => {
            if (groupedData[timeLabel] && groupedData[timeLabel][brand] && groupedData[timeLabel][brand].length > 0) {
                return groupedData[timeLabel][brand].reduce(
                    (sum, item) => sum + (item.performance_score || 0), 0
                ) / groupedData[timeLabel][brand].length;
            }
            return null;
        });

        // Log the number of data points for debugging
        console.log(`Brand ${brand}: Found ${scores.filter(s => s !== null).length} data points out of ${timeLabels.length} time labels`);

        // Generate color based on index
        const hue = (index * 137) % 360;
        const color = `hsl(${hue}, 70%, 50%)`;

        return {
            label: brand,
            data: scores,
            borderColor: color,
            backgroundColor: `hsla(${hue}, 70%, 50%, 0.1)`,
            fill: false,
            tension: 0.4,
            spanGaps: !isReportSpecific // Only connect across gaps for non-report-specific data
        };
    });

    // Clear previous chart if it exists
    if (window.performanceChart) {
        window.performanceChart.destroy();
    }

    // Format x-axis labels based on view type
    const formattedLabels = timeLabels.map(timeLabel => {
        if (effectiveViewType === 'hourly') {
            // For hourly view, format nicely
            // For example: "2025-06-14 15:00" -> "Jun 14, 15:00"
            try {
                if (timeLabel.includes(' ')) {
                    const [datePart, hourPart] = timeLabel.split(' ');
                    const dateObj = new Date(datePart);
                    const month = dateObj.toLocaleString('default', { month: 'short' });
                    const day = dateObj.getDate();
                    return `${month} ${day}, ${hourPart}`;
                } else if (timeLabel.includes('T')) {
                    const [datePart, timePart] = timeLabel.split('T');
                    const dateObj = new Date(datePart);
                    const month = dateObj.toLocaleString('default', { month: 'short' });
                    const day = dateObj.getDate();
                    const hour = timePart.split(':')[0];
                    return `${month} ${day}, ${hour}:00`;
                }
                return timeLabel;
            } catch (e) {
                return timeLabel; // Fall back to original format if parsing fails
            }
        } else if (effectiveViewType === 'daily') {
            // For daily view, format "YYYY-MM-DD" to "Mon DD"
            try {
                const dateObj = new Date(timeLabel);
                const month = dateObj.toLocaleString('default', { month: 'short' });
                const day = dateObj.getDate();
                return `${month} ${day}`;
            } catch (e) {
                return timeLabel; // Fall back to original format if parsing fails
            }
        } else if (effectiveViewType === 'weekly') {
            // For weekly view, format to "Week of Mon DD"
            try {
                const dateObj = new Date(timeLabel);
                const month = dateObj.toLocaleString('default', { month: 'short' });
                const day = dateObj.getDate();
                return `Week of ${month} ${day}`;
            } catch (e) {
                return timeLabel; // Fall back to original format if parsing fails
            }
        } else if (effectiveViewType === 'monthly') {
            // For monthly view, format "YYYY-MM" to "Mon YYYY"
            try {
                const [year, month] = timeLabel.split('-');
                const dateObj = new Date(year, parseInt(month) - 1, 1);
                const monthName = dateObj.toLocaleString('default', { month: 'short' });
                return `${monthName} ${year}`;
            } catch (e) {
                return timeLabel; // Fall back to original format if parsing fails
            }
        } else if (effectiveViewType === 'yearly') {
            // For yearly view, just show the year
            return timeLabel;
        } else {
            return timeLabel; // Fallback for any other view type
        }
    });

    // Create new chart
    window.performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: getChartTitle(effectiveViewType)
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: function(tooltipItems) {
                            // Show detailed time in tooltip
                            const idx = tooltipItems[0].dataIndex;
                            const originalLabel = timeLabels[idx];
                            return originalLabel;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Performance Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: getAxisTitle(effectiveViewType)
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Helper function for chart titles based on view type
function getChartTitle(viewType) {
    switch(viewType) {
        case 'hourly':
            return 'Hourly Performance Score Trend';
        case 'daily':
            return 'Daily Performance Score Trend';
        case 'weekly':
            return 'Weekly Performance Score Trend';
        case 'monthly':
            return 'Monthly Performance Score Trend';
        case 'yearly':
            return 'Yearly Performance Score Trend';
        default:
            return 'Performance Score Trend';
    }
}

// Helper function for x-axis titles based on view type
function getAxisTitle(viewType) {
    switch(viewType) {
        case 'hourly':
            return 'Date and Hour';
        case 'daily':
            return 'Date';
        case 'weekly':
            return 'Week';
        case 'monthly':
            return 'Month';
        case 'yearly':
            return 'Year';
        default:
            return 'Time Period';
    }
}

// Update Core Web Vitals chart
function updateWebVitalsChart(filteredData) {
    const ctx = document.getElementById('core-web-vitals-chart').getContext('2d');

    // Prepare data for chart
    const brands = [...new Set(filteredData.map(item => item.brand))];

    // Aggregate latest data for each brand
    const latestDataByBrand = {};

    brands.forEach(brand => {
        const brandData = filteredData.filter(item => item.brand === brand);
        if (brandData.length > 0) {
            // Get the most recent date for this brand
            const latestDate = new Date(Math.max(...brandData.map(item => new Date(item.date))));

            // Get data for this latest date
            const latestItems = brandData.filter(item => new Date(item.date).getTime() === latestDate.getTime());

            // Average the metrics
            if (latestItems.length > 0) {
                latestDataByBrand[brand] = {
                    fcp: latestItems.reduce((sum, item) => sum + (item.fcp || 0), 0) / latestItems.length,
                    lcp: latestItems.reduce((sum, item) => sum + (item.lcp || 0), 0) / latestItems.length,
                    cls: latestItems.reduce((sum, item) => sum + (item.cls || 0), 0) / latestItems.length,
                    inp: latestItems.reduce((sum, item) => sum + (item.inp || 0), 0) / latestItems.length,
                    core_web_vitals_score: latestItems.reduce((sum, item) => sum + (item.core_web_vitals_score || 0), 0) / latestItems.length
                };
            }
        }
    });

    // Define metric thresholds for core web vitals
    const thresholds = {
        fcp: { good: 1.8, needsImprovement: 3.0 },
        lcp: { good: 2.5, needsImprovement: 4.0 },
        cls: { good: 0.1, needsImprovement: 0.25 },
        inp: { good: 200, needsImprovement: 500 }
    };

    // Convert metrics to normalized scores (0-100)
    const normalizedScores = {};

    Object.entries(latestDataByBrand).forEach(([brand, metrics]) => {
        normalizedScores[brand] = {
            fcp: metrics.fcp <= thresholds.fcp.good ? 100 :
                 metrics.fcp <= thresholds.fcp.needsImprovement ?
                 50 + (thresholds.fcp.needsImprovement - metrics.fcp) /
                 (thresholds.fcp.needsImprovement - thresholds.fcp.good) * 50 :
                 Math.max(0, 50 - (metrics.fcp - thresholds.fcp.needsImprovement) * 10),

            lcp: metrics.lcp <= thresholds.lcp.good ? 100 :
                 metrics.lcp <= thresholds.lcp.needsImprovement ?
                 50 + (thresholds.lcp.needsImprovement - metrics.lcp) /
                 (thresholds.lcp.needsImprovement - thresholds.lcp.good) * 50 :
                 Math.max(0, 50 - (metrics.lcp - thresholds.lcp.needsImprovement) * 10),

            cls: metrics.cls <= thresholds.cls.good ? 100 :
                 metrics.cls <= thresholds.cls.needsImprovement ?
                 50 + (thresholds.cls.needsImprovement - metrics.cls) /
                 (thresholds.cls.needsImprovement - thresholds.cls.good) * 50 :
                 Math.max(0, 50 - (metrics.cls - thresholds.cls.needsImprovement) * 100),

            inp: metrics.inp <= thresholds.inp.good ? 100 :
                 metrics.inp <= thresholds.inp.needsImprovement ?
                 50 + (thresholds.inp.needsImprovement - metrics.inp) /
                 (thresholds.inp.needsImprovement - thresholds.inp.good) * 50 :
                 Math.max(0, 50 - (metrics.inp - thresholds.inp.needsImprovement) / 10)
        };
    });

    // Create datasets
    const datasets = [
        {
            label: 'FCP Score',
            data: brands.map(brand => normalizedScores[brand]?.fcp || 0),
            backgroundColor: 'rgba(153, 102, 255, 0.7)'
        },
        {
            label: 'LCP Score',
            data: brands.map(brand => normalizedScores[brand]?.lcp || 0),
            backgroundColor: 'rgba(255, 99, 132, 0.7)'
        },
        {
            label: 'CLS Score',
            data: brands.map(brand => normalizedScores[brand]?.cls || 0),
            backgroundColor: 'rgba(54, 162, 235, 0.7)'
        },
        {
            label: 'INP Score',
            data: brands.map(brand => normalizedScores[brand]?.inp || 0),
            backgroundColor: 'rgba(75, 192, 192, 0.7)'
        }
    ];

    // Clear previous chart if it exists
    if (window.webVitalsChart) {
        window.webVitalsChart.destroy();
    }

    // Create new chart
    window.webVitalsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: brands,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Core Web Vitals Scores by Brand'
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const brand = context.label;
                            const originalMetric = latestDataByBrand[brand];
                            const metricName = context.dataset.label;

                            if (metricName === 'FCP Score') {
                                return `Original FCP: ${originalMetric.fcp.toFixed(2)}s`;
                            } else if (metricName === 'LCP Score') {
                                return `Original LCP: ${originalMetric.lcp.toFixed(2)}s`;
                            } else if (metricName === 'CLS Score') {
                                return `Original CLS: ${originalMetric.cls.toFixed(3)}`;
                            } else if (metricName === 'INP Score') {
                                return `Original INP: ${originalMetric.inp.toFixed(0)}ms`;
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Normalized Score (0-100)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Brand'
                    }
                }
            }
        }
    });
}

// Update KPIs based on the filtered data
function updateKPIs(filteredData) {
    if (!filteredData || filteredData.length === 0) {
        // Set default values if no data
        document.getElementById('kpi-performance').textContent = '-';
        document.getElementById('kpi-cwv').textContent = '-';
        document.getElementById('cwv-progress-bar').style.width = '0%';
        document.getElementById('kpi-lcp').textContent = '-';
        document.getElementById('kpi-cls').textContent = '-';
        document.getElementById('kpi-fcp').textContent = '-';
        document.getElementById('kpi-inp').textContent = '-';
        return;
    }

    // Calculate average performance score across all brands
    const avgPerformanceScore = filteredData.reduce((sum, item) =>
        sum + (item.performance_score || 0), 0) / filteredData.length;

    // Calculate average Core Web Vitals score
    const avgCWVScore = filteredData.reduce((sum, item) =>
        sum + (item.core_web_vitals_score || 0), 0) / filteredData.length;

    // Calculate average LCP
    const avgLCP = filteredData.reduce((sum, item) =>
        sum + (item.lcp || 0), 0) / filteredData.length;

    // Calculate average CLS
    const avgCLS = filteredData.reduce((sum, item) =>
        sum + (item.cls || 0), 0) / filteredData.length;

    // Calculate average FCP
    const avgFCP = filteredData.reduce((sum, item) =>
        sum + (item.fcp || 0), 0) / filteredData.length;

    // Calculate average INP - handle cases where INP might be stored in different formats
    let avgINP = 0;
    let inpCount = 0;
    filteredData.forEach(item => {
        if (item.inp !== null && item.inp !== undefined) {
            // Convert string format (like "150 ms") to number if necessary
            if (typeof item.inp === 'string') {
                const match = item.inp.match(/(\d+(\.\d+)?)/);
                if (match) {
                    let value = parseFloat(match[1]);
                    // If the value is very small (likely in seconds), convert to milliseconds
                    if (value < 10 && !item.inp.includes('ms')) {
                        value *= 1000;
                    }
                    avgINP += value;
                    inpCount++;
                }
            } else if (typeof item.inp === 'number') {
                let value = item.inp;
                // If the value is very small (likely in seconds), convert to milliseconds
                if (value < 10) {
                    value *= 1000;
                }
                avgINP += value;
                inpCount++;
            }
        }
    });
    avgINP = inpCount > 0 ? avgINP / inpCount : 0;

    // Update KPI elements
    document.getElementById('kpi-performance').textContent = avgPerformanceScore.toFixed(1);
    document.getElementById('kpi-cwv').textContent = avgCWVScore.toFixed(1) + '%';
    document.getElementById('cwv-progress-bar').style.width = `${avgCWVScore}%`;
    document.getElementById('kpi-lcp').textContent = avgLCP.toFixed(2) + 's';
    document.getElementById('kpi-cls').textContent = avgCLS.toFixed(3);
    document.getElementById('kpi-fcp').textContent = avgFCP.toFixed(2) + 's';
    document.getElementById('kpi-inp').textContent = avgINP.toFixed(0) + 'ms';

    // Add trend indicators if we have historical data to compare
    if (dashboardState.data && dashboardState.data.time_series) {
        updateTrendIndicators(filteredData);
    }
}

// Update trend indicators for KPIs
function updateTrendIndicators(filteredData) {
    // Sort the filtered data by date
    const sortedData = [...filteredData].sort((a, b) => {
        return new Date(a.date) - new Date(b.date);
    });

    // We need at least two data points to calculate trends
    if (sortedData.length < 2) return;

    // Get first and last data points for comparison
    const firstQuarter = sortedData.slice(0, Math.ceil(sortedData.length / 4));
    const lastQuarter = sortedData.slice(-Math.ceil(sortedData.length / 4));

    // Calculate average metrics for first and last quarter
    const firstQtrPerf = firstQuarter.reduce((sum, item) => sum + (item.performance_score || 0), 0) / firstQuarter.length;
    const lastQtrPerf = lastQuarter.reduce((sum, item) => sum + (item.performance_score || 0), 0) / lastQuarter.length;

    const firstQtrLCP = firstQuarter.reduce((sum, item) => sum + (item.lcp || 0), 0) / firstQuarter.length;
    const lastQtrLCP = lastQuarter.reduce((sum, item) => sum + (item.lcp || 0), 0) / lastQuarter.length;

    const firstQtrCLS = firstQuarter.reduce((sum, item) => sum + (item.cls || 0), 0) / firstQuarter.length;
    const lastQtrCLS = lastQuarter.reduce((sum, item) => sum + (item.cls || 0), 0) / lastQuarter.length;

    const firstQtrFCP = firstQuarter.reduce((sum, item) => sum + (item.fcp || 0), 0) / firstQuarter.length;
    const lastQtrFCP = lastQuarter.reduce((sum, item) => sum + (item.fcp || 0), 0) / lastQuarter.length;

    const firstQtrINP = firstQuarter.reduce((sum, item) => sum + (item.inp || 0), 0) / firstQuarter.length;
    const lastQtrINP = lastQuarter.reduce((sum, item) => sum + (item.inp || 0), 0) / lastQuarter.length;

    // Update performance score trend indicator
    const perfTrend = document.getElementById('kpi-performance-trend');
    const perfChange = lastQtrPerf - firstQtrPerf;
    // Only calculate percentage if firstQtrPerf is not zero to avoid division by zero
    const perfChangePercent = firstQtrPerf !== 0 ? ((perfChange / Math.abs(firstQtrPerf)) * 100).toFixed(1) : 0;

    if (Math.abs(perfChange) >= 1) {
        perfTrend.innerHTML = perfChange > 0 ?
            `<span class="trend-up good">↑ ${perfChangePercent}%</span>` :
            `<span class="trend-down bad">↓ ${Math.abs(perfChangePercent)}%</span>`;
    } else {
        perfTrend.innerHTML = `<span class="trend-neutral">→ Stable</span>`;
    }

    // Update LCP trend indicator (lower is better)
    const lcpTrend = document.getElementById('kpi-lcp-trend');
    const lcpChange = lastQtrLCP - firstQtrLCP;
    // Only calculate percentage if firstQtrLCP is not zero
    const lcpChangePercent = firstQtrLCP !== 0 ? ((lcpChange / Math.abs(firstQtrLCP)) * 100).toFixed(1) : 0;

    if (Math.abs(lcpChange) >= 0.1) {
        lcpTrend.innerHTML = lcpChange < 0 ?
            `<span class="trend-down good">↓ ${Math.abs(lcpChangePercent)}%</span>` :
            `<span class="trend-up bad">↑ ${lcpChangePercent}%</span>`;
    } else {
        lcpTrend.innerHTML = `<span class="trend-neutral">→ Stable</span>`;
    }

    // Update CLS trend indicator (lower is better)
    const clsTrend = document.getElementById('kpi-cls-trend');
    const clsChange = lastQtrCLS - firstQtrCLS;
    // Only calculate percentage if firstQtrCLS is not zero
    const clsChangePercent = firstQtrCLS !== 0 ? ((clsChange / Math.abs(firstQtrCLS)) * 100).toFixed(1) : 0;

    if (Math.abs(clsChange) >= 0.01) {
        clsTrend.innerHTML = clsChange < 0 ?
            `<span class="trend-down good">↓ ${Math.abs(clsChangePercent)}%</span>` :
            `<span class="trend-up bad">↑ ${clsChangePercent}%</span>`;
    } else {
        clsTrend.innerHTML = `<span class="trend-neutral">→ Stable</span>`;
    }

    // Update FCP trend indicator (lower is better)
    const fcpTrend = document.getElementById('kpi-fcp-trend');
    if (fcpTrend) {
        const fcpChange = lastQtrFCP - firstQtrFCP;
        // Only calculate percentage if firstQtrFCP is not zero
        const fcpChangePercent = firstQtrFCP !== 0 ? ((fcpChange / Math.abs(firstQtrFCP)) * 100).toFixed(1) : 0;

        if (Math.abs(fcpChange) >= 0.1) {
            fcpTrend.innerHTML = fcpChange < 0 ?
                `<span class="trend-down good">↓ ${Math.abs(fcpChangePercent)}%</span>` :
                `<span class="trend-up bad">↑ ${fcpChangePercent}%</span>`;
        } else {
            fcpTrend.innerHTML = `<span class="trend-neutral">→ Stable</span>`;
        }
    }

    // Update INP trend indicator (lower is better)
    const inpTrend = document.getElementById('kpi-inp-trend');
    if (inpTrend) {
        const inpChange = lastQtrINP - firstQtrINP;
        // Only calculate percentage if firstQtrINP is not zero
        const inpChangePercent = firstQtrINP !== 0 ? ((inpChange / Math.abs(firstQtrINP)) * 100).toFixed(1) : 0;

        if (Math.abs(inpChange) >= 10) {
            inpTrend.innerHTML = inpChange < 0 ?
                `<span class="trend-down good">↓ ${Math.abs(inpChangePercent)}%</span>` :
                `<span class="trend-up bad">↑ ${inpChangePercent}%</span>`;
        } else {
            inpTrend.innerHTML = `<span class="trend-neutral">→ Stable</span>`;
        }
    }
}

// Update performance metrics table
function updatePerformanceTable(filteredData) {
    const tableBody = document.querySelector('#performance-table tbody');
    tableBody.innerHTML = '';

    // Group data by brand and mode
    const groupedData = {};

    filteredData.forEach(item => {
        const key = `${item.brand}_${item.mode}`;
        if (!groupedData[key]) {
            groupedData[key] = [];
        }
        groupedData[key].push(item);
    });

    // Create table rows for each brand and mode
    Object.entries(groupedData).forEach(([key, items]) => {
        const [brand, mode] = key.split('_');

        // Calculate average metrics
        const avgPerformance = items.reduce((sum, item) => sum + (item.performance_score || 0), 0) / items.length;
        const avgLCP = items.reduce((sum, item) => sum + (item.lcp || 0), 0) / items.length;
        const avgFCP = items.reduce((sum, item) => sum + (item.fcp || 0), 0) / items.length;
        const avgCLS = items.reduce((sum, item) => sum + (item.cls || 0), 0) / items.length;

        // Special handling for INP values to account for different formats
        let avgINP = 0;
        let inpCount = 0;
        items.forEach(item => {
            if (item.inp !== null && item.inp !== undefined) {
                // Convert string format (like "150 ms") to number if necessary
                if (typeof item.inp === 'string') {
                    const match = item.inp.match(/(\d+(\.\d+)?)/);
                    if (match) {
                        let value = parseFloat(match[1]);
                        // If the value is very small (likely in seconds), convert to milliseconds
                        if (value < 10 && !item.inp.includes('ms')) {
                            value *= 1000;
                        }
                        avgINP += value;
                        inpCount++;
                    }
                } else if (typeof item.inp === 'number') {
                    let value = item.inp;
                    // If the value is very small (likely in seconds), convert to milliseconds
                    if (value < 10) {
                        value *= 1000;
                    }
                    avgINP += value;
                    inpCount++;
                }
            }
        });
        avgINP = inpCount > 0 ? avgINP / inpCount : 0;

        const avgCWV = items.reduce((sum, item) => sum + (item.core_web_vitals_score || 0), 0) / items.length;

        // Create table row
        const row = document.createElement('tr');

        // Create and append cells
        const brandCell = document.createElement('td');
        brandCell.textContent = brand;
        row.appendChild(brandCell);

        const modeCell = document.createElement('td');
        modeCell.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
        row.appendChild(modeCell);

        const perfCell = document.createElement('td');
        perfCell.textContent = avgPerformance.toFixed(1);
        perfCell.className = getScoreClass(avgPerformance);
        row.appendChild(perfCell);

        const lcpCell = document.createElement('td');
        lcpCell.textContent = avgLCP.toFixed(2) + 's';
        lcpCell.className = getLCPClass(avgLCP);
        row.appendChild(lcpCell);

        const fcpCell = document.createElement('td');
        fcpCell.textContent = avgFCP.toFixed(2) + 's';
        fcpCell.className = getFCPClass(avgFCP);
        row.appendChild(fcpCell);

        const clsCell = document.createElement('td');
        clsCell.textContent = avgCLS.toFixed(3);
        clsCell.className = getCLSClass(avgCLS);
        row.appendChild(clsCell);

        const inpCell = document.createElement('td');
        inpCell.textContent = avgINP.toFixed(0) + 'ms';
        inpCell.className = getINPClass(avgINP);
        row.appendChild(inpCell);

        const cwvCell = document.createElement('td');
        cwvCell.textContent = avgCWV.toFixed(1);
        cwvCell.className = getScoreClass(avgCWV);
        row.appendChild(cwvCell);

        tableBody.appendChild(row);
    });
}

// Update business insights
function updateBusinessInsights() {
    if (!dashboardState.data || !dashboardState.data.insights) return;

    const insightsContainer = document.getElementById('business-insights');
    insightsContainer.innerHTML = '';

    // Display insights
    dashboardState.data.insights.forEach(insight => {
        const insightCard = document.createElement('div');
        insightCard.className = 'insight-card';

        // Set impact level class
        let impactClass = 'impact-medium';
        if (insight.impact === 'High') {
            impactClass = 'impact-high';
        } else if (insight.impact === 'Low') {
            impactClass = 'impact-low';
        }

        insightCard.classList.add(impactClass);

        // Create insight content
        insightCard.innerHTML = `
            <h3>${insight.title}</h3>
            <div class="impact-label ${impactClass}">${insight.impact} Impact</div>
            <p>${insight.description}</p>
            <div class="insight-recommendation">
                <strong>Recommendation:</strong> ${insight.recommendation}
            </div>
        `;

        insightsContainer.appendChild(insightCard);
    });
}

// Transform diagnostics data from JSON format to bubble chart format
function transformDiagnosticsData(diagnosticsData) {
    console.log("Transforming diagnostics data:", diagnosticsData);

    const bubbleData = [];
    let colorIndex = 0;

    diagnosticsData.forEach(item => {
        if (!item.brand || !item.mode) return;

        // Only add if we have valid numeric values
        const pageSize = parseFloat(item.page_size_kb);
        const requestCount = parseFloat(item.request_count);

        if (isNaN(pageSize) || isNaN(requestCount) || pageSize <= 0 || requestCount <= 0) {
            console.warn(`Skipping invalid data point: ${item.brand}/${item.mode} - page_size_kb: ${item.page_size_kb}, request_count: ${item.request_count}`);
            return;
        }

        console.log(`Adding diagnostics data point: ${item.brand}/${item.mode} - Size: ${pageSize.toFixed(2)} KB, Requests: ${requestCount.toFixed(0)}`);

        // Generate a color for this brand/mode
        const hue = (colorIndex * 137) % 360;
        const color = `hsla(${hue}, 70%, 50%, 0.7)`;
        colorIndex++;

        bubbleData.push({
            x: requestCount,
            y: pageSize,
            r: Math.max(8, Math.min(25, 5 + (pageSize / 200))), // Bubble size based on page size (min: 8, max: 25)
            brand: item.brand,
            mode: item.mode,
            color: color,
            borderColor: color.replace('0.7', '1')
        });
    });

    return bubbleData;
}

// Create the diagnostics chart with the provided data
function createDiagnosticsChart(ctx, bubbleData) {
    if (!bubbleData || bubbleData.length === 0) {
        showNoDataMessage(ctx);
        return;
    }

    console.log(`Creating diagnostics chart with ${bubbleData.length} data points`);

    // Create the chart
    window.diagnosticsSummaryChart = new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: bubbleData.map(item => ({
                label: `${item.brand} (${item.mode})`,
                data: [{
                    x: item.x,
                    y: item.y,
                    r: item.r
                }],
                backgroundColor: item.color,
                borderColor: item.borderColor,
                borderWidth: 1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Page Size vs Number of Requests by Brand and Mode'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const brand = bubbleData[context.datasetIndex].brand;
                            const mode = bubbleData[context.datasetIndex].mode;
                            const pageSize = bubbleData[context.datasetIndex].y;
                            const requestCount = bubbleData[context.datasetIndex].x;

                            return [
                                `Brand: ${brand} (${mode})`,
                                `Page Size: ${pageSize.toFixed(2)} KB`,
                                `Requests: ${requestCount.toFixed(0)}`
                            ];
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Requests'
                    },
                    beginAtZero: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Page Size (KB)'
                    },
                    beginAtZero: true
                }
            }
        }
    });

    // Add a legend below the chart
    createCustomLegend(bubbleData, ctx.canvas);
}

// Create a custom legend below the chart
function createCustomLegend(bubbleData, canvas) {
    // Remove existing legend if any
    const existingLegend = document.getElementById('custom-diagnostics-legend');
    if (existingLegend) {
        existingLegend.remove();
    }

    // Create legend container
    const legendContainer = document.createElement('div');
    legendContainer.id = 'custom-diagnostics-legend';
    legendContainer.style.display = 'flex';
    legendContainer.style.flexWrap = 'wrap';
    legendContainer.style.justifyContent = 'center';
    legendContainer.style.marginTop = '10px';

    // Group items by brand
    const brandGroups = {};
    bubbleData.forEach(item => {
        if (!brandGroups[item.brand]) {
            brandGroups[item.brand] = [];
        }
        brandGroups[item.brand].push(item);
    });

    // Create legend items
    Object.entries(brandGroups).forEach(([brand, items]) => {
        items.forEach(item => {
            const legendItem = document.createElement('div');
            legendItem.style.display = 'flex';
            legendItem.style.alignItems = 'center';
            legendItem.style.margin = '0 10px 5px 0';

            const colorBox = document.createElement('div');
            colorBox.style.width = '12px';
            colorBox.style.height = '12px';
            colorBox.style.backgroundColor = item.color;
            colorBox.style.borderRadius = '50%';
            colorBox.style.marginRight = '5px';

            const label = document.createElement('span');
            label.textContent = `${item.brand} (${item.mode})`;
            label.style.fontSize = '12px';

            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legendContainer.appendChild(legendItem);
        });
    });

    // Add the legend below the chart
    canvas.parentNode.appendChild(legendContainer);
}

// Update Diagnostics Summary chart (Page Size vs Requests)
function updateDiagnosticsSummaryChart(filteredData) {
    const canvas = document.getElementById('diagnostics-summary-chart');
    if (!canvas) {
        console.error("Diagnostics summary chart canvas element not found");
        return;
    }

    const ctx = canvas.getContext('2d');

    // Clear previous chart if it exists
    if (window.diagnosticsSummaryChart) {
        window.diagnosticsSummaryChart.destroy();
    }

    // First try to use page_size_kb and request_count direct fields from filteredData
    console.log("Extracting diagnostics data from filtered data...");
    let bubbleData = extractPageSizeAndRequestsData(filteredData);

    // If no valid data found in the filtered dataset, try the dedicated file
    if (bubbleData.length === 0) {
        console.log("No diagnostics data found in main dataset, trying charts/diagnostics_data.json");

        // First try to look for the data in the 'charts' directory (static files generated by Python)
        fetch('charts/diagnostics_data.json')
            .then(response => {
                if (!response.ok) {
                    // If not found in charts directory, try the data directory
                    console.log("No diagnostics data in charts directory, trying data/diagnostics_data.json");
                    return fetch('data/diagnostics_data.json');
                }
                return response;
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to load diagnostics data: ${response.status} ${response.statusText}`);
                }
                return response.json();
            })
            .then(diagnosticsData => {
                console.log("Loaded diagnostics data:", diagnosticsData);
                if (Array.isArray(diagnosticsData) && diagnosticsData.length > 0) {
                    console.log(`Transforming ${diagnosticsData.length} diagnostics data points`);
                    const transformedData = transformDiagnosticsData(diagnosticsData);
                    createDiagnosticsChart(ctx, transformedData);
                } else {
                    console.warn("Diagnostics data is empty or not an array");
                    showNoDataMessage(ctx);
                }
            })
            .catch(error => {
                console.error("Error loading diagnostics data:", error);
                showNoDataMessage(ctx);

                // As a last resort, try generating synthetic data for demonstration
                const syntheticData = generateSyntheticDiagnosticsData();
                if (syntheticData.length > 0) {
                    console.log("Using synthetic diagnostics data for demonstration");
                    createDiagnosticsChart(ctx, syntheticData);
                }
            });
    } else {
        // Use the data we extracted from the filtered dataset
        console.log(`Using extracted diagnostics data: ${bubbleData.length} points`);
        createDiagnosticsChart(ctx, bubbleData);
    }
}

// Generate synthetic diagnostics data for demonstration if all else fails
function generateSyntheticDiagnosticsData() {
    // Only use this as a last resort when no real data is available
    console.log("Generating synthetic diagnostics data for demonstration");

    // Get unique brands from the dashboard state if available
    let brands = [];
    if (dashboardState.data && dashboardState.data.unique_brands) {
        brands = dashboardState.data.unique_brands;
    } else {
        // Default brands if not available
        brands = ['Brand A', 'Brand B', 'Brand C'];
    }

    const bubbleData = [];
    let colorIndex = 0;

    brands.forEach(brand => {
        // Generate desktop point
        const desktopHue = (colorIndex * 137) % 360;
        const desktopColor = `hsla(${desktopHue}, 70%, 50%, 0.7)`;
        colorIndex++;

        const desktopRequests = Math.floor(100 + Math.random() * 200); // 100-300 requests
        const desktopSize = 1500 + Math.random() * 3000; // 1500-4500 KB

        bubbleData.push({
            x: desktopRequests,
            y: desktopSize,
            r: Math.max(8, Math.min(25, 5 + (desktopSize / 200))),
            brand: brand,
            mode: 'desktop',
            color: desktopColor,
            borderColor: desktopColor.replace('0.7', '1')
        });

        // Generate mobile point
        const mobileHue = (colorIndex * 137) % 360;
        const mobileColor = `hsla(${mobileHue}, 70%, 50%, 0.7)`;
        colorIndex++;

        const mobileRequests = Math.floor(80 + Math.random() * 150); // 80-230 requests
        const mobileSize = 800 + Math.random() * 2000; // 800-2800 KB

        bubbleData.push({
            x: mobileRequests,
            y: mobileSize,
            r: Math.max(8, Math.min(25, 5 + (mobileSize / 200))),
            brand: brand,
            mode: 'mobile',
            color: mobileColor,
            borderColor: mobileColor.replace('0.7', '1')
        });
    });

    return bubbleData;
}

// Show a message when no data is available
function showNoDataMessage(ctx) {
    // Show a message in the chart area
    ctx.font = "14px Arial";
    ctx.fillStyle = "#666";
    ctx.textAlign = "center";
    ctx.fillText("No page size or request count data available.",
        ctx.canvas.width / 2, ctx.canvas.height / 2);
}

// Extract page size and request count data from the filtered data
function extractPageSizeAndRequestsData(filteredData) {
    if (!filteredData || filteredData.length === 0) {
        console.log("No filtered data provided to extractPageSizeAndRequestsData");
        return [];
    }

    console.log("Processing diagnostics data from", filteredData.length, "records");

    // Group by brand and mode
    const brandModeData = {};

    filteredData.forEach(item => {
        if (!item.brand) return;

        const key = `${item.brand}_${item.mode || 'unknown'}`;

        if (!brandModeData[key]) {
            brandModeData[key] = [];
        }

        brandModeData[key].push(item);
    });

    // Prepare bubble data for the chart
    const bubbleData = [];
    const colors = [];
    let colorIndex = 0;

    for (const [brandMode, items] of Object.entries(brandModeData)) {
        const [brand, mode] = brandMode.split('_');

        // Check for diagnostics_summary first and extract data if available
        const pageSizes = [];
        const requestCounts = [];

        items.forEach(item => {
            let pageSize = null;
            let requestCount = null;

            // Try direct page_size_kb and request_count fields first (these should be pre-extracted by Python)
            if (item.page_size_kb !== undefined && item.page_size_kb !== null) {
                pageSize = parseFloat(item.page_size_kb);
                console.log(`Found direct page_size_kb: ${pageSize} for ${brand}/${mode}`);
            }

            if (item.request_count !== undefined && item.request_count !== null) {
                requestCount = parseInt(item.request_count, 10);
                console.log(`Found direct request_count: ${requestCount} for ${brand}/${mode}`);
            }

            // If direct fields aren't available, try to extract from diagnostics_summary
            if ((pageSize === null || requestCount === null) && item.diagnostics_summary && typeof item.diagnostics_summary === 'string') {
                const diagnostics = item.diagnostics_summary;
                console.log(`Processing diagnostics summary: ${diagnostics} for ${brand}/${mode}`);

                // Extract page size - with more specific pattern matching for "Total Size: 3.82 MB" format
                if (pageSize === null) {
                    const sizePatterns = [
                        /Total Size:\s*([\d.]+)\s*([KkMmGg][Bb])/i,  // Match "Total Size: 3.82 MB" exactly
                        /Page Size:\s*([\d.]+)\s*([KkMmGg][Bb])/i,
                        /Transfer Size:\s*([\d.]+)\s*([KkMmGg][Bb])/i,
                        /Page Weight:\s*([\d.]+)\s*([KkMmGg][Bb])/i,
                        /([\d.]+)\s*([KkMmGg][Bb])/i  // Generic pattern as fallback
                    ];

                    for (const pattern of sizePatterns) {
                        const match = diagnostics.match(pattern);
                        if (match) {
                            const value = parseFloat(match[1]);
                            const unit = match[2].toLowerCase();

                            // Convert to KB
                            if (unit.startsWith('m')) {
                                pageSize = value * 1024;  // MB to KB
                                console.log(`Extracted page size from diagnostics: ${value} MB = ${pageSize} KB`);
                            } else if (unit.startsWith('k')) {
                                pageSize = value;
                                console.log(`Extracted page size from diagnostics: ${value} KB`);
                            } else if (unit.startsWith('g')) {
                                pageSize = value * 1024 * 1024;  // GB to KB
                                console.log(`Extracted page size from diagnostics: ${value} GB = ${pageSize} KB`);
                            } else {
                                pageSize = value / 1024;  // B to KB
                                console.log(`Extracted page size from diagnostics: ${value} B = ${pageSize} KB`);
                            }
                            break;
                        }
                    }
                }

                // Extract request count - with more specific pattern matching for "Requests: 293" format
                if (requestCount === null) {
                    const requestPatterns = [
                        /Requests:\s*(\d+)/i,  // Match "Requests: 293" exactly
                        /Total Requests:\s*(\d+)/i,
                        /Request Count:\s*(\d+)/i,
                        /Num Requests:\s*(\d+)/i
                    ];

                    for (const pattern of requestPatterns) {
                        const match = diagnostics.match(pattern);
                        if (match) {
                            requestCount = parseInt(match[1], 10);
                            console.log(`Extracted request count from diagnostics: ${requestCount}`);
                            break;
                        }
                    }
                }
            }

            // If still no page size, try alternative fields
            if (pageSize === null) {
                if (item.page_size !== undefined && item.page_size !== null) {
                    pageSize = parsePageSize(item.page_size);
                } else if (item.page_weight !== undefined && item.page_weight !== null) {
                    pageSize = parsePageSize(item.page_weight);
                } else if (item.total_byte_weight !== undefined && item.total_byte_weight !== null) {
                    pageSize = parsePageSize(item.total_byte_weight);
                } else if (item.transfer_size !== undefined && item.transfer_size !== null) {
                    pageSize = parsePageSize(item.transfer_size);
                }
            }

            // If still no request count, try alternative fields
            if (requestCount === null) {
                if (item.requests !== undefined && item.requests !== null) {
                    requestCount = parseRequestCount(item.requests);
                } else if (item.num_requests !== undefined && item.num_requests !== null) {
                    requestCount = parseRequestCount(item.num_requests);
                }
            }

            // Add valid values to arrays
            if (pageSize !== null && !isNaN(pageSize) && pageSize > 0) {
                pageSizes.push(pageSize);
            }

            if (requestCount !== null && !isNaN(requestCount) && requestCount > 0) {
                requestCounts.push(requestCount);
            }
        });

        // Calculate averages
        const avgPageSize = pageSizes.length > 0 ?
            pageSizes.reduce((sum, size) => sum + size, 0) / pageSizes.length : 0;

        const avgRequests = requestCounts.length > 0 ?
            requestCounts.reduce((sum, count) => sum + count, 0) / requestCounts.length : 0;

        console.log(`${brand}/${mode}: Avg Page Size=${avgPageSize.toFixed(2)} KB, Avg Requests=${avgRequests.toFixed(1)}, Sample sizes: ${pageSizes.length}/${requestCounts.length}`);

        // Only add datapoint if we have valid data
        if (avgPageSize > 0 && avgRequests > 0) {
            // Generate a color for this brand/mode
            const hue = (colorIndex * 137) % 360;
            const color = `hsla(${hue}, 70%, 50%, 0.7)`;
            colors.push(color);
            colorIndex++;

            bubbleData.push({
                x: avgRequests,
                y: avgPageSize,
                r: Math.max(8, Math.min(25, 5 + (avgPageSize / 200))), // Bubble size based on page size (min: 8, max: 25)
                brand: brand,
                mode: mode,
                color: color,
                borderColor: color.replace('0.7', '1')
            });
        }
    }

    console.log(`Generated ${bubbleData.length} data points for diagnostics chart`);
    return bubbleData;
}

// Helper function to parse page size from various formats
function parsePageSize(value) {
    if (typeof value === 'number') {
        // If value is already a number, handle different units based on magnitude
        if (value > 1000000) return value / 1024; // Convert B to KB
        else if (value > 1000) return value; // Likely already KB
        else if (value < 10) return value * 1024 * 1024; // Likely MB or smaller unit
        return value; // Assume KB
    }
    else if (typeof value === 'string') {
        value = value.toLowerCase();
        const numValue = parseFloat(value.replace(/[^\d.]/g, ''));

        if (isNaN(numValue)) return null;

        if (value.includes('mb')) return numValue * 1024; // MB to KB
        else if (value.includes('kb')) return numValue;
        else if (value.includes('b') && !value.includes('kb')) return numValue / 1024; // B to KB
        return numValue; // Assume KB if no unit specified
    }
}

// Helper function to parse request count
function parseRequestCount(value) {
    if (typeof value === 'number') {
        return value;
    }
    else if (typeof value === 'string') {
        const numValue = parseInt(value.replace(/[^\d]/g, ''), 10);
        return isNaN(numValue) ? null : numValue;
    }
    return null;
}

// Helper functions for styling based on metric values
function getScoreClass(score) {
    if (score >= 90) return 'score-good';
    if (score >= 50) return 'score-average';
    return 'score-poor';
}

function getLCPClass(lcp) {
    if (lcp <= 2.5) return 'score-good';
    if (lcp <= 4.0) return 'score-average';
    return 'score-poor';
}

function getFCPClass(fcp) {
    if (fcp <= 1.8) return 'score-good';
    if (fcp <= 3.0) return 'score-average';
    return 'score-poor';
}

function getCLSClass(cls) {
    if (cls <= 0.1) return 'score-good';
    if (cls <= 0.25) return 'score-average';
    return 'score-poor';
}

function getINPClass(inp) {
    if (inp <= 200) return 'score-good';
    if (inp <= 500) return 'score-average';
    return 'score-poor';
}
