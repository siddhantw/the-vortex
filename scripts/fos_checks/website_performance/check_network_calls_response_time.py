import argparse
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Ensure the screenshots directory exists
os.makedirs("network_screenshots", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_network_calls(driver, url, brand_url, depth):
    logging.info(f"Fetching network calls for URL: {url} at depth {depth}")
    try:
        for attempt in range(3):  # Retry mechanism
            try:
                driver.get(url)
                time.sleep(2)
                performance_logs = driver.execute_script("return window.performance.getEntries()")
                logging.info(f"Fetched {len(performance_logs)} performance logs for URL: {url}")
                network_calls = []

                for entry in performance_logs:
                    if entry['entryType'] == 'resource':
                        response_time = entry.get('responseEnd', 0) - entry.get('startTime', 0)
                        breakdown = {
                            "Network Latency": entry.get('connectEnd', 0) - entry.get('connectStart', 0),
                            "Server Processing Time": entry.get('responseStart', 0) - entry.get('requestStart', 0),
                            "Database Queries": 0,
                            "Asynchronous Operations": 0
                        }
                        # Fetch CF-Cache-Status
                        cf_cache_status = "Unknown"
                        try:
                            driver.execute_script("window.open('');")
                            driver.switch_to.window(driver.window_handles[1])
                            driver.get(entry.get('name', ''))
                            headers = driver.execute_script("""
                                var req = new XMLHttpRequest();
                                req.open('GET', document.location, false);
                                req.send(null);
                                return req.getAllResponseHeaders();
                            """)
                            for header in headers.split('\n'):
                                if 'cf-cache-status' in header.lower():
                                    cf_cache_status = header.split(':')[1].strip()
                                    break
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                        except Exception:
                            pass

                        BRAND_NAME_MAPPING = {
                            "networksolutions.com": "Network Solutions",
                            "bluehost.com": "Bluehost",
                            "domain.com": "Domain",
                            "hostgator.com": "HostGator",
                            "register.com": "Register",
                            "web.com": "Web"
                        }
                        normalized_brand_url = urlparse(brand_url).netloc.replace("www.", "")
                        brand_name = BRAND_NAME_MAPPING.get(normalized_brand_url, "Unknown")
                        network_calls.append({
                            "Brand": brand_name,
                            "WebPage URL": url,
                            "Depth": depth,
                            "Network Call": entry.get('name', ''),
                            "CF-Cache-Status": cf_cache_status,
                            "Response Time": response_time,
                            "Breakdown": breakdown
                        })
                return network_calls
            except Exception as e:
                logging.warning(f"Retrying {url} due to error: {e}")
                if attempt == 2:
                    raise
                return None
        return None
    except Exception as e:
        logging.error(f"Error fetching network calls from {url}: {e}")
        return []


def crawl_website_for_network_calls(brand_url, max_depth=1, max_workers=4):
    logging.info(f"Starting crawl for brand URL: {brand_url} with max depth {max_depth}")
    start_time = time.time()
    firefox_options = FirefoxOptions()
    firefox_options.add_argument('--headless')
    firefox_options.add_argument('--disable-gpu')
    firefox_options.add_argument('--disable-dev-shm-usage')
    firefox_options.add_argument('--no-sandbox')

    try:
        driver = WebDriver(service=FirefoxService(), options=firefox_options)
    except Exception as e:
        print(f"Error initializing FirefoxDriver: {e}")
        return []

    driver.set_page_load_timeout(180)
    visited_urls = set()
    network_calls = []
    urls_to_visit = [(brand_url, 0)]

    def process_url(current_url, depth):
        if depth > max_depth or current_url in visited_urls:
            return [], []
        visited_urls.add(current_url)
        print(f"Visiting: {current_url} at depth {depth}")
        try:
            calls = fetch_network_calls(driver, current_url, brand_url, depth)
            driver.get(current_url)
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            new_links = [
                (link['href'], depth + 1)
                for link in soup.find_all('a', href=True)
                if urlparse(link['href']).netloc == urlparse(brand_url).netloc
            ]
            return calls, new_links
        except Exception as e:
            print(f"Error processing {current_url}: {e}")
            return [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_url, url, depth): (url, depth) for url, depth in urls_to_visit}
        while futures:
            for future in as_completed(futures):
                try:
                    calls, new_links = future.result()
                    network_calls.extend(calls)
                    for link in new_links:
                        if len(link) == 2:
                            urls_to_visit.append(link)
                    futures.update({executor.submit(process_url, url, depth): (url, depth) for url, depth in new_links})
                except Exception as e:
                    print(f"Error in thread: {e}")
                finally:
                    futures.pop(future, None)
    logging.info(f"Finished crawling {brand_url} in {time.time() - start_time:.2f} seconds")
    driver.quit()
    return network_calls


def main():
    logging.info("Starting the network calls report generation")
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Crawl websites and generate a network calls report.")
    parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth for crawling (default: 2)")
    args = parser.parse_args()

    brand_urls = [
        "https://www.bluehost.com/",
        "https://www.domain.com/",
        "https://www.hostgator.com/",
        "https://www.networksolutions.com/",
        "https://www.register.com/",
        "https://www.web.com/"
    ]
    max_depth = args.max_depth
    all_network_calls = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(crawl_website_for_network_calls, url, max_depth): url for url in brand_urls}
        for future in as_completed(futures):
            all_network_calls.extend(future.result())

    df = pd.DataFrame(all_network_calls)
    if df.empty:
        print("No network calls were captured.")
        return

    html_output_path = "network_calls_report.html"
    with open(html_output_path, "w") as file:
        file.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FOS Report - Response Time for Network Calls across Jarvis Brands</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.4.0/jspdf.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .filter-container {
            margin-bottom: 20px;
            text-align: center;
        }
        .filter-container select {
            padding: 5px;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        .dark-green { background-color: #006400; color: white; }
        .yellow { background-color: #FFFF00; color: black; }
        .orange { background-color: #FFA500; color: black; }
        .dark-red { background-color: #8B0000; color: white; }
    </style>
</head>
<body>
    <h1>FOS Report - Response Time for Network Calls across Jarvis Brands</h1>
    <div class="filter-container">
        <label for="brandFilter">Filter by Brand:</label>
        <select id="brandFilter">
            <option value="">All</option>
        </select>
        <label for="cacheFilter">Filter by CF-Cache-Status:</label>
        <select id="cacheFilter">
            <option value="">All</option>
        </select> <br/><br/>
        <label for="responseTimeFilter">Filter by Response Time:</label>
        <select id="responseTimeFilter">
            <option value="">All</option>
            <option value="low">Less than 100ms</option>
            <option value="medium">100-200ms</option>
            <option value="high">200-1000ms</option>
            <option value="very-high">More than 1000ms</option>
        </select>
        <button id="exportPdf">Export to PDF</button>
        <button id="exportExcel">Export to Excel</button>
    </div>
    <table id="networkCallsReport" class="display" style="width:100%">
        <thead>
            <tr>
                <th>Brand</th>
                <th>WebPage URL</th>
                <th>Depth</th>
                <th>Network Call</th>
                <th>CF-Cache-Status</th>
                <th>Response Time (ms)</th>
                <th>Network Latency (ms)</th>
                <th>Server Processing Time (ms)</th>
                <th>Database Queries (ms)</th>
                <th>Asynchronous Operations (ms)</th>
            </tr>
        </thead>
        <tbody>
""")
        for _, row in df.iterrows():
            response_time = row['Response Time']
            if response_time < 100:
                color_class = "#006400"
            elif 100 <= response_time < 200:
                color_class = "#FFFF00"
            elif 200 <= response_time <= 1000:
                color_class = "#FFA500"
            else:
                color_class = "#8B0000"

            file.write(f"""
                    <tr style="background-color: {color_class};">
                        <td>{row['Brand']}</td>
                        <td><a href="{row['WebPage URL']}" target="_blank">{row['WebPage URL'][:100]}{'...' if len(row['WebPage URL']) > 100 else ''}</a></td>
                        <td>{row['Depth']}</td>
                        <td><a href="{row['Network Call']}" target="_blank">{row['Network Call'][:100]}{'...' if len(row['Network Call']) > 100 else ''}</a></td>
                        <td>{row['CF-Cache-Status']}</td>
                        <td>{response_time:.2f}</td>
                        <td>{row['Breakdown']['Network Latency']:.2f}</td>
                        <td>{row['Breakdown']['Server Processing Time']:.2f}</td>
                        <td>{row['Breakdown']['Database Queries']:.2f}</td>
                        <td>{row['Breakdown']['Asynchronous Operations']:.2f}</td>
                    </tr>
            """)
        file.write("""
        </tbody>
    </table>
    <script>
        $(document).ready(function() {
            const table = $('#networkCallsReport').DataTable({
                "paging": true,
                "searching": true,
                "ordering": true,
                "dom": 'Bfrtip',
                "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                "pageLength": 25, // Default number of rows per page
                "buttons": [
                    'pageLength' // Add this to enable the dropdown for entries per page
                ]
            });

            var brandSet = new Set();
            var cacheStatusSet = new Set();

            table.rows().every(function() {
                var data = this.data();
                brandSet.add(data[0]);
                cacheStatusSet.add(data[4]);
            });

            brandSet.forEach(function(brand) {
                $('#brandFilter').append('<option value="' + brand + '">' + brand + '</option>');
            });

            cacheStatusSet.forEach(function(status) {
                $('#cacheFilter').append('<option value="' + status + '">' + status + '</option>');
            });

            // Custom filter for response time
            $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
                const selectedCategory = $('#responseTimeFilter').val();
                const responseTime = parseFloat(data[5]) || 0; // Column 5 is Response Time
            
                if (selectedCategory === "low" && responseTime < 100) return true;
                if (selectedCategory === "medium" && responseTime >= 100 && responseTime < 200) return true;
                if (selectedCategory === "high" && responseTime >= 200 && responseTime <= 1000) return true;
                if (selectedCategory === "very-high" && responseTime > 1000) return true;
                if (selectedCategory === "") return true;
        
                return false;
            });
        
            // Trigger filter on dropdown change
            $('#responseTimeFilter').on('change', function() {
                table.draw();
            });

            $('#brandFilter').on('change', function() {
                var searchTerm = $(this).val();
                table.column(0).search(searchTerm).draw();
            });

            $('#cacheFilter').on('change', function() {
                var searchTerm = $(this).val();
                table.column(4).search(searchTerm).draw();
            });
    
            // Export to PDF
            document.getElementById('exportPdf').addEventListener('click', function() {
                // Trigger browser's print dialog
                window.print();
            });
    
            // Export to Excel
            document.getElementById('exportExcel').addEventListener('click', function() {
                const tableElement = document.getElementById('networkCallsReport');
                const workbook = XLSX.utils.table_to_book(tableElement, { sheet: "Sheet1" });
                XLSX.writeFile(workbook, 'network_calls_report.xlsx');
            });
        });
    </script>
</body>
</html>
""")
    logging.info(f"Report generation completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Report saved at: {html_output_path}")


if __name__ == "__main__":
    main()
