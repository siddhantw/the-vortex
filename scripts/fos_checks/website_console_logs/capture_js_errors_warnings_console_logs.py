import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

# Ensure the screenshots directory exists
os.makedirs("screenshots", exist_ok=True)


def get_all_links(driver, url, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            return links
        except Exception as e:
            print(f"Error fetching links from {url} (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(backoff)
            backoff *= 2
    return []


def get_console_logs(driver):
    try:
        logs = driver.get_log('browser')
        return [
            {
                "level": entry['level'],
                "message": entry['message'],
                "timestamp": entry['timestamp']
            } for entry in logs if entry['level'] in ['SEVERE', 'WARNING']
        ]
    except WebDriverException as e:
        print(f"Error fetching console logs: {e}")
        return []

def sanitize_filename(s):
    """ Sanitise the string to create a valid filename. """
    return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in s)

def crawl_website_for_js_errors(brand_url, max_depth=1):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

    with tempfile.TemporaryDirectory(prefix=f"chrome_user_data_{os.getpid()}_") as user_data_dir:
        chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
        driver = webdriver.Chrome(options=chrome_options)

        visited_urls = set()
        js_errors = []
        urls_to_visit = [(brand_url, 0)]
        brand_name = urlparse(brand_url).netloc.split('.')[1].capitalize()
        if brand_name.lower() == "networksolutions":
            brand_name = "Network Solutions"

        start_time = time.time()

        try:
            while urls_to_visit:
                current_url, depth = urls_to_visit.pop(0)
                if depth > max_depth or current_url in visited_urls:
                    continue

                visited_urls.add(current_url)
                print(f"Visiting: {current_url} at depth {depth}")

                try:
                    driver.get(current_url)
                    time.sleep(2)

                    logs = get_console_logs(driver)
                    if logs:
                        screenshot_filename = sanitize_filename(urlparse(current_url).path or 'homepage') + ".png"
                        screenshot_path = os.path.join("screenshots", screenshot_filename)
                        driver.save_screenshot(screenshot_path)

                        for log in logs:
                            js_errors.append({
                                "Brand": brand_name,
                                "URL": current_url,
                                "Depth": depth,
                                "Level": log['level'],
                                "Message": log['message'],
                                "Timestamp": log['timestamp'],
                                "Screenshot": screenshot_path
                            })
                    else:
                        print(f"No JS errors found on {current_url}")

                    links = get_all_links(driver, current_url)
                    for link in links:
                        parsed_link = urlparse(link)
                        if parsed_link.netloc == urlparse(brand_url).netloc and link not in visited_urls:
                            urls_to_visit.append((link, depth + 1))

                except Exception as e:
                    print(f"Error processing {current_url}: {e}")

                time.sleep(1)

        finally:
            driver.quit()
            print(f"Crawling completed for {brand_name} in {time.time() - start_time:.2f} seconds.")

        return js_errors


def main():
    parser = argparse.ArgumentParser(description="Crawl websites for JavaScript errors and warnings.")
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
    all_js_errors = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(crawl_website_for_js_errors, url, max_depth): url for url in brand_urls}
        for future in as_completed(futures):
            all_js_errors.extend(future.result())

    df = pd.DataFrame(all_js_errors)
    html_output_path = "js_errors_and_warnings_report.html"
    with open(html_output_path, "w") as file:
        file.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FOS Report - JS Console Logs Errors and Warnings across Jarvis Brands</title>
            <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css">
            <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css">
            <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
            <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
            <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
            <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.html5.min.js"></script>
            <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.print.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                #levelFilter {
                    margin-bottom: 20px;
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
                    border: 1px solid #ddd.
                }
                th {
                    background-color: #f4f4f4;
                }
            </style>
        </head>
        <body>
            <h1>FOS Report - JS Console Logs Errors and Warnings across Jarvis Brands</h1>
            <label for="levelFilter">Filter by Level:</label>
            <select id="levelFilter">
                <option value="">All</option>
                <option value="SEVERE">SEVERE</option>
                <option value="WARNING">WARNING</option>
            </select>
            <table id="jsErrorsReport" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Brand</th>
                        <th>URL</th>
                        <th>Depth</th>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Timestamp</th>
                        <th>Screenshot</th>
                    </tr>
                </thead>
                <tbody>
        """)
        for _, row in df.iterrows():
            color = "#FF0000" if row['Level'] == "SEVERE" else "#FFFF00" if row['Level'] == "WARNING" else "#FFFFFF"
            readable_timestamp = pd.to_datetime(row['Timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"""
                    <tr style="background-color: {color};" data-level="{row['Level']}">
                        <td>{row['Brand']}</td>
                        <td><a href="{row['URL']}" target="_blank">{row['URL'][:100]}{'...' if len(row['URL']) > 100 else ''}</a></td>
                        <td>{row['Depth']}</td>
                        <td>{row['Level']}</td>
                        <td>{row['Message']}</td>
                        <td>{readable_timestamp}</td>
                        <td><a href="{row['Screenshot']}" target="_blank">View Screenshot</a></td>
                    </tr>
            """)
        file.write("""
                </tbody>
            </table>
            <script>
                $(document).ready(function() {
                    const table = $('#jsErrorsReport').DataTable({
                        "paging": true,
                        "searching": true,
                        "ordering": true,
                        "dom": 'Bfrtip', // Ensure buttons are displayed
                        "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                        "pageLength": 25, // Default number of rows per page
                        "buttons": [
                            'pageLength', // Add dropdown for entries per page
                            {
                                extend: 'excelHtml5',
                                text: 'Export to Excel',
                                className: 'exportButton'
                            },
                            {
                                extend: 'pdfHtml5',
                                text: 'Export to PDF',
                                className: 'exportButton'
                            },
                            {
                                extend: 'print',
                                text: 'Export to PDF',
                                className: 'exportButton'
                            }
                        ]
                    });

                    $('#levelFilter').on('change', function() {
                        const selectedLevel = $(this).val();
                        table.column(3).search(selectedLevel).draw();
                    });
                });
            </script>
        </body>
        </html>
        """)

    print(f"HTML report generated at: {html_output_path}")


if __name__ == "__main__":
    main()
