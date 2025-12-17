"""
This script performs a multi-threaded web crawling operation for a list of brand URLs.
It captures all internal links up to a specified depth, along with their HTTP status codes,
response times, and other metadata. The results are saved in an Excel file and an HTML report.

Modules Used:
- time: For measuring response times and adding delays.
- datetime: For timestamping the last update.
- urllib.parse: For URL parsing and joining.
- pandas: For data manipulation and saving results to Excel.
- requests: For making HTTP requests.
- bs4 (BeautifulSoup): For parsing HTML content.
- concurrent.futures: For multi-threaded execution.

Functions:
- get_all_links(url): Fetches all links from a given URL.
- crawl_website(start_url, brand, max_depth): Crawls a website up to a specified depth.
- crawl_brand(brand_url, max_depth): Initiates crawling for a specific brand URL.
"""

import argparse
import time
import random
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import Timeout, ConnectionError

# Configure logging
logging.basicConfig(
    filename="web_crawler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# List of User-Agents for random selection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    # Add more User-Agents as needed
]

# Configurable delay between requests
REQUEST_DELAY = 1  # seconds

def get_all_links(url, retries=3):
    """
    Fetches all links from a given URL with retry mechanism and extracts hreflang attributes.

    Args:
        url (str): The URL to fetch links from.
        retries (int): Number of retries for failed requests.

    Returns:
        tuple: A list of links, HTTP status code, response time in seconds, and hreflang data.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=5)
            status_code = response.status_code
            response_time = round(time.time() - start_time, 2)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract regular links
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]

            # Extract hreflang attributes from link elements
            hreflang_data = []

            # Check for hreflang in <link> elements (most common)
            for link_elem in soup.find_all('link', hreflang=True):
                hreflang_data.append({
                    'hreflang': link_elem.get('hreflang'),
                    'href': urljoin(url, link_elem.get('href', '')),
                    'rel': link_elem.get('rel', []),
                    'type': 'link_element'
                })

            # Check for hreflang in <a> elements (alternative implementation)
            for a_elem in soup.find_all('a', hreflang=True):
                hreflang_data.append({
                    'hreflang': a_elem.get('hreflang'),
                    'href': urljoin(url, a_elem.get('href', '')),
                    'text': a_elem.get_text(strip=True),
                    'type': 'anchor_element'
                })

            # Format hreflang data for display
            hreflang_summary = []
            if hreflang_data:
                for item in hreflang_data:
                    lang = item['hreflang']
                    href = item['href']
                    elem_type = item['type']
                    hreflang_summary.append(f"{lang}: {href} ({elem_type})")

            hreflang_string = "; ".join(hreflang_summary) if hreflang_summary else "None"

            return links, status_code, response_time, hreflang_string

        except Timeout:
            logging.warning(f"Timeout error for {url} on attempt {attempt + 1}")
        except ConnectionError:
            logging.warning(f"Connection error for {url} on attempt {attempt + 1}")
        except requests.RequestException as e:
            logging.error(f"Request error for {url}: {e}")
        time.sleep(REQUEST_DELAY)
    logging.error(f"Failed to fetch {url} after {retries} retries")
    return [], None, None, "Error fetching page"


def crawl_website(start_url, brand, max_depth=3):
    """
    Crawls a website starting from a given URL up to a specified depth.

    Args:
        start_url (str): The starting URL for the crawl.
        brand (str): The brand name or URL being crawled.
        max_depth (int): The maximum depth to crawl.

    Returns:
        list: A list of dictionaries containing metadata for each crawled URL.
    """
    visited_urls = set()
    url_data = []
    urls_to_visit = [(start_url, 0)]
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    while urls_to_visit:
        current_url, depth = urls_to_visit.pop(0)
        if depth > max_depth:
            continue

        if current_url not in visited_urls:
            visited_urls.add(current_url)
            logging.info(f"Crawling {current_url} at depth {depth}")
            links, status_code, response_time, hreflang = get_all_links(current_url)

            if status_code:
                url_data.append({
                    "Brand URL": brand,
                    "URLs Captured": current_url,
                    "Depth": depth,
                    "HTTP Status Code": status_code,
                    "Response Time (s)": response_time,
                    "Last Updated": last_updated,
                    "Hreflang": hreflang
                })
                print(f"{current_url} at depth {depth} - {status_code} - {response_time}s - Hreflang: {hreflang}")
            else:
                logging.warning(f"Failed to fetch metadata for {current_url}")

            for link in links:
                parsed_link = urlparse(link)
                if parsed_link.netloc == urlparse(start_url).netloc and link not in visited_urls:
                    urls_to_visit.append((link, depth + 1))

        time.sleep(REQUEST_DELAY)  # Use configurable delay

    return url_data


def crawl_brand(brand_url, max_depth):
    """
    Initiates crawling for a specific brand URL.

    Args:
        brand_url (str): The brand's base URL.
        max_depth (int): The maximum depth to crawl.

    Returns:
        tuple: The brand name and the crawled data.
    """
    brand_name = urlparse(brand_url).netloc.split('.')[1].capitalize()
    print(f"\nStarting crawl for {brand_name}...")
    return brand_name, crawl_website(brand_url, brand_url, max_depth)


if __name__ == "__main__":
    """
    Main execution block. Crawls a list of brand URLs in parallel, saves the results
    to an Excel file, and generates an HTML report.
    """
    parser = argparse.ArgumentParser(description="Website Crawler Script")
    parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth for crawling (default: 2)")
    args = parser.parse_args()

    max_depth = args.max_depth  # Use runtime input or default value

    brand_urls = [
        "https://www.bluehost.com/",
        "https://www.domain.com/",
        "https://www.hostgator.com/",
        "https://www.networksolutions.com/"
    ]
    all_data = {}

    logging.info("Starting website crawler")
    try:
        # Use ThreadPoolExecutor for parallel crawling
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_brand = {executor.submit(crawl_brand, url, max_depth): url for url in brand_urls}
            for future in as_completed(future_to_brand):
                brand_name, data = future.result()
                all_data[brand_name] = data

        # Save all data to a single Excel file with separate sheets
        with pd.ExcelWriter("website_urls_crawled_with_depth_and_status_code.xlsx") as writer:
            for brand_name, data in all_data.items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=brand_name, index=False)

        print(f"\nCrawling completed. Data saved to 'website_urls_crawled_with_depth_and_status_code.xlsx'")
        logging.info("Crawling completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during crawling: {e}")

    # Generate an HTML report with colour-coded rows and export options
    html_output_path = "website_crawler_report.html"
    with open(html_output_path, "w") as file:
        file.write("""
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>FOS Report - Web Crawler for Broken Links across Jarvis Brands</title>
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
                    #statusFilter {
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
                        border: 1px solid #ddd;
                    }
                    th {
                        background-color: #f4f4f4;
                    }
                </style>
            </head>
            <body>
                <h1>FOS Report - Web Crawler for Broken Links across Jarvis Brands</h1>
                <label for="statusFilter">Filter by Status Code:</label>
                <select id="statusFilter">
                    <option value="">All</option>
                    <option value="2">2xx (Success)</option>
                    <option value="3">3xx (Redirection)</option>
                    <option value="4">4xx (Client Error)</option>
                    <option value="5">5xx (Server Error)</option>
                </select>
                <table id="crawlerReport" class="display" style="width:100%">
                    <thead>
                        <tr>
                            <th>Brand URL</th>
                            <th>URLs Captured</th>
                            <th>Depth</th>
                            <th>HTTP Status Code</th>
                            <th>Response Time (s)</th>
                            <th>Hreflang</th>
                            <th>Last Updated</th>
                        </tr>
                    </thead>
                    <tbody>
        """)
        for brand_name, data in all_data.items():
            for row in data:
                status_code = row['HTTP Status Code']
                # Determine the inline style based on the HTTP status code
                if status_code >= 400:
                    color_style = "background-color: #FF0000; color: white;"  # Broken link
                elif 300 <= status_code < 400:
                    color_style = "background-color: #FFFF00; color: black;"  # Redirection
                else:
                    color_style = "background-color: #006400; color: white;"  # Not broken

                file.write(f"""
                <tr style="{color_style}" data-status="{str(status_code)[0]}">
                    <td>{row['Brand URL']}</td>
                    <td><a href="{row['URLs Captured']}" target="_blank">{row['URLs Captured'][:100]}{'...' if len(row['URLs Captured']) > 100 else ''}</a></td>
                    <td>{row['Depth']}</td>
                    <td>{status_code}</td>
                    <td>{row['Response Time (s)']}</td>
                    <td>{row['Hreflang']}</td>
                    <td>{row['Last Updated']}</td>
                </tr>
        """)
        file.write("""
                    </tbody>
                </table>
                <script>
                    $(document).ready(function() {
                        const table = $('#crawlerReport').DataTable({
                            "paging": true,
                            "searching": true,
                            "ordering": true,
                            "dom": 'Bfrtip', // Ensure buttons are displayed
                            "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                            "pageLength": 25, // Default number of rows per page
                            "buttons": [
                                'pageLength', // Dropdown for entries per page
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
                
                        $('#statusFilter').on('change', function() {
                            const selectedStatus = $(this).val();
                            table.rows().every(function() {
                                const row = this.node();
                                const status = $(row).attr('data-status'); // Correctly access the data-status attribute
                                if (selectedStatus === "" || status.startsWith(selectedStatus)) {
                                    $(row).show();
                                } else {
                                    $(row).hide();
                                }
                            });
                            table.draw(); // Redraw the table after filtering
                        });
                    });
                </script>
        </body>
        </html>
        """)

    print(f"HTML report with inline color-coded rows generated at: {html_output_path}")
    logging.info("Crawling completed successfully")
