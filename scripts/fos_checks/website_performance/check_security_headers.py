import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_security_headers(url):
    """
    Fetches and checks the security headers of a given URL.

    Args:
        url (str): The URL to check.

    Returns:
        dict: A dictionary containing the URL and its security headers.
    """
    # Send a GET request to the URL
    response = requests.get(url)
    # Extract headers from the response
    headers = response.headers
    # List of required security headers to check
    required_headers = [
        "Content-Security-Policy", "Strict-Transport-Security", "X-Frame-Options",
        "X-Content-Type-Options", "Referrer-Policy"
    ]
    # Initialize the result dictionary with the URL
    result = {"URL": url}
    # Check for the presence of each required header
    for header in required_headers:
        result[header] = headers.get(header, "Not Present")
    return result

# List of brand URLs to check for security headers
brand_urls = [
    "https://www.bluehost.com/",
    "https://www.domain.com/",
    "https://www.hostgator.com/",
    "https://www.networksolutions.com/",
    "https://www.register.com/",
    "https://www.web.com/"
]

# Parallel execution of security header checks
results = []
with ThreadPoolExecutor(max_workers=6) as executor:
    # Submit tasks to the thread pool for each URL
    futures = {executor.submit(check_security_headers, url): url for url in brand_urls}
    # Collect results as tasks complete
    for future in as_completed(futures):
        results.append(future.result())

# Path to save the generated HTML report
html_output_path = "security_headers_report.html"
with open(html_output_path, "w") as file:
    # Write the initial HTML structure and include DataTables CSS and JS
    file.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Headers Report</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
</head>
<body>
    <h1>Security Headers Report</h1>
    <table id="securityHeadersReport" class="display" style="width:100%">
        <thead>
            <tr>
                <th>URL</th>
                <th>Content-Security-Policy</th>
                <th>Strict-Transport-Security</th>
                <th>X-Frame-Options</th>
                <th>X-Content-Type-Options</th>
                <th>Referrer-Policy</th>
            </tr>
        </thead>
        <tbody>
""")
    # Populate the table with results for each URL
    for result in results:
        file.write(f"""
            <tr>
                <td>{result['URL']}</td>
                <td>{result['Content-Security-Policy']}</td>
                <td>{result['Strict-Transport-Security']}</td>
                <td>{result['X-Frame-Options']}</td>
                <td>{result['X-Content-Type-Options']}</td>
                <td>{result['Referrer-Policy']}</td>
            </tr>
""")
    # Close the table and include DataTables initialization script
    file.write("""
        </tbody>
    </table>
    <script>
        // Initialize DataTables for the report table
        $(document).ready(function() {
            $('#securityHeadersReport').DataTable({
                "paging": true,
                "searching": true,
                "ordering": true
            });
        });
    </script>
</body>
</html>
""")

# Print the location of the generated HTML report
print(f"HTML report generated at: {html_output_path}")