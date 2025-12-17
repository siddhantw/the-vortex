import re

import pandas as pd

# Read the file
file_path = 'website_urls_crawled_with_depth_and_status_code.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract URL, depth, and status using regex
data = []
pattern = r"(https?://[^\s]+) at depth (\d+) - (\d+)"
for line in lines:
    match = re.match(pattern, line)
    if match:
        url, depth, status = match.groups()
        data.append({
            "URL": url,
            "Depth Level": int(depth),
            "HTTP Status Code": int(status)
        })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_path = 'website_urls_crawled_with_depth_and_status_code.xlsx'
df.to_excel(output_path, index=False)

print(f"Excel file saved at: {output_path}")
