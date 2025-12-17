from time import time

import requests


def measure_resource_performance(url):
    try:
        start = time()
        response = requests.get(url)
        load_time = time() - start
        size_kb = len(response.content) / 1024

        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Size: {size_kb:.2f} KB")
    except requests.exceptions.RequestException as e:
        print(f"Failed to load {url}: {e}")


# Example usage
measure_resource_performance("https://www.bluehost.com/")
