import asyncio
from pyppeteer import launch
import json
import os
from urllib.parse import urljoin, urlparse

# Ensure the reports directory exists
os.makedirs("performance_reports", exist_ok=True)


async def get_metrics(url, browser):
    page = await browser.newPage()
    await page.goto(url)
    await page.waitForSelector('body')

    # Capture metrics using the Performance API
    metrics = await page.evaluate('''() => {
        const { timing } = performance;
        const loadTime = timing.loadEventEnd - timing.navigationStart;
        const domContentLoadedTime = timing.domContentLoadedEventEnd - timing.navigationStart;

        // Capture largest-contentful-paint
        const lcpEntry = performance.getEntriesByName('largest-contentful-paint')[0];
        const lcp = lcpEntry?.startTime || null;

        // Calculate Cumulative Layout Shift (CLS)
        let cls = 0;
        performance.getEntriesByType('layout-shift').forEach(entry => {
            if (!entry.hadRecentInput) cls += entry.value;
        });

        // First Contentful Paint (FCP)
        const fcpEntry = performance.getEntriesByName('first-contentful-paint')[0];
        const fcp = fcpEntry?.startTime || null;

        // Interaction to Next Paint (INP) approximation
        let inp = null;
        const interactions = performance.getEntriesByType('event');
        if (interactions.length > 0) {
            inp = Math.max(...interactions.map(event => event.processingStart - event.startTime));
        }

        return {
            url: window.location.href,
            loadTime,
            domContentLoadedTime,
            lcp,
            cls,
            fcp,
            inp
        };
    }''')

    return metrics


async def crawl_performance(start_url, max_depth=1):
    browser = await launch(headless=True)
    visited_urls = set()
    urls_to_visit = [(start_url, 0)]
    performance_data = []

    while urls_to_visit:
        current_url, depth = urls_to_visit.pop(0)
        if depth > max_depth or current_url in visited_urls:
            continue

        print(f"Visiting {current_url} at depth {depth}")
        visited_urls.add(current_url)

        try:
            metrics = await get_metrics(current_url, browser)
            performance_data.append(metrics)
            print(f"Metrics collected for {current_url}")

            # Extract links for deeper analysis
            page = await browser.newPage()
            await page.goto(current_url)
            links = await page.evaluate('''() => {
                return Array.from(document.querySelectorAll('a'))
                    .map(a => a.href)
                    .filter(href => href.startsWith('http'));
            }''')

            # Add valid links to the queue for crawling
            for link in links:
                if urlparse(link).netloc == urlparse(start_url).netloc and link not in visited_urls:
                    urls_to_visit.append((link, depth + 1))

        except Exception as e:
            print(f"Error visiting {current_url}: {e}")

    await browser.close()

    # Save performance data to JSON
    with open("performance_reports/performance_data.json", "w") as f:
        json.dump(performance_data, f, indent=4)

    print(f"Performance data saved to 'performance_reports/performance_data.json'")


# Run the function
asyncio.run(crawl_performance("https://www.bluehost.com/", max_depth=1))
