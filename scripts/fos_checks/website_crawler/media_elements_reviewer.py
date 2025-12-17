"""
This script performs a multi-threaded web crawling operation for a list of brand URLs
to identify and analyze images on webpages. It captures image metadata like size, dimensions,
loading type, and section where the image appears. The results are saved in an Excel file
and an interactive HTML report.

Modules Used:
- time: For measuring response times and adding delays.
- datetime: For timestamping the last update.
- urllib.parse: For URL parsing and joining.
- pandas: For data manipulation and saving results to Excel.
- requests: For making HTTP requests.
- bs4 (BeautifulSoup): For parsing HTML content.
- concurrent.futures: For multi-threaded execution.
- PIL: For image processing and metadata extraction.
- io: For handling binary data.
"""

import argparse
import time
import random
import logging
import os
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import Timeout, ConnectionError
from PIL import Image
import io
import sys

# Configure logging
logging.basicConfig(
    filename="media_elements_reviewer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# List of User-Agents for random selection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

# Configurable delay between requests
REQUEST_DELAY = 1  # seconds

# Size threshold for flagging large images (in bytes)
LARGE_IMAGE_SIZE = 500 * 1024  # 500KB

# Define image budget thresholds by role
IMAGE_BUDGETS = {
    "Hero": {"size_kb": 1000, "width": 1920, "height": 1080},  # Hero/LCP images
    "Banner": {"size_kb": 200, "width": 1200, "height": 800},  # In-article banners
    "Product": {"size_kb": 100, "width": 600, "height": 600},  # Product thumbnails
    "Icon": {"size_kb": 50, "width": 200, "height": 200},      # Icons/UI graphics
    # Default budget for unknown sections
    "Default": {"size_kb": 200, "width": 1200, "height": 800}
}

# Vendor-specific budgets (stricter for mobile-focused sites)
VENDOR_BUDGETS = {
    "Bluehost": {
        "Hero": {"size_kb": 800, "width": 1600, "height": 900},
        "Banner": {"size_kb": 150, "width": 1000, "height": 600},
        "Product": {"size_kb": 80, "width": 500, "height": 500},
        "Icon": {"size_kb": 40, "width": 180, "height": 180}
    },
    "HostGator": {
        "Hero": {"size_kb": 700, "width": 1500, "height": 800},
        "Banner": {"size_kb": 130, "width": 900, "height": 500},
        "Product": {"size_kb": 70, "width": 450, "height": 450},
        "Icon": {"size_kb": 30, "width": 160, "height": 160}
    },
    # Default vendor budget (used for vendors not explicitly defined)
    "Default": IMAGE_BUDGETS
}

# Ideal compression ratios by image format (KiB/megapixel)
IDEAL_COMPRESSION_RATIOS = {
    "JPEG": 150,    # ~150 KiB per megapixel for good quality JPEG
    "PNG": 600,     # ~600 KiB per megapixel for PNG with alpha
    "WEBP": 100,    # ~100 KiB per megapixel for good quality WEBP
    "AVIF": 60,     # ~60 KiB per megapixel for good quality AVIF
    "SVG": 50,      # SVGs are vector, so this is a rough estimate
    "GIF": 800,     # GIFs are inefficient
    "Default": 200  # Default fallback
}

# Map section names to budget categories
SECTION_TO_BUDGET = {
    "Hero": "Hero",
    "Banner": "Banner",
    "Header": "Hero",
    "Carousel": "Hero",
    "Slider": "Hero",
    "Gallery": "Banner",
    "Product": "Product",
    "Thumbnail": "Product",
    "Icon": "Icon",
    "Nav": "Icon",
    "Footer": "Icon",
    "Logo": "Icon"
}

def get_budget_category(section_name):
    """
    Maps a section name to its corresponding budget category.

    Args:
        section_name (str): The section name from the page.

    Returns:
        str: The budget category name.
    """
    if not section_name:
        return "Default"

    for key, category in SECTION_TO_BUDGET.items():
        if key.lower() in section_name.lower():
            return category

    return "Default"

def get_vendor_from_url(url):
    """
    Extract vendor name from URL.

    Args:
        url (str): The URL of the page.

    Returns:
        str: The vendor name.
    """
    try:
        hostname = urlparse(url).netloc.lower()

        # Extract vendor name from hostname
        if "bluehost" in hostname:
            return "Bluehost"
        elif "hostgator" in hostname:
            return "HostGator"
        elif "domain.com" in hostname:
            return "Domain"
        elif "networksolutions" in hostname:
            return "NetworkSolutions"
        elif "register.com" in hostname:
            return "Register"
        elif "web.com" in hostname:
            return "Web"

        # Default case
        return "Default"
    except:
        return "Default"

def calculate_compression_delta(size_kb, width, height, format_name):
    """
    Calculate the compression delta (how far from ideal compression the image is).

    Args:
        size_kb (float): Size of the image in KB.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        format_name (str): Format of the image (JPEG, PNG, etc.).

    Returns:
        float: The compression delta (percentage above ideal).
        float: The ideal size for this image.
    """
    if width <= 0 or height <= 0 or size_kb <= 0:
        return 0, 0

    # Calculate megapixels
    megapixels = (width * height) / 1000000

    # Get the ideal compression ratio for this format
    format_name = format_name.upper() if format_name else "Default"
    ideal_ratio = IDEAL_COMPRESSION_RATIOS.get(format_name, IDEAL_COMPRESSION_RATIOS["Default"])

    # Calculate ideal size in KB
    ideal_size = megapixels * ideal_ratio

    # Calculate delta percentage (how much larger than ideal)
    if ideal_size > 0:
        delta_percentage = ((size_kb - ideal_size) / ideal_size) * 100
    else:
        delta_percentage = 0

    return round(delta_percentage, 1), round(ideal_size, 1)

def check_image_issues(metadata):
    """
    Check if an image exceeds its budget thresholds based on its section/role.

    Args:
        metadata (dict): The image metadata.

    Returns:
        dict: Updated metadata with issues flag.
    """
    section = metadata.get("Section", "Unknown")
    budget_category = get_budget_category(section)

    # Get vendor-specific budget if available
    vendor = get_vendor_from_url(metadata.get("Page URL", ""))
    vendor_budgets = VENDOR_BUDGETS.get(vendor, VENDOR_BUDGETS["Default"])
    budget = vendor_budgets.get(budget_category, IMAGE_BUDGETS.get(budget_category, IMAGE_BUDGETS["Default"]))

    size_kb = metadata.get("Size (KB)", 0)
    width = metadata.get("Width", 0)
    height = metadata.get("Height", 0)
    format_name = metadata.get("Format", "Unknown")
    loading_type = metadata.get("Loading Type", "Unknown")
    rendered_width = metadata.get("Rendered Width", 0)
    image_type = metadata.get("Image Type", "Unknown")
    has_srcset = metadata.get("Has Srcset", False)
    in_picture = metadata.get("In Picture Element", False)
    has_responsive_options = metadata.get("Has Responsive Options", False)
    responsive_candidates = metadata.get("Responsive Candidates", [])

    issues = []
    severity_score = 0
    issue_categories = []  # For CSS classes

    # Check for eager-loaded hero images > 500 KB (highest priority issue for CWV)
    if loading_type == "Eager" and budget_category == "Hero" and size_kb > 500:
        issues.append("⚠️ HIGH CWV IMPACT: Eager-loaded hero image exceeds LCP threshold (500 KB)")
        severity_score += 300  # Highest severity - will push to top when sorting
        issue_categories.append("high-priority-issue")
        metadata["High Priority"] = True

        # Calculate estimated impact on LCP/CLS
        estimated_lcp_delay_ms = max(0, int((size_kb - 500) * 0.8))
        if estimated_lcp_delay_ms > 0:
            issues.append(f"Est. LCP delay: +{estimated_lcp_delay_ms}ms on slow 4G")

    # Check file size threshold
    if size_kb > budget["size_kb"]:
        size_issue = f"Exceeds size budget: {size_kb:.1f} KB > {budget['size_kb']} KB"
        issues.append(size_issue)
        # Higher severity for larger overages
        severity_score += min(100, int((size_kb / budget["size_kb"]) * 10))

    # Check dimensions threshold
    if width > budget["width"] or height > budget["height"]:
        dim_issue = f"Exceeds dimension budget: {width}x{height} > {budget['width']}x{budget['height']}"
        issues.append(dim_issue)
        # Add to severity based on how much it exceeds dimensions
        if width > 0 and height > 0 and budget["width"] > 0 and budget["height"] > 0:
            area_ratio = (width * height) / (budget["width"] * budget["height"])
            severity_score += min(100, int(area_ratio * 10))

    # Calculate compression efficiency
    if width > 0 and height > 0 and size_kb > 0:
        compression_delta, ideal_size = calculate_compression_delta(size_kb, width, height, format_name)
        metadata["Compression Delta"] = f"{compression_delta:.1f}%"
        metadata["Ideal Size (KB)"] = ideal_size

        # Flag significant compression inefficiency
        if compression_delta > 100:
            issues.append(f"SEVERE compression inefficiency: {compression_delta:.1f}% larger than ideal ({ideal_size:.1f} KB)")
            severity_score += min(150, int(compression_delta / 2))
            issue_categories.append("compression-issue")
        elif compression_delta > 50:
            issues.append(f"Poor compression efficiency: {compression_delta:.1f}% larger than ideal ({ideal_size:.1f} KB)")
            severity_score += min(75, int(compression_delta / 2))
            issue_categories.append("compression-issue")

        # Recommend next-gen formats if using older formats
        if format_name.upper() == "JPEG" and size_kb > 100:
            estimated_webp_size = ideal_size * (IDEAL_COMPRESSION_RATIOS["WEBP"] / IDEAL_COMPRESSION_RATIOS["JPEG"])
            estimated_savings = size_kb - estimated_webp_size
            if estimated_savings > 30:  # Only suggest if savings are significant
                issues.append(f"Convert to WebP for ~{int(estimated_savings)} KB savings ({int((estimated_savings/size_kb)*100)}% smaller)")

        elif format_name.upper() == "PNG" and size_kb > 100:
            estimated_webp_size = ideal_size * (IDEAL_COMPRESSION_RATIOS["WEBP"] / IDEAL_COMPRESSION_RATIOS["PNG"])
            estimated_savings = size_kb - estimated_webp_size
            if estimated_savings > 100:  # PNGs can have dramatic savings
                issues.append(f"Convert to WebP for ~{int(estimated_savings)} KB savings ({int((estimated_savings/size_kb)*100)}% smaller)")
    else:
        metadata["Compression Delta"] = "N/A"
        metadata["Ideal Size (KB)"] = "N/A"

    # Check for intrinsic vs. rendered dimension mismatch
    if rendered_width > 0 and width > 0:
        ratio = width / rendered_width
        metadata["Intrinsic/Rendered Ratio"] = f"{ratio:.1f}x"

        if width > rendered_width * 3:
            issues.append(f"SEVERE oversized image: {width}px intrinsic vs {rendered_width}px rendered (>{ratio:.1f}x)")
            severity_score += min(150, int((ratio) * 20))
            issue_categories.append("dimension-mismatch")
        elif width > rendered_width * 2:
            issues.append(f"Oversized image: {width}px intrinsic vs {rendered_width}px rendered (>{ratio:.1f}x)")
            severity_score += min(100, int((ratio) * 15))
            issue_categories.append("dimension-mismatch")

        # Calculate wasted bytes due to dimension mismatch
        if ratio > 1.5:  # Only if significantly oversized
            wasted_pixels_percent = 1 - (1 / (ratio * ratio))  # % of wasted pixels (area)
            wasted_bytes = size_kb * wasted_pixels_percent
            if wasted_bytes > 50:  # Only mention if significant waste
                issues.append(f"Wasted bytes: ~{int(wasted_bytes)} KB due to dimension mismatch")

    # Check for slow download time (> 200ms on 3G)
    if size_kb > 200 and loading_type == "Eager":
        issues.append(f"Potential slow download on 3G: {size_kb:.1f} KB > 200 KB")
        # Add severity for 3G impact
        severity_score += min(50, int((size_kb - 200) / 10))

    # Check for proper responsive image implementation
    if image_type == "img tag" and width > 400:
        # Check if there's a srcset or it's in a picture element
        if not has_srcset and not in_picture:
            issues.append("Missing responsive image attributes (srcset/sizes)")
            severity_score += 40
            issue_categories.append("responsive-issue")

        # Even if there's a srcset, check if any candidate is close to rendered width
        elif has_responsive_options and rendered_width > 0 and responsive_candidates:
            has_appropriate_size = False
            smallest_candidate = None

            # Parse responsive candidates to check if any is close to rendered size
            for candidate in responsive_candidates:
                # Try to extract width from descriptor (e.g., "100w")
                descriptor = candidate.get("descriptor", "")
                if "w" in descriptor:
                    try:
                        candidate_width = int(descriptor.replace("w", ""))
                        if smallest_candidate is None or candidate_width < smallest_candidate:
                            smallest_candidate = candidate_width

                        # Check if this candidate is appropriate for rendered width
                        if candidate_width <= rendered_width * 1.5:
                            has_appropriate_size = True
                            break
                    except ValueError:
                        pass

            if not has_appropriate_size and smallest_candidate and smallest_candidate > rendered_width * 2:
                issues.append(f"No appropriate srcset candidate for {rendered_width}px display width (smallest: {smallest_candidate}px)")
                severity_score += 35
                issue_categories.append("responsive-issue")

    # Update metadata
    metadata["Budget Category"] = budget_category
    metadata["Vendor"] = vendor
    metadata["Issues"] = ", ".join(issues) if issues else "None"
    metadata["Severity Score"] = severity_score
    metadata["Issue Categories"] = issue_categories

    return metadata

def get_image_metadata(img_url, page_url, section_name="Unknown", retries=2, img_type="Unknown", depth=0):
    """
    Fetches metadata for a given image URL.

    Args:
        img_url (str): The URL of the image.
        page_url (str): The URL of the page where the image was found.
        section_name (str): The section of the page where the image appears.
        retries (int): Number of retries for failed requests.
        img_type (str): The type of image (img tag, background-image, css-variable-background, etc.)
        depth (int): The crawl depth at which this image was found.

    Returns:
        dict: A dictionary containing image metadata.
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    metadata = {
        "Image URL": img_url,
        "Page URL": page_url,
        "Section": section_name,
        "Loading Type": "Unknown",
        "Size (KB)": 0,
        "Width": 0,
        "Height": 0,
        "Format": "Unknown",
        "File Name": os.path.basename(urlparse(img_url).path),
        "Last Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Content Type": "Unknown",
        "Aspect Ratio": "Unknown",
        "Is SVG": False,
        "Image Type": img_type,  # Store the image type in metadata
        "Depth": depth,  # Store the depth at which the image was found
    }

    # Skip obviously non-image files
    skip_extensions = ['.eot', '.woff', '.woff2', '.ttf', '.otf', '.css', '.js', '.txt', '.xml', '.json']
    url_lower = img_url.lower()
    if any(url_lower.endswith(ext) for ext in skip_extensions):
        logging.info(f"Skipping non-image file: {img_url}")
        metadata["Format"] = "Non-image file"
        return metadata

    # Check if it's an SVG based on the file extension
    if img_url.lower().endswith('.svg'):
        metadata["Is SVG"] = True
        metadata["Format"] = "SVG"
    elif img_url.lower().endswith('.png'):
        metadata["Format"] = "PNG"
    elif img_url.lower().endswith('.jpg') or img_url.lower().endswith('.jpeg'):
        metadata["Format"] = "JPEG"
    elif img_url.lower().endswith('.gif'):
        metadata["Format"] = "GIF"
    elif img_url.lower().endswith('.webp'):
        metadata["Format"] = "WEBP"

    # If we have a CSS variable background type, set the format accordingly
    if img_type == "css-variable-background":
        # Try to determine format from URL extension for CSS backgrounds
        file_ext = os.path.splitext(urlparse(img_url).path)[1].lower()
        if file_ext == '.svg':
            metadata["Format"] = "SVG"
        elif file_ext == '.png':
            metadata["Format"] = "PNG"
        elif file_ext in ['.jpg', '.jpeg']:
            metadata["Format"] = "JPEG"
        elif file_ext == '.gif':
            metadata["Format"] = "GIF"
        elif file_ext == '.webp':
            metadata["Format"] = "WEBP"

        # If no extension found, check if we can determine from filename patterns
        if metadata["Format"] == "Unknown" and "--c.png" in img_url:
            metadata["Format"] = "PNG"
        elif metadata["Format"] == "Unknown" and "begin_d.png" in img_url:
            metadata["Format"] = "PNG"

        # Print debug info for CSS variable backgrounds
        print(f"CSS Variable Background Image: {img_url} | Format: {metadata['Format']}")

    # Fix URLs with spaces (likely CSS variable URLs)
    if ' ' in img_url:
        # Fix the URL by replacing spaces with slashes
        fixed_url = img_url.replace(' ', '/')
        metadata["Image URL"] = fixed_url
        img_url = fixed_url  # Update the URL for the request below
        metadata["Fixed URL"] = True
        print(f"Fixed URL: {fixed_url} | Type: {img_type} | Format: {metadata['Format']}")

    for attempt in range(retries):
        try:
            # Use shorter timeout and head request first for efficiency
            head_response = requests.head(img_url, headers=headers, timeout=5)
            if head_response.status_code != 200:
                # Try GET if HEAD fails
                response = requests.get(img_url, headers=headers, timeout=8, stream=True)
            else:
                response = head_response
                
            if response.status_code == 200:
                # Get content type from headers first
                content_type = response.headers.get('content-type', 'Unknown').lower()
                metadata["Content Type"] = content_type

                # Skip if content type indicates it's not an image
                if not any(img_type in content_type for img_type in ['image/', 'svg']):
                    if 'text/' in content_type or 'application/' in content_type:
                        logging.info(f"Skipping non-image content type: {content_type} for {img_url}")
                        metadata["Format"] = f"Non-image ({content_type})"
                        return metadata

                # Get image size in KB
                content_length = response.headers.get('content-length')
                if content_length:
                    metadata["Size (KB)"] = round(int(content_length) / 1024, 2)
                else:
                    # If content-length header is not available and it's not a HEAD request, get the size from the content
                    if hasattr(response, 'content'):
                        content = response.content
                        metadata["Size (KB)"] = round(len(content) / 1024, 2)

                # Check for SVG content type
                if 'svg' in content_type:
                    metadata["Is SVG"] = True
                    metadata["Format"] = "SVG"

                # Get image dimensions, format and calculate aspect ratio (only for actual image files)
                if any(img_type in content_type for img_type in ['image/png', 'image/jpeg', 'image/gif', 'image/webp']):
                    try:
                        # Only fetch content if we haven't already
                        if not hasattr(response, 'content') or response.request.method == 'HEAD':
                            response = requests.get(img_url, headers=headers, timeout=8)
                            
                        img = Image.open(io.BytesIO(response.content))
                        width, height = img.size
                        metadata["Width"] = width
                        metadata["Height"] = height
                        metadata["Format"] = img.format

                        # Calculate aspect ratio with proper formatting
                        if width > 0 and height > 0:
                            ratio = width / height
                            if abs(ratio - round(ratio)) < 0.01:  # Close to an integer
                                metadata["Aspect Ratio"] = f"{round(ratio)}:1"
                            elif abs(ratio - 4/3) < 0.01:
                                metadata["Aspect Ratio"] = "4:3"
                            elif abs(ratio - 16/9) < 0.01:
                                metadata["Aspect Ratio"] = "16:9"
                            elif abs(ratio - 3/2) < 0.01:
                                metadata["Aspect Ratio"] = "3:2"
                            else:
                                metadata["Aspect Ratio"] = f"{round(ratio, 2)}:1"

                    except Exception as e:
                        logging.warning(f"Could not process image dimensions for {img_url}: {e}")

                        # Special handling for SVG files
                        if metadata["Is SVG"] or 'svg' in content_type:
                            try:
                                # Only fetch content if we haven't already
                                if not hasattr(response, 'content') or response.request.method == 'HEAD':
                                    response = requests.get(img_url, headers=headers, timeout=8)
                                    
                                # Try to extract dimensions from SVG content
                                svg_content = response.content.decode('utf-8')
                                width_match = re.search(r'width="(\d+)', svg_content)
                                height_match = re.search(r'height="(\d+)', svg_content)
                                view_box_match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_content)

                                if width_match and height_match:
                                    metadata["Width"] = int(width_match.group(1))
                                    metadata["Height"] = int(height_match.group(1))
                                elif view_box_match:
                                    metadata["Width"] = int(view_box_match.group(1))
                                    metadata["Height"] = int(view_box_match.group(2))
                            except Exception as svg_err:
                                logging.warning(f"Could not extract SVG dimensions for {img_url}: {svg_err}")

                metadata = check_image_issues(metadata)  # Check for budget issues

                return metadata
            else:
                logging.warning(f"Failed to fetch image {img_url}: Status code {response.status_code}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout fetching image {img_url} on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            logging.warning(f"Connection error fetching image {img_url} on attempt {attempt + 1}")
        except Exception as e:
            logging.warning(f"Error fetching image {img_url} on attempt {attempt + 1}: {e}")
        
        if attempt < retries - 1:  # Don't sleep on the last attempt
            time.sleep(REQUEST_DELAY)

    return metadata

def identify_page_section(element):
    """
    Attempts to identify the section of the page where an element appears.

    Args:
        element (BeautifulSoup object): The HTML element to analyze.

    Returns:
        str: Name of the identified section.
    """
    # Check parent elements for section identifiers
    parent = element
    for _ in range(5):  # Check up to 5 levels up
        if not parent:
            break

        # Check for common section identifiers in classes and IDs
        attrs = parent.attrs if hasattr(parent, 'attrs') else {}
        classes = attrs.get('class', [])
        element_id = attrs.get('id', '')

        # Convert class list to string for easier checking
        class_str = ' '.join(classes).lower() if isinstance(classes, list) else str(classes).lower()
        id_str = element_id.lower()

        # Check for common section names
        for section in ['hero', 'banner', 'header', 'footer', 'nav', 'sidebar', 'carousel',
                       'slider', 'gallery', 'content', 'main', 'article', 'feature', 'promo']:
            if section in class_str or section in id_str:
                return section.capitalize()

        parent = parent.parent

    # If we're in a container div with specific role
    if parent and parent.name == 'div' and parent.get('role'):
        return parent.get('role').capitalize()

    return "Unknown"

def determine_loading_type(img_tag):
    """
    Determines if an image is lazy loaded or eager loaded.

    Args:
        img_tag (BeautifulSoup object): The img tag to analyze.

    Returns:
        str: "Lazy" or "Eager"
    """
    if img_tag.get('loading') == 'lazy':
        return "Lazy"
    elif img_tag.get('loading') == 'eager':
        return "Eager"

    # Check for common lazy loading library attributes
    lazy_attrs = ['data-src', 'data-lazy', 'data-original', 'lazy-src']
    for attr in lazy_attrs:
        if img_tag.has_attr(attr):
            return "Lazy"

    # Check for common lazy loading classes
    lazy_classes = ['lazy', 'lazyload', 'b-lazy']
    img_classes = img_tag.get('class', [])
    for lazy_class in lazy_classes:
        if lazy_class in img_classes:
            return "Lazy"

    return "Eager"  # Default to eager if no lazy loading indications found

def get_all_links_and_images(url, retries=3):
    """
    Fetches all links and images from a given URL with retry mechanism.
    Detects both <img> tags and background images in CSS, including those with CSS escape sequences.
    Also analyzes responsive image features and rendered dimensions vs. intrinsic dimensions.

    Args:
        url (str): The URL to fetch content from.
        retries (int): Number of retries for failed requests.

    Returns:
        tuple: A list of links, list of image data dictionaries, HTTP status code, and response time.
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        # Adding width info to mimic browser window and help with responsive image selection
        "Viewport-Width": "1280",
        "Width": "1280"
    }
    links = []
    image_data = []

    for attempt in range(retries):
        try:
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=10)
            status_code = response.status_code
            response_time = round(time.time() - start_time, 2)

            if status_code != 200:
                return links, image_data, status_code, response_time

            soup = BeautifulSoup(response.content, "html.parser")
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]

            # Process all img tags
            for img in soup.find_all('img'):
                src = img.get('src', img.get('data-src', ''))
                if not src:
                    continue

                # Complete the URL if it's relative
                img_url = urljoin(url, src)

                # Skip base64 encoded images
                if img_url.startswith('data:'):
                    continue

                section = identify_page_section(img)
                loading_type = determine_loading_type(img)

                # Check for responsive image attributes
                has_srcset = bool(img.get('srcset'))
                has_sizes = bool(img.get('sizes'))
                in_picture = img.parent.name == 'picture' if img.parent else False

                # Get width and height attributes for analysis
                width_attr = img.get('width', '')
                height_attr = img.get('height', '')
                display_style = img.get('style', '')

                # Extract width from style attribute if present
                style_width = None
                if 'width' in display_style:
                    width_match = re.search(r'width:\s*(\d+)px', display_style)
                    if width_match:
                        style_width = int(width_match.group(1))

                # Estimate rendered width based on available information
                # In a real browser this would be actual rendered dimensions
                rendered_width = None
                if width_attr and width_attr.isdigit():
                    rendered_width = int(width_attr)
                elif style_width:
                    rendered_width = style_width
                elif in_picture:
                    # For picture elements, check source media queries for viewport hints
                    for source in img.parent.find_all('source'):
                        media = source.get('media', '')
                        if 'max-width' in media:
                            max_width_match = re.search(r'max-width:\s*(\d+)px', media)
                            if max_width_match:
                                rendered_width = int(max_width_match.group(1))
                                break

                # Check for responsive options - either srcset or a parent picture element with sources
                responsive_candidates = []
                has_responsive_options = False

                if has_srcset:
                    srcset = img.get('srcset', '')
                    # Parse srcset format: "url1 100w, url2 200w, ..." or "url1 1x, url2 2x, ..."
                    srcset_entries = [s.strip() for s in srcset.split(',')]
                    for entry in srcset_entries:
                        parts = entry.strip().split()
                        if len(parts) >= 2:
                            candidate_url = parts[0]
                            descriptor = parts[1]
                            responsive_candidates.append({
                                'url': urljoin(url, candidate_url),
                                'descriptor': descriptor
                            })
                    has_responsive_options = len(responsive_candidates) > 0

                elif in_picture:
                    # Check if parent picture element has multiple source elements
                    sources = img.parent.find_all('source')
                    for source in sources:
                        srcset = source.get('srcset', '')
                        if srcset:
                            srcset_entries = [s.strip() for s in srcset.split(',')]
                            for entry in srcset_entries:
                                parts = entry.strip().split()
                                if len(parts) >= 1:
                                    candidate_url = parts[0]
                                    # Get media attribute if available
                                    media = source.get('media', '')
                                    responsive_candidates.append({
                                        'url': urljoin(url, candidate_url),
                                        'media': media
                                    })
                    has_responsive_options = len(responsive_candidates) > 0

                # Add basic image data to be enriched later
                image_data.append({
                    "Image URL": img_url,
                    "Page URL": url,
                    "Section": section,
                    "Loading Type": loading_type,
                    "Alt Text": img.get('alt', ''),
                    "Image Type": "img tag",
                    "Has Srcset": has_srcset,
                    "Has Sizes": has_sizes,
                    "In Picture Element": in_picture,
                    "Has Responsive Options": has_responsive_options,
                    "Responsive Candidates": responsive_candidates,
                    "Rendered Width": rendered_width,
                    "Width Attribute": width_attr,
                    "Height Attribute": height_attr,
                })

            # Process picture tags with source elements
            for picture in soup.find_all('picture'):
                # First check for <img> inside picture as fallback
                img = picture.find('img')
                if not img or not img.get('src'):
                    continue  # Skip if no valid img fallback

                img_url = urljoin(url, img.get('src'))
                section = identify_page_section(picture)

                # Collect all source elements within the picture
                responsive_options = []

                for source in picture.find_all('source'):
                    srcset = source.get('srcset', '')
                    media = source.get('media', '')
                    type_attr = source.get('type', '')

                    if not srcset:
                        continue

                    # Handle srcset format (could contain multiple URLs with width descriptors)
                    srcset_urls = []
                    srcset_entries = [s.strip() for s in srcset.split(',')]

                    for entry in srcset_entries:
                        parts = entry.strip().split()
                        if len(parts) >= 1:
                            src = parts[0]
                            descriptor = parts[1] if len(parts) > 1 else ''
                            srcset_urls.append({
                                'url': src,
                                'descriptor': descriptor
                            })

                    for src_item in srcset_urls:
                        src = src_item['url']
                        if not src:
                            continue

                        img_url = urljoin(url, src)
                        if img_url.startswith('data:'):
                            continue

                        # Store the source element information
                        responsive_options.append({
                            'url': img_url,
                            'media': media,
                            'type': type_attr,
                            'descriptor': src_item.get('descriptor', '')
                        })

                        image_data.append({
                            "Image URL": img_url,
                            "Page URL": url,
                            "Section": section,
                            "Loading Type": determine_loading_type(source),
                            "Alt Text": img.get('alt', '') if img else '',
                            "Image Type": "picture source",
                            "In Picture Element": True,
                            "Has Responsive Options": True,
                            "Media Query": media,
                            "Source Type": type_attr,
                            "Parent Picture": True
                        })

            # Find background images in inline styles - with support for CSS escape sequences
            elements_with_style = soup.find_all(lambda tag: tag.has_attr('style'))
            for element in elements_with_style:
                style = element['style']

                # Try to estimate the element's rendered width
                rendered_width = None
                width_match = re.search(r'width:\s*(\d+)px', style)
                if width_match:
                    rendered_width = int(width_match.group(1))
                else:
                    # Check if element has width attribute
                    width_attr = element.get('width', '')
                    if width_attr and width_attr.isdigit():
                        rendered_width = int(width_attr)

                # Check for --background-image CSS variable pattern specifically
                css_var_match = re.search(r'--background-image:\s*url\((.*?)\)', style, re.IGNORECASE | re.DOTALL)
                if css_var_match:
                    bg_url = css_var_match.group(1).strip('\'"')

                    # Handle CSS escape sequences like \2f for /
                    if '\\' in bg_url:
                        # Process escaped characters
                        processed_url = ''
                        i = 0
                        while i < len(bg_url):
                            if bg_url[i] == '\\' and i + 2 < len(bg_url):
                                # Handle hex escapes like \2f
                                if bg_url[i+1:i+3].isalnum():
                                    try:
                                        hex_val = bg_url[i+1:i+3]
                                        char = chr(int(hex_val, 16))
                                        processed_url += char
                                        i += 3  # Skip the escape sequence
                                        continue
                                    except ValueError:
                                        pass  # If not a valid hex, just continue normally
                            processed_url += bg_url[i]
                            i += 1

                        bg_url = processed_url

                    img_url = urljoin(url, bg_url)
                    section = identify_page_section(element)

                    image_data.append({
                        "Image URL": img_url,
                        "Page URL": url,
                        "Section": section,
                        "Loading Type": "Eager",  # CSS variables are usually eager-loaded
                        "Alt Text": "",
                        "Image Type": "css-variable-background",
                        "Has Responsive Options": False,  # CSS backgrounds don't have responsive options like srcset
                        "Rendered Width": rendered_width
                    })

                # Enhanced regex pattern to catch various CSS background image formats including escaped sequences
                url_patterns = [
                    # Standard background-image: url(...)
                    r'background-image:\s*url\([\'"]?(.*?)[\'"]?\)',
                    # Background with url() shorthand
                    r'background:.*?url\([\'"]?(.*?)[\'"]?\)',
                    # CSS escaped sequences like \2f (which is /)
                    r'background(?:-image)?:.*?url\(\\?[\'"]?((?:\\[0-9a-f]{2,}.*?)|.*?)[\'"]?\)',
                ]

                # Apply all patterns to find URLs
                bg_urls = []
                for pattern in url_patterns:
                    found_urls = re.findall(pattern, style, re.IGNORECASE)
                    bg_urls.extend(found_urls)

                for bg_url in bg_urls:
                    if not bg_url or bg_url.startswith('data:'):
                        continue

                    # Handle CSS escape sequences like \2f for /
                    if '\\' in bg_url:
                        # Unescape the CSS sequence
                        try:
                            # Replace sequences like \2f with their character equivalents
                            def replace_escape(match):
                                hex_val = match.group(1)
                                try:
                                    return chr(int(hex_val, 16))
                                except:
                                    return match.group(0)

                            bg_url = re.sub(r'\\([0-9a-f]{2,})', replace_escape, bg_url)
                        except Exception as e:
                            logging.warning(f"Failed to unescape CSS URL: {bg_url}, error: {e}")

                    # Remove any remaining backslashes
                    bg_url = bg_url.replace('\\', '')

                    img_url = urljoin(url, bg_url)
                    section = identify_page_section(element)

                    image_data.append({
                        "Image URL": img_url,
                        "Page URL": url,
                        "Section": section,
                        "Loading Type": "Eager",  # Background images are usually eager-loaded
                        "Alt Text": "",  # Background images don't have alt text
                        "Image Type": "background-image",
                        "Has Responsive Options": False,
                        "Rendered Width": rendered_width
                    })

            # Special handling for divs with CSS variables
            for div in soup.find_all('div'):
                if div.has_attr('style'):
                    style = div['style']

                    # Try to estimate the element's rendered width
                    rendered_width = None
                    width_match = re.search(r'width:\s*(\d+)px', style)
                    if width_match:
                        rendered_width = int(width_match.group(1))
                    else:
                        # Check if element has width attribute
                        width_attr = div.get('width', '')
                        if width_attr and width_attr.isdigit():
                            rendered_width = int(width_attr)

                    # Direct extraction of --background-image CSS variable
                    bg_img_vars = re.findall(r'--background-image:\s*url\((.*?)\)', style, re.IGNORECASE | re.DOTALL)

                    for bg_url in bg_img_vars:
                        bg_url = bg_url.strip('\'"')

                        # Handle CSS escape sequences specifically for the \2f format
                        if '\\2f' in bg_url or '\\' in bg_url:
                            # Process each escaped sequence
                            processed_url = ''
                            i = 0
                            while i < len(bg_url):
                                if bg_url[i:i+3] == '\\2f' or bg_url[i:i+3] == '\\2F':
                                    processed_url += '/'
                                    i += 3
                                elif bg_url[i] == '\\' and i + 2 < len(bg_url) and bg_url[i+1:i+3].isalnum():
                                    # Handle other hex escapes
                                    try:
                                        hex_val = bg_url[i+1:i+3]
                                        char = chr(int(hex_val, 16))
                                        processed_url += char
                                        i += 3
                                    except ValueError:
                                        processed_url += bg_url[i]
                                        i += 1
                                else:
                                    processed_url += bg_url[i]
                                    i += 1

                            bg_url = processed_url

                        # Remove any quotes that might still be present
                        bg_url = bg_url.strip('\'"')

                        if not bg_url or bg_url.startswith('data:'):
                            continue

                        img_url = urljoin(url, bg_url)
                        section = identify_page_section(div)

                        # Print debug info for this specific case
                        print(f"Found CSS var background image: {img_url} in section {section}")

                        image_data.append({
                            "Image URL": img_url,
                            "Page URL": url,
                            "Section": section,
                            "Loading Type": "Eager",
                            "Alt Text": "",
                            "Image Type": "css-variable-background",
                            "Has Responsive Options": False,
                            "Rendered Width": rendered_width
                        })

            # Find SVG elements embedded directly in HTML
            for svg in soup.find_all('svg'):
                section = identify_page_section(svg)
                # Create a unique identifier for this SVG
                svg_id = svg.get('id', '')
                svg_class = ' '.join(svg.get('class', []))
                identifier = f"{svg_id}_{svg_class}".strip('_')

                # Try to get width and height
                width = svg.get('width', '')
                height = svg.get('height', '')
                viewbox = svg.get('viewBox', '')

                # Try to get rendered dimensions
                rendered_width = None
                if width and width.isdigit():
                    rendered_width = int(width)
                elif viewbox:
                    # Try to extract width from viewBox
                    viewbox_parts = viewbox.split()
                    if len(viewbox_parts) == 4:  # Format: min-x min-y width height
                        try:
                            rendered_width = int(float(viewbox_parts[2]))
                        except (ValueError, IndexError):
                            pass

                image_data.append({
                    "Image URL": f"{url}#svg-{identifier}",  # Use fragment identifier
                    "Page URL": url,
                    "Section": section,
                    "Loading Type": "Inline",
                    "Alt Text": svg.get('aria-label', '') or svg.get('title', ''),
                    "Image Type": "inline-svg",
                    "SVG Width": width,
                    "SVG Height": height,
                    "SVG ViewBox": viewbox,
                    "Has Responsive Options": False,  # SVGs are scalable by nature
                    "Rendered Width": rendered_width
                })

            # Find CSS stylesheet links and check for background images in external CSS
            css_links = []
            for link in soup.find_all('link', rel='stylesheet'):
                if link.get('href'):
                    css_url = urljoin(url, link.get('href'))
                    css_links.append(css_url)

            # Sample the first few CSS files (to avoid overwhelming)
            for css_url in css_links[:3]:  # Limit to 3 CSS files per page
                try:
                    css_response = requests.get(css_url, headers=headers, timeout=5)
                    if css_response.status_code == 200:
                        css_content = css_response.text

                        # Find URLs in the CSS content
                        css_urls = re.findall(r'url\([\'"]?(.*?)[\'"]?\)', css_content)

                        for css_img_url in css_urls:
                            if not css_img_url or css_img_url.startswith('data:'):
                                continue

                            # Handle CSS escape sequences
                            if '\\' in css_img_url:
                                try:
                                    def replace_escape(match):
                                        hex_val = match.group(1)
                                        try:
                                            return chr(int(hex_val, 16))
                                        except:
                                            return match.group(0)

                                    css_img_url = re.sub(r'\\([0-9a-fA-F]{2,})', replace_escape, css_img_url)
                                except Exception as e:
                                    logging.warning(f"Failed to unescape CSS URL: {css_img_url}, error: {e}")

                            full_img_url = urljoin(css_url, css_img_url)

                            image_data.append({
                                "Image URL": full_img_url,
                                "Page URL": url,
                                "Section": "CSS Stylesheet",
                                "Loading Type": "Eager",
                                "Alt Text": "",
                                "Image Type": "css-file-image",
                                "Has Responsive Options": False
                            })
                except Exception as e:
                    logging.warning(f"Failed to process CSS file {css_url}: {e}")

            return links, image_data, status_code, response_time
        except Timeout:
            logging.warning(f"Timeout error for {url} on attempt {attempt + 1}")
        except ConnectionError:
            logging.warning(f"Connection error for {url} on attempt {attempt + 1}")
        except requests.RequestException as e:
            logging.error(f"Request error for {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error for {url}: {e}")

        time.sleep(REQUEST_DELAY)

    logging.error(f"Failed to fetch {url} after {retries} retries")
    return [], [], None, None

def crawl_website_for_images(start_url, brand, max_depth=0):
    """
    Crawls a website starting from a given URL up to a specified depth,
    collecting data about images.

    Args:
        start_url (str): The starting URL for the crawl.
        brand (str): The brand name being crawled.
        max_depth (int): The maximum depth to crawl.

    Returns:
        list: A list of dictionaries containing metadata for each image found.
    """
    visited_urls = set()
    all_image_data = []
    urls_to_visit = [(start_url, 0)]
    processed_images = set()
    brand_name = urlparse(start_url).netloc.split('.')[1].capitalize()

    # Add termination controls
    MAX_URLS_TO_PROCESS = 50  # Limit total URLs processed
    MAX_IMAGES_PER_BRAND = 200  # Limit total images per brand
    MAX_PROCESSING_TIME = 300  # 5 minutes max processing time
    start_time = time.time()

    print(f"\nStarting image crawl for {brand_name}...")
    print(f"Limits: Max URLs={MAX_URLS_TO_PROCESS}, Max Images={MAX_IMAGES_PER_BRAND}, Max Time={MAX_PROCESSING_TIME}s")

    while urls_to_visit and len(visited_urls) < MAX_URLS_TO_PROCESS:
        # Check timeout
        if time.time() - start_time > MAX_PROCESSING_TIME:
            print(f"Timeout reached for {brand_name}. Stopping crawl.")
            logging.warning(f"Timeout reached for {brand_name} after {MAX_PROCESSING_TIME}s")
            break

        # Check image limit
        if len(all_image_data) >= MAX_IMAGES_PER_BRAND:
            print(f"Image limit reached for {brand_name}. Stopping crawl.")
            logging.warning(f"Image limit reached for {brand_name} at {MAX_IMAGES_PER_BRAND} images")
            break

        current_url, depth = urls_to_visit.pop(0)
        if depth > max_depth:
            continue

        if current_url not in visited_urls:
            visited_urls.add(current_url)
            logging.info(f"Crawling for images on {current_url} at depth {depth}")
            print(f"Crawling {len(visited_urls)}/{MAX_URLS_TO_PROCESS}: {current_url} (depth {depth})")

            try:
                links, page_images, status_code, _ = get_all_links_and_images(current_url)

                if status_code == 200 and page_images:
                    # Process only a subset of images to avoid overwhelming the system
                    for img_data in page_images[:5]:  # Reduced from 10 to 5 images per page
                        if len(all_image_data) >= MAX_IMAGES_PER_BRAND:
                            break

                        img_url = img_data["Image URL"]
                        if img_url not in processed_images:
                            processed_images.add(img_url)

                            try:
                                # Get detailed metadata for the image
                                metadata = get_image_metadata(
                                    img_url,
                                    current_url,
                                    img_data["Section"],
                                    retries=1,  # Reduced retries from 2 to 1
                                    img_type=img_data["Image Type"],
                                    depth=depth
                                )

                                # Update with data from initial scan
                                metadata["Loading Type"] = img_data["Loading Type"]
                                metadata["Alt Text"] = img_data.get("Alt Text", "")
                                metadata["Brand"] = brand_name

                                all_image_data.append(metadata)

                                # Print progress for large images
                                if metadata["Size (KB)"] > LARGE_IMAGE_SIZE / 1024:
                                    print(f"Found large image ({metadata['Size (KB)']} KB): {img_url}")
                            except Exception as e:
                                logging.warning(f"Failed to process image {img_url}: {e}")
                                continue

                    # Add new URLs to visit (with limits)
                    new_links_added = 0
                    for link in links[:20]:  # Limit to 20 links per page
                        if new_links_added >= 10:  # Max 10 new URLs per page
                            break
                        parsed_link = urlparse(link)
                        if (parsed_link.netloc == urlparse(start_url).netloc and
                            link not in visited_urls and
                            len(urls_to_visit) < 100):  # Limit queue size
                            urls_to_visit.append((link, depth + 1))
                            new_links_added += 1
                else:
                    logging.warning(f"Failed to process {current_url}, status: {status_code}")

            except Exception as e:
                logging.error(f"Error processing URL {current_url}: {e}")
                continue

        time.sleep(REQUEST_DELAY)  # Use configurable delay

    print(f"Crawl completed for {brand_name}: {len(all_image_data)} images from {len(visited_urls)} URLs")
    return all_image_data

def crawl_brand_for_images(brand_url, max_depth):
    """
    Initiates image crawling for a specific brand URL.

    Args:
        brand_url (str): The brand's base URL.
        max_depth (int): The maximum depth to crawl.

    Returns:
        tuple: The brand name and the crawled image data.
    """
    brand_name = urlparse(brand_url).netloc.split('.')[1].capitalize()
    return brand_name, crawl_website_for_images(brand_url, brand_name, max_depth)

def generate_html_report(all_data, output_path):
    """
    Generates an HTML report with filtering, sorting, and export capabilities.

    Args:
        all_data (dict): Dictionary with brand names as keys and image data as values.
        output_path (str): Path where the HTML report should be saved.
    """
    with open(output_path, "w") as file:
        file.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FOS Report - Media Elements Review across Jarvis Brands</title>
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
                    padding: 0;
                    box-sizing: border-box;
                }
                
                * {
                    box-sizing: border-box;
                }
                
                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                    font-weight: bold;
                    padding: 20px 0;
                    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                    color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,123,255,0.3);
                    margin: 0 0 30px 0;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }
                
                /* Notes section styling for Media Performance Thresholds */
                .notes-section {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 25px;
                    margin: 20px 0 30px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .notes-section h2 {
                    color: #007bff;
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 5px;
                }
                
                .notes-section h3 {
                    color: #495057;
                    font-size: 1.2em;
                    margin: 20px 0 15px 0;
                }
                
                .notes-section p {
                    line-height: 1.6;
                    margin-bottom: 15px;
                    color: #495057;
                }
                
                .notes-section ul {
                    margin: 15px 0;
                    padding-left: 25px;
                }
                
                .notes-section li {
                    margin-bottom: 8px;
                    line-height: 1.5;
                    color: #495057;
                }
                
                .budget-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    background-color: white;
                    border-radius: 6px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .budget-table th {
                    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                    color: white;
                    padding: 12px 15px;
                    text-align: left;
                    font-weight: bold;
                    font-size: 14px;
                }
                
                .budget-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid #dee2e6;
                    color: #495057;
                }
                
                .budget-table tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                
                .budget-table tr:hover {
                    background-color: #e3f2fd;
                }
                
                .filters {
                    margin: 20px 0;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }
                
                .filter-group {
                    display: flex;
                    flex-direction: column;
                    min-width: 150px;
                }
                
                label {
                    font-weight: bold;
                    margin-bottom: 5px;
                    font-size: 14px;
                    color: #495057;
                }
                
                select, input {
                    padding: 8px 12px;
                    font-size: 14px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                    width: 100%;
                }
                
                select:focus, input:focus {
                    outline: none;
                    border-color: #007bff;
                    box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
                }
                
                /* DataTables wrapper improvements */
                .dataTables_wrapper {
                    width: 100%;
                    overflow-x: auto;
                    margin-top: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    padding: 20px;
                }
                
                .dataTables_length,
                .dataTables_filter {
                    margin-bottom: 20px;
                }
                
                .dataTables_length select {
                    width: auto;
                    min-width: 80px;
                    margin: 0 5px;
                }
                
                .dataTables_filter input {
                    width: auto;
                    min-width: 200px;
                    margin-left: 10px;
                }
                
                /* Table improvements */
                table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                    min-width: 1400px; /* Increased for better data visibility */
                    background-color: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                th, td {
                    text-align: left;
                    padding: 12px 10px;
                    border: 1px solid #dee2e6;
                    word-wrap: break-word;
                    vertical-align: top;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                
                th {
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    font-weight: bold;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                    color: #495057;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    border-bottom: 2px solid #007bff;
                }
                
                /* Enhanced column-specific widths and formatting */
                th:nth-child(1), td:nth-child(1) { 
                    width: 130px; 
                    text-align: center;
                } /* Preview */
                
                th:nth-child(2), td:nth-child(2) { 
                    width: 90px;
                    font-weight: 600;
                    color: #007bff;
                } /* Brand */
                
                th:nth-child(3), td:nth-child(3) { 
                    width: 250px;
                    max-width: 250px;
                    white-space: normal;
                } /* Page URL */
                
                th:nth-child(4), td:nth-child(4) { 
                    width: 250px;
                    max-width: 250px;
                    white-space: normal;
                } /* Image URL */
                
                th:nth-child(5), td:nth-child(5) { 
                    width: 100px;
                    text-align: center;
                } /* Section */
                
                th:nth-child(6), td:nth-child(6) { 
                    width: 90px;
                    text-align: center;
                } /* Loading Type */
                
                th:nth-child(7), td:nth-child(7) { 
                    width: 80px;
                    text-align: right;
                    font-family: 'Courier New', monospace;
                    font-weight: bold;
                } /* Size */
                
                th:nth-child(8), td:nth-child(8) { 
                    width: 120px;
                    text-align: center;
                    font-family: 'Courier New', monospace;
                } /* Dimensions */
                
                th:nth-child(9), td:nth-child(9) { 
                    width: 80px;
                    text-align: center;
                } /* Format */
                
                th:nth-child(10), td:nth-child(10) { 
                    width: 140px;
                    font-size: 11px;
                } /* Content Type */
                
                th:nth-child(11), td:nth-child(11) { 
                    width: 80px;
                    text-align: center;
                    font-family: 'Courier New', monospace;
                } /* Aspect Ratio */
                
                th:nth-child(12), td:nth-child(12) { 
                    width: 180px;
                    white-space: normal;
                    font-style: italic;
                    color: #6c757d;
                } /* Alt Text */
                
                th:nth-child(13), td:nth-child(13) { 
                    width: 120px;
                    text-align: center;
                } /* Image Type */
                
                th:nth-child(14), td:nth-child(14) { 
                    width: 60px;
                    text-align: center;
                    font-weight: bold;
                } /* Depth */
                
                th:nth-child(15), td:nth-child(15) { 
                    width: 140px;
                    text-align: center;
                    font-size: 11px;
                } /* Last Updated */
                
                th:nth-child(16), td:nth-child(16) { 
                    width: 100px;
                    text-align: center;
                    font-weight: 600;
                } /* Budget Category */
                
                th:nth-child(17), td:nth-child(17) { 
                    width: 300px;
                    white-space: normal;
                    font-size: 11px;
                    line-height: 1.4;
                } /* Issues */
                
                /* Row striping and hover effects */
                tbody tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                
                tbody tr:hover {
                    background-color: #e3f2fd !important;
                    transform: scale(1.01);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    transition: all 0.2s ease;
                }
                
                /* Enhanced data formatting */
                td[data-size] {
                    position: relative;
                }
                
                /* Size indicators */
                td:nth-child(7) {
                    position: relative;
                }
                
                td:nth-child(7)::after {
                    content: attr(data-size-indicator);
                    position: absolute;
                    right: 5px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 10px;
                    padding: 2px 4px;
                    border-radius: 2px;
                    font-weight: normal;
                }
                
                /* URL truncation with tooltips */
                td a {
                    color: #007bff;
                    text-decoration: none;
                    word-break: break-all;
                    display: block;
                    max-width: 100%;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    position: relative;
                }
                
                td a:hover {
                    text-decoration: underline;
                    color: #0056b3;
                }
                
                td a::after {
                    content: attr(href);
                    position: absolute;
                    bottom: 100%;
                    left: 0;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 11px;
                    white-space: nowrap;
                    opacity: 0;
                    visibility: hidden;
                    transition: all 0.3s ease;
                    z-index: 1000;
                    max-width: 400px;
                    word-break: break-all;
                    white-space: normal;
                }
                
                td a:hover::after {
                    opacity: 1;
                    visibility: visible;
                }
                
                /* Enhanced format badges */
                .format-badge {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 10px;
                    font-weight: bold;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    min-width: 45px;
                    text-align: center;
                }
                
                .svg-badge {
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    box-shadow: 0 2px 4px rgba(0,123,255,0.3);
                }
                
                .css-badge {
                    background: linear-gradient(135deg, #28a745, #1e7e34);
                    color: white;
                    box-shadow: 0 2px 4px rgba(40,167,69,0.3);
                }
                
                .png-badge {
                    background: linear-gradient(135deg, #6f42c1, #563d7c);
                    color: white;
                }
                
                .jpeg-badge {
                    background: linear-gradient(135deg, #fd7e14, #e55a00);
                    color: white;
                }
                
                .webp-badge {
                    background: linear-gradient(135deg, #20c997, #16a2b5);
                    color: white;
                }
                
                /* Enhanced row highlighting based on data */
                .large-image {
                    background: linear-gradient(135deg, #ffebee, #ffcdd2) !important;
                    border-left: 4px solid #f44336;
                }
                
                .zero-dimensions {
                    background: linear-gradient(135deg, #fff8e1, #ffecb3) !important;
                    border-left: 4px solid #ff9800;
                }
                
                .svg-format {
                    background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important;
                    border-left: 4px solid #2196f3;
                }
                
                .high-priority-issue {
                    background: linear-gradient(135deg, #ffebee, #ffcdd2) !important;
                    font-weight: bold;
                    border: 2px solid #f44336;
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
                    70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
                }
                
                .compression-issue {
                    background: linear-gradient(135deg, #fce4ec, #f8bbd9) !important;
                    border-left: 4px solid #e91e63;
                }
                
                .responsive-issue {
                    background: linear-gradient(135deg, #f3e5f5, #e1bee7) !important;
                    border-left: 4px solid #9c27b0;
                }
                
                .dimension-mismatch {
                    background: linear-gradient(135deg, #fff3e0, #ffe0b2) !important;
                    border-left: 4px solid #ff9800;
                }
                
                /* Loading type indicators */
                td:nth-child(6) {
                    font-weight: bold;
                }
                
                td:nth-child(6):contains("Eager") {
                    color: #dc3545;
                }
                
                td:nth-child(6):contains("Lazy") {
                    color: #28a745;
                }
                
                /* Issues column formatting */
                td:nth-child(17) {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                
                td:nth-child(17):contains("HIGH CWV IMPACT") {
                    color: #dc3545;
                    font-weight: bold;
                }
                
                td:nth-child(17):contains("SEVERE") {
                    color: #fd7e14;
                    font-weight: 600;
                }
                
                /* Responsive table enhancements */
                @media (max-width: 1400px) {
                    table {
                        min-width: 1200px;
                        font-size: 12px;
                    }
                    
                    th, td {
                        padding: 8px 6px;
                    }
                }
                
                @media (max-width: 1000px) {
                    table {
                        min-width: 900px;
                        font-size: 11px;
                    }
                    
                    th, td {
                        padding: 6px 4px;
                    }
                    
                    /* Hide less critical columns on smaller screens */
                    th:nth-child(10), td:nth-child(10), /* Content Type */
                    th:nth-child(11), td:nth-child(11), /* Aspect Ratio */
                    th:nth-child(15), td:nth-child(15) { /* Last Updated */
                        display: none;
                    }
                }
                
                @media (max-width: 768px) {
                    table {
                        min-width: 600px;
                        font-size: 10px;
                    }
                    
                    /* Hide more columns on mobile */
                    th:nth-child(12), td:nth-child(12), /* Alt Text */
                    th:nth-child(14), td:nth-child(14), /* Depth */
                    th:nth-child(16), td:nth-child(16) { /* Budget Category */
                        display: none;
                    }
                }
            </style>
        </head>
        <body>
            <h1>FOS Report - Media Elements Review across Jarvis Brands</h1>
            
            <div class="notes-section">
                <h2>Media Performance Thresholds</h2>
                <p>This report analyzes images across brand websites and flags issues based on context-specific budgets.
                Images that exceed size or dimension thresholds can negatively impact website performance:</p>
                <ul>
                    <li><strong>File size (KB/MB)</strong> - Impacts download time. Large images may slow down page loading, especially on mobile networks.</li>
                    <li><strong>Pixel dimensions (px)</strong> - Impacts decode & render time. Oversized images require significant processing power.</li>
                    <li><strong>Load Time</strong> - Images larger than ~200 KB may download slower than 200ms on a 3G mobile connection.</li>
                </ul>
                
                <h3>Image Budget Thresholds by Role</h3>
                <table class="budget-table">
                    <thead>
                        <tr>
                            <th>Image Role</th>
                            <th>File-size Budget</th>
                            <th>Pixel-dimension Budget</th>
                            <th>Examples</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Hero / LCP images</td>
                            <td>≤ 500 KB–1 MB</td>
                            <td>≤ 1,920 × 1,080 px</td>
                            <td>Header banners, featured images</td>
                        </tr>
                        <tr>
                            <td>In-article banners</td>
                            <td>≤ 200 KB</td>
                            <td>≤ 1,200 × 800 px</td>
                            <td>Content section images, carousels</td>
                        </tr>
                        <tr>
                            <td>Product thumbnails</td>
                            <td>≤ 100 KB</td>
                            <td>≤ 600 × 600 px</td>
                            <td>Product images, gallery thumbnails</td>
                        </tr>
                        <tr>
                            <td>Icons / UI graphics</td>
                            <td>≤ 50 KB</td>
                            <td>≤ 200 × 200 px</td>
                            <td>Navigation icons, logos, buttons</td>
                        </tr>
                    </tbody>
                </table>
                
                <p><strong>Note:</strong> The "Issues" column flags any images that exceed the thresholds based on their role/section.
                These budgets may be tuned for specific audiences (e.g., emerging-market mobile users).</p>
            </div>
            
            <div class="filters">
                <div class="filter-group">
                    <label for="brandFilter">Brand:</label>
                    <select id="brandFilter">
                        <option value="">All Brands</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="sectionFilter">Section:</label>
                    <select id="sectionFilter">
                        <option value="">All Sections</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="loadingFilter">Loading Type:</label>
                    <select id="loadingFilter">
                        <option value="">All</option>
                        <option value="Lazy">Lazy</option>
                        <option value="Eager">Eager</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="formatFilter">Format:</label>
                    <select id="formatFilter">
                        <option value="">All Formats</option>
                        <option value="SVG">SVG</option>
                        <option value="PNG">PNG</option>
                        <option value="JPEG">JPEG</option>
                        <option value="GIF">GIF</option>
                        <option value="WEBP">WEBP</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="sizeFilter">Size:</label>
                    <select id="sizeFilter">
                        <option value="">All</option>
                        <option value="large">Large (>500KB)</option>
                        <option value="medium">Medium (100-500KB)</option>
                        <option value="small">Small (<100KB)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="dimensionFilter">Dimensions:</label>
                    <select id="dimensionFilter">
                        <option value="">All</option>
                        <option value="zero">Zero Dimensions (0x0)</option>
                        <option value="nonzero">Non-Zero Dimensions</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="imageTypeFilter">Image Type:</label>
                    <select id="imageTypeFilter">
                        <option value="">All Types</option>
                        <option value="img tag">Standard Images</option>
                        <option value="background-image">Background Images</option>
                        <option value="css-variable-background">CSS Variable Images</option>
                        <option value="css-file-image">CSS File Images</option>
                        <option value="inline-svg">Inline SVG</option>
                        <option value="picture source">Picture Sources</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="depthFilter">Depth Level:</label>
                    <select id="depthFilter">
                        <option value="">All Depths</option>
                        <option value="0">Level 0 (Homepage)</option>
                        <option value="1">Level 1</option>
                        <option value="2">Level 2</option>
                        <option value="3">Level 3</option>
                        <option value="4">Level 4</option>
                        <option value="5">Level 5+</option>
                    </select>
                </div>
            </div>
            
            <table id="imageReport" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Preview</th>
                        <th>Brand</th>
                        <th>Page URL</th>
                        <th>Image URL</th>
                        <th>Section</th>
                        <th>Loading Type</th>
                        <th>Size (KB)</th>
                        <th>Dimensions</th>
                        <th>Format</th>
                        <th>Content Type</th>
                        <th>Aspect Ratio</th>
                        <th>Alt Text</th>
                        <th>Image Type</th>
                        <th>Depth</th>
                        <th>Last Updated</th>
                        <th>Budget Category</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
        """)

        # Flatten the data for the report
        flattened_data = []
        for brand_data in all_data.values():
            flattened_data.extend(brand_data)

        # Print total count for debugging
        print(f"Total images found for report: {len(flattened_data)}")

        # Count by image type
        image_types = {}
        for row in flattened_data:
            img_type = row.get('Image Type', 'Unknown')
            if img_type in image_types:
                image_types[img_type] += 1
            else:
                image_types[img_type] = 1

        # Print image type counts
        print("Image types found:")
        for img_type, count in image_types.items():
            print(f"  - {img_type}: {count}")

        for row in flattened_data:
            size_kb = row.get('Size (KB)', 0)
            is_large = size_kb > (LARGE_IMAGE_SIZE / 1024)
            width = row.get('Width', 0)
            height = row.get('Height', 0)
            is_zero_dim = (width == 0 or height == 0)
            is_svg = row.get('Is SVG', False) or row.get('Format', '').upper() == 'SVG'
            is_css_var = row.get('Image Type', '') == 'css-variable-background'
            depth = row.get('Depth', 0)

            # Determine row class based on conditions
            row_class = ''
            if is_large:
                row_class = 'large-image'
            elif is_css_var:
                row_class = 'css-variable'
            elif is_zero_dim:
                row_class = 'zero-dimensions'
            elif is_svg:
                row_class = 'svg-format'

            dimensions = f"{width} × {height}"
            format_display = row.get('Format', 'Unknown')
            if is_svg:
                format_display = f'<span class="format-badge svg-badge">SVG</span>'
            elif is_css_var:
                format_display = f'<span class="format-badge css-badge">CSS VAR</span>'

            # Data attributes for filtering
            data_attrs = f'data-size="{size_kb}" data-format="{row.get("Format", "Unknown")}" data-dimensions="{("zero" if is_zero_dim else "nonzero")}" data-image-type="{row.get("Image Type", "Unknown")}" data-depth="{depth}"'

            file.write(f"""
                <tr class="{row_class}" {data_attrs}>
                    <td><img src="{row['Image URL']}" alt="Preview" class="image-preview" onerror="this.src='https://via.placeholder.com/100x100?text=No+Preview';"></td>
                    <td>{row.get('Brand', 'Unknown')}</td>
                    <td><a href="{row['Page URL']}" target="_blank">{row['Page URL'][:50]}{'...' if len(row['Page URL']) > 50 else ''}</a></td>
                    <td><a href="{row['Image URL']}" target="_blank">{row['Image URL'][:50]}{'...' if len(row['Image URL']) > 50 else ''}</a></td>
                    <td>{row.get('Section', 'Unknown')}</td>
                    <td>{row.get('Loading Type', 'Unknown')}</td>
                    <td>{size_kb}</td>
                    <td>{dimensions}</td>
                    <td>{format_display}</td>
                    <td>{row.get('Content Type', 'Unknown')}</td>
                    <td>{row.get('Aspect Ratio', 'Unknown')}</td>
                    <td>{row.get('Alt Text', '')[:50]}{'...' if len(row.get('Alt Text', '')) > 50 else ''}</td>
                    <td>{row.get('Image Type', 'Unknown')}</td>
                    <td>{depth}</td>
                    <td>{row.get('Last Updated', '')}</td>
                    <td>{row.get('Budget Category', 'Unknown')}</td>
                    <td>{row.get('Issues', 'None')}</td>
                </tr>
            """)

        file.write("""
                </tbody>
            </table>
            
            <!-- Image Modal -->
            <div id="imageModal">
                <span class="close">&times;</span>
                <img class="modal-content" id="modalImg">
            </div>
            
            <script>
                $(document).ready(function() {
                    // Initialize the DataTable
                    const table = $('#imageReport').DataTable({
                        "paging": true,
                        "searching": true,
                        "ordering": true,
                        "dom": 'Bfrtip',
                        "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                        "pageLength": 25,
                        "buttons": [
                            'pageLength',
                            {
                                extend: 'excelHtml5',
                                text: 'Export to Excel',
                                className: 'exportButton',
                                exportOptions: {
                                    columns: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                                }
                            },
                            {
                                extend: 'csvHtml5',
                                text: 'Export to CSV',
                                className: 'exportButton',
                                exportOptions: {
                                    columns: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                                }
                            },
                            {
                                extend: 'pdfHtml5',
                                text: 'Export to PDF',
                                className: 'exportButton',
                                exportOptions: {
                                    columns: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                                }
                            }
                        ]
                    });
                    
                    // Populate filter dropdowns
                    const brands = new Set();
                    const sections = new Set();
                    const formats = new Set();
                    const imageTypes = new Set();
                    const depths = new Set();
                    
                    table.rows().every(function() {
                        const data = this.data();
                        brands.add(data[1]);  // Brand column
                        sections.add(data[4]); // Section column
                        imageTypes.add(data[12]); // Image Type column
                        
                        // Extract format (without HTML tags)
                        const format = $(data[8]).text().trim();
                        if (format) formats.add(format);
                    });
                    
                    // Add options to dropdowns
                    brands.forEach(brand => {
                        $('#brandFilter').append(`<option value="${brand}">${brand}</option>`);
                    });
                    
                    sections.forEach(section => {
                        $('#sectionFilter').append(`<option value="${section}">${section}</option>`);
                    });
                    
                    formats.forEach(format => {
                        if (format !== 'Unknown')
                            $('#formatFilter').append(`<option value="${format}">${format}</option>`);
                    });
                    
                    imageTypes.forEach(type => {
                        if (type !== 'Unknown')
                            $('#imageTypeFilter').append(`<option value="${type}">${type}</option>`);
                    });
                    
                    // Apply filters when changed
                    $('#brandFilter, #sectionFilter, #loadingFilter, #formatFilter, #sizeFilter, #dimensionFilter, #imageTypeFilter, #depthFilter').on('change', function() {
                        applyFilters();
                    });
                    
                    function applyFilters() {
                        const brandFilter = $('#brandFilter').val();
                        const sectionFilter = $('#sectionFilter').val();
                        const loadingFilter = $('#loadingFilter').val();
                        const formatFilter = $('#formatFilter').val();
                        const sizeFilter = $('#sizeFilter').val();
                        const dimensionFilter = $('#dimensionFilter').val();
                        const imageTypeFilter = $('#imageTypeFilter').val();
                        const depthFilter = $('#depthFilter').val();
                        
                        // Set table to show all entries if any filter is applied
                        if (brandFilter || sectionFilter || loadingFilter || formatFilter || sizeFilter || dimensionFilter || imageTypeFilter || depthFilter) {
                            table.page.len(-1).draw(); // Set to "All"
                            $('.dataTables_length select').val('-1').trigger('change');
                        }
                        
                        table.rows().every(function() {
                            const data = this.data();
                            const row = $(this.node());
                            
                            let showRow = true;
                            
                            // Brand filter
                            if (brandFilter && data[1] !== brandFilter) {
                                showRow = false;
                            }
                            
                            // Section filter
                            if (sectionFilter && data[4] !== sectionFilter) {
                                showRow = false;
                            }
                            
                            // Loading type filter
                            if (loadingFilter && data[5] !== loadingFilter) {
                                showRow = false;
                            }
                            
                            // Format filter
                            if (formatFilter) {
                                const rowFormat = row.attr('data-format');
                                if (formatFilter !== rowFormat) {
                                    showRow = false;
                                }
                            }
                            
                            // Size filter
                            if (sizeFilter) {
                                const size = parseFloat(row.attr('data-size'));
                                if (sizeFilter === 'large' && size <= 500) showRow = false;
                                if (sizeFilter === 'medium' && (size < 100 || size > 500)) showRow = false;
                                if (sizeFilter === 'small' && size >= 100) showRow = false;
                            }
                            
                            // Dimension filter
                            if (dimensionFilter) {
                                const dimensions = row.attr('data-dimensions');
                                if (dimensionFilter !== dimensions) showRow = false;
                            }
                            
                            // Image type filter
                            if (imageTypeFilter) {
                                const imageType = row.attr('data-image-type');
                                if (imageTypeFilter !== imageType) showRow = false;
                            }
                            
                            // Depth filter
                            if (depthFilter) {
                                const depth = parseInt(row.attr('data-depth'));
                                if (depthFilter === '5' && depth < 5) showRow = false;
                                else if (depthFilter !== '5' && depth != parseInt(depthFilter)) showRow = false;
                            }
                            
                            if (showRow) {
                                row.show();
                            } else {
                                row.hide();
                            }
                        });
                        
                        table.draw();
                    }
                    
                    // Image modal functionality
                    const modal = document.getElementById('imageModal');
                    const modalImg = document.getElementById('modalImg');
                    const closeBtn = document.getElementsByClassName('close')[0];
                    
                    // Open modal on image click
                    $(document).on('click', '.image-preview', function() {
                        modal.style.display = "block";
                        modalImg.src = this.src.replace('https://via.placeholder.com/100x100?text=No+Preview', this.getAttribute('src'));
                    });
                    
                    // Close modal
                    closeBtn.onclick = function() {
                        modal.style.display = "none";
                    }
                    
                    // Close modal on outside click
                    window.onclick = function(event) {
                        if (event.target == modal) {
                            modal.style.display = "none";
                        }
                    }
                });
            </script>
        </body>
        </html>
        """)

    print(f"HTML report generated at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Media Elements Reviewer Script")
    parser.add_argument("--max_depth", type=int, default=0, help="Maximum depth for crawling (default: 2)")
    parser.add_argument("--large_size", type=int, default=500, help="Size threshold in KB to flag large images (default: 500KB)")
    args = parser.parse_args()

    global LARGE_IMAGE_SIZE
    LARGE_IMAGE_SIZE = args.large_size * 1024  # Convert to bytes
    max_depth = args.max_depth

    brand_urls = [
        "https://www.bluehost.com/",
        "https://www.domain.com/",
        "https://www.hostgator.com/",
        "https://www.networksolutions.com/",
        "https://www.register.com/",
        "https://www.web.com/"
    ]
    all_data = {}

    logging.info("Starting media elements reviewer")
    try:
        # Use ThreadPoolExecutor for parallel crawling
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_brand = {executor.submit(crawl_brand_for_images, url, max_depth): url for url in brand_urls}
            for future in as_completed(future_to_brand):
                brand_name, data = future.result()
                all_data[brand_name] = data

        # Save all data to a single Excel file
        excel_path = "media_elements_review.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            # Create a summary sheet
            all_images = []
            for brand_name, images in all_data.items():
                for img in images:
                    img['Brand'] = brand_name
                    all_images.append(img)

            if all_images:
                df_all = pd.DataFrame(all_images)
                df_all.to_excel(writer, sheet_name='All Images', index=False)

                # Create individual brand sheets
                for brand_name, images in all_data.items():
                    if images:  # Only create sheet if there's data
                        df_brand = pd.DataFrame(images)
                        df_brand.to_excel(writer, sheet_name=brand_name[:31], index=False)  # Excel limits sheet names to 31 chars
            else:
                # Create empty sheet if no images found
                pd.DataFrame().to_excel(writer, sheet_name='No Images Found')

        print(f"\nImage analysis completed. Data saved to '{excel_path}'")

        # Generate HTML report
        html_path = "media_elements_report.html"
        generate_html_report(all_data, html_path)

        logging.info("Media elements review completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
