"""
Postman Collection Parser for API Test Generation

This module provides functionality to parse Postman collection files and extract
API endpoint information in a format compatible with the API test generator.
"""

import json
import streamlit as st


def parse_postman_collection(collection_data):
    """
    Parse a Postman collection and extract endpoints.

    Args:
        collection_data (dict): The parsed JSON data of the Postman collection

    Returns:
        list: A list of endpoint dictionaries compatible with the app's format
    """
    endpoints = []

    # Process items recursively to handle nested folders
    def process_items(items):
        for item in items:
            # Skip folders without requests but process their items
            if "item" in item and isinstance(item["item"], list):
                process_items(item["item"])
            # Process actual request items
            elif "request" in item:
                request = item["request"]

                # Extract HTTP method
                method = request.get("method", "GET")

                # Extract URL information
                url = request.get("url", {})
                if isinstance(url, str):
                    # Handle case where url is a string instead of an object
                    path = url.split("?")[0]  # Remove query parameters
                else:
                    # Extract path from url object
                    path = ""
                    if "path" in url:
                        path = "/" + "/".join(url.get("path", []))
                    elif "raw" in url:
                        # Extract path from raw URL
                        raw_url = url.get("raw", "")
                        try:
                            # Remove protocol, host and query
                            path_part = raw_url.split("://")[-1].split("/", 1)
                            if len(path_part) > 1:
                                path = "/" + path_part[1].split("?")[0]
                        except:
                            path = "/"

                # Extract parameters
                parameters = []

                # Extract path variables if any
                if isinstance(url, dict) and "variable" in url:
                    for var in url.get("variable", []):
                        parameters.append({
                            "name": var.get("key", ""),
                            "in": "path",
                            "required": True,
                            "description": var.get("description", ""),
                        })

                # Extract query parameters if any
                if isinstance(url, dict) and "query" in url:
                    for query in url.get("query", []):
                        parameters.append({
                            "name": query.get("key", ""),
                            "in": "query",
                            "required": not query.get("disabled", False),
                            "description": query.get("description", ""),
                        })

                # Extract headers
                headers = request.get("header", [])
                for header in headers:
                    if not header.get("disabled", False):
                        parameters.append({
                            "name": header.get("key", ""),
                            "in": "header",
                            "required": True,
                            "description": header.get("description", ""),
                        })

                # Extract request body if it exists
                if "body" in request and request["body"] is not None:
                    body = request["body"]
                    body_description = "Request body"

                    # Try to extract example from body based on mode
                    body_example = {}
                    body_mode = body.get("mode", "")

                    if body_mode == "raw" and "raw" in body:
                        try:
                            if isinstance(body["raw"], str):
                                body_example = body["raw"]
                        except:
                            pass
                    elif body_mode == "formdata" and "formdata" in body:
                        body_example = {item.get("key", ""): item.get("value", "") for item in body.get("formdata", [])}

                    # Add body parameter
                    parameters.append({
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "description": body_description,
                        "example": body_example
                    })

                # Create endpoint dictionary
                endpoint = {
                    "path": path,
                    "method": method,
                    "parameters": parameters,
                    "description": item.get("name", "") + ((" - " + item.get("description", "")) if item.get("description") else "")
                }

                endpoints.append(endpoint)

    # Start processing from the top-level items
    if "item" in collection_data:
        process_items(collection_data["item"])

    return endpoints


def load_postman_collection(uploaded_file):
    """
    Load and parse a Postman collection file.

    Args:
        uploaded_file: The uploaded Postman collection file

    Returns:
        list: A list of endpoints extracted from the collection
    """
    try:
        content = uploaded_file.read()
        collection_data = json.loads(content)

        # Basic validation that this is a Postman collection
        if "info" in collection_data and "schema" in collection_data["info"]:
            schema_url = collection_data["info"]["schema"]
            if "postman" in schema_url.lower():
                # This looks like a valid Postman collection
                endpoints = parse_postman_collection(collection_data)

                return endpoints
            else:
                st.error(f"The uploaded file '{uploaded_file.name}' doesn't appear to be a valid Postman collection.")
                return None
        else:
            st.error(f"The uploaded file '{uploaded_file.name}' is missing required Postman collection structure.")
            return None

    except Exception as e:
        st.error(f"Error parsing Postman collection file '{uploaded_file.name}': {str(e)}")
        return None


def load_multiple_postman_collections(uploaded_files):
    """
    Load and parse multiple Postman collection files.

    Args:
        uploaded_files: List of uploaded Postman collection files

    Returns:
        tuple: (combined_endpoints, results_summary)
            - combined_endpoints: list of all endpoints extracted from the collections
            - results_summary: dictionary with success and error information
    """
    all_endpoints = []
    results = {
        "successful_files": [],
        "failed_files": [],
        "total_endpoints": 0
    }

    for uploaded_file in uploaded_files:
        try:
            endpoints = load_postman_collection(uploaded_file)
            if endpoints:
                all_endpoints.extend(endpoints)
                results["successful_files"].append(uploaded_file.name)
                results["total_endpoints"] += len(endpoints)
            else:
                results["failed_files"].append(uploaded_file.name)
        except Exception as e:
            results["failed_files"].append(f"{uploaded_file.name} (Error: {str(e)})")

    return all_endpoints, results


def show_postman_ui():
    """
    Display the Postman collection upload interface.

    Returns:
        list or None: A list of parsed endpoints if successful, None otherwise
    """
    st.subheader("Upload Postman Collection")

    # Option to choose between single file and bulk upload
    upload_mode = st.radio(
        "Upload mode",
        ["Single Collection", "Bulk Upload (Multiple Collections)"],
        horizontal=True
    )

    st.markdown("""
    Upload Postman collection export(s) (.json) to extract API endpoints for test generation.
    
    1. In Postman, click on a collection
    2. Click the "..." menu (three dots)
    3. Select "Export"
    4. Choose "Collection v2.1" format
    5. Upload the JSON file(s) below
    """)

    if upload_mode == "Single Collection":
        uploaded_file = st.file_uploader("Upload Postman Collection", type=["json"])

        if uploaded_file:
            endpoints = load_postman_collection(uploaded_file)

            if endpoints:
                st.success(f"Successfully extracted {len(endpoints)} endpoints from the Postman collection.")

                # Display a preview of the endpoints
                with st.expander("Preview Extracted Endpoints"):
                    for i, endpoint in enumerate(endpoints):
                        st.markdown(f"**{i+1}. {endpoint['method']} {endpoint['path']}**")
                        st.markdown(f"Description: {endpoint['description']}")

                        if endpoint['parameters']:
                            st.markdown("Parameters:")
                            for param in endpoint['parameters']:
                                st.markdown(f"- {param['name']} ({param['in']}){' (required)' if param.get('required', False) else ''}")

                        st.divider()

                return endpoints
    else:  # Bulk Upload mode
        uploaded_files = st.file_uploader("Upload Multiple Postman Collections", type=["json"], accept_multiple_files=True)

        if uploaded_files:
            st.info(f"Processing {len(uploaded_files)} Postman collection files...")

            endpoints, results = load_multiple_postman_collections(uploaded_files)

            if endpoints:
                # Show summary of processed files
                st.success(f"Successfully extracted {results['total_endpoints']} total endpoints from {len(results['successful_files'])} collections.")

                if results['failed_files']:
                    st.warning(f"Failed to process {len(results['failed_files'])} files: {', '.join(results['failed_files'])}")

                # Display a preview of the endpoints with file sources
                with st.expander("Preview Extracted Endpoints"):
                    for i, endpoint in enumerate(endpoints):
                        st.markdown(f"**{i+1}. {endpoint['method']} {endpoint['path']}**")
                        st.markdown(f"Description: {endpoint['description']}")

                        if endpoint['parameters']:
                            st.markdown("Parameters:")
                            for param in endpoint['parameters']:
                                st.markdown(f"- {param['name']} ({param['in']}){' (required)' if param.get('required', False) else ''}")

                        st.divider()

                return endpoints
            else:
                st.error("No valid endpoints were extracted from any of the uploaded collections.")
                return None

    return None
