#!/usr/bin/env python3
"""
Waterfall Chart Utilities
Additional helper functions for enhanced waterfall chart generation and analysis
"""

import logging
from urllib.parse import urlparse


class WaterfallUtils:
    """Utility class for waterfall chart enhancements"""

    @staticmethod
    def validate_network_timing(start_time, end_time, duration):
        """Validate network timing data for consistency"""
        try:
            # Basic validation
            if start_time < 0 or end_time < 0 or duration < 0:
                return False

            # Check if end_time matches start_time + duration (within tolerance)
            calculated_end = start_time + duration
            tolerance = 5  # 5ms tolerance

            if abs(end_time - calculated_end) > tolerance:
                # Use calculated end time if original is inconsistent
                return start_time, calculated_end, duration

            return start_time, end_time, duration

        except (TypeError, ValueError):
            return False

    @staticmethod
    def extract_har_data(lighthouse_report):
        """Extract HAR (HTTP Archive) data if available for enhanced waterfall"""
        try:
            # Some Lighthouse versions include HAR data
            if "artifacts" in lighthouse_report and "devtoolsLogs" in lighthouse_report["artifacts"]:
                # Process devtools logs to extract network events
                devtools_logs = lighthouse_report["artifacts"]["devtoolsLogs"].get("defaultPass", [])

                network_events = []
                for log_entry in devtools_logs:
                    if log_entry.get("method") == "Network.responseReceived":
                        params = log_entry.get("params", {})
                        response = params.get("response", {})

                        network_events.append({
                            "requestId": params.get("requestId"),
                            "url": response.get("url"),
                            "status": response.get("status"),
                            "mimeType": response.get("mimeType"),
                            "headers": response.get("headers", {}),
                            "timing": response.get("timing", {}),
                            "timestamp": log_entry.get("timestamp")
                        })

                return network_events

        except Exception as e:
            logging.debug(f"Could not extract HAR data: {str(e)}")

        return []

    @staticmethod
    def add_critical_path_analysis(network_requests, lighthouse_report):
        """Add critical path analysis to identify render-blocking resources"""
        try:
            critical_requests = []

            # Get critical request chains from Lighthouse
            if "audits" in lighthouse_report and "critical-request-chains" in lighthouse_report["audits"]:
                critical_chains = lighthouse_report["audits"]["critical-request-chains"].get("details", {}).get("chains", {})

                def extract_critical_urls(chains, parent_url=None):
                    urls = []
                    for url, chain_data in chains.items():
                        urls.append(url)
                        if "children" in chain_data:
                            urls.extend(extract_critical_urls(chain_data["children"], url))
                    return urls

                critical_urls = extract_critical_urls(critical_chains)

                # Mark requests as critical
                for req in network_requests:
                    if req['url'] in critical_urls:
                        req['is_critical'] = True
                        critical_requests.append(req)
                    else:
                        req['is_critical'] = False

            return critical_requests

        except Exception as e:
            logging.debug(f"Could not perform critical path analysis: {str(e)}")
            return []

    @staticmethod
    def calculate_waterfall_metrics(network_requests):
        """Calculate additional metrics from waterfall data"""
        if not network_requests:
            return {}

        try:
            # Calculate various waterfall metrics
            total_requests = len(network_requests)
            total_size = sum(req.get('transferSize', 0) for req in network_requests)
            cached_requests = sum(1 for req in network_requests if req.get('fromCache', False))

            # Calculate resource type distribution
            resource_distribution = {}
            for req in network_requests:
                resource_type = req.get('resourceType', 'Other')
                resource_distribution[resource_type] = resource_distribution.get(resource_type, 0) + 1

            # Calculate timing metrics
            start_times = [req['startTime'] for req in network_requests if 'startTime' in req]
            end_times = [req['endTime'] for req in network_requests if 'endTime' in req]

            total_load_time = max(end_times) - min(start_times) if start_times and end_times else 0

            # Calculate parallel vs sequential loading efficiency
            overlapping_requests = 0
            for i, req1 in enumerate(network_requests):
                for req2 in network_requests[i+1:]:
                    if (req1['startTime'] < req2['endTime'] and
                        req2['startTime'] < req1['endTime']):
                        overlapping_requests += 1

            parallelization_ratio = overlapping_requests / max(1, total_requests * (total_requests - 1) / 2)

            return {
                'total_requests': total_requests,
                'total_size': total_size,
                'cached_requests': cached_requests,
                'cache_hit_ratio': cached_requests / total_requests if total_requests > 0 else 0,
                'resource_distribution': resource_distribution,
                'total_load_time': total_load_time,
                'parallelization_ratio': parallelization_ratio,
                'avg_request_size': total_size / total_requests if total_requests > 0 else 0
            }

        except Exception as e:
            logging.debug(f"Could not calculate waterfall metrics: {str(e)}")
            return {}

    @staticmethod
    def identify_performance_bottlenecks(network_requests):
        """Identify potential performance bottlenecks from waterfall data"""
        bottlenecks = []

        if not network_requests:
            return bottlenecks

        try:
            # Sort by start time
            sorted_requests = sorted(network_requests, key=lambda x: x.get('startTime', 0))

            # Identify long-running requests
            long_requests = [req for req in network_requests if req.get('duration', 0) > 2000]  # > 2 seconds
            if long_requests:
                bottlenecks.append({
                    'type': 'slow_requests',
                    'severity': 'high',
                    'description': f'Found {len(long_requests)} slow requests (>2s)',
                    'requests': [req['url'] for req in long_requests[:5]]  # Limit to first 5
                })

            # Identify large resources
            large_requests = [req for req in network_requests if req.get('transferSize', 0) > 1024 * 1024]  # > 1MB
            if large_requests:
                bottlenecks.append({
                    'type': 'large_resources',
                    'severity': 'medium',
                    'description': f'Found {len(large_requests)} large resources (>1MB)',
                    'requests': [req['url'] for req in large_requests[:5]]
                })

            # Identify sequential loading patterns
            sequential_chains = []
            for i, req in enumerate(sorted_requests[:-1]):
                next_req = sorted_requests[i + 1]
                if abs(req.get('endTime', 0) - next_req.get('startTime', 0)) < 10:  # < 10ms gap
                    sequential_chains.append((req['url'], next_req['url']))

            if len(sequential_chains) > 5:
                bottlenecks.append({
                    'type': 'sequential_loading',
                    'severity': 'medium',
                    'description': f'Found {len(sequential_chains)} sequential request chains that could be parallelized'
                })

            # Identify too many requests from same domain
            domain_counts = {}
            for req in network_requests:
                domain = urlparse(req.get('url', '')).netloc
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            domains_with_many_requests = {domain: count for domain, count in domain_counts.items() if count > 20}
            if domains_with_many_requests:
                bottlenecks.append({
                    'type': 'too_many_requests',
                    'severity': 'medium',
                    'description': f'Some domains have too many requests: {domains_with_many_requests}'
                })

        except Exception as e:
            logging.debug(f"Could not identify performance bottlenecks: {str(e)}")

        return bottlenecks

    @staticmethod
    def generate_waterfall_summary(network_requests, metrics=None):
        """Generate a summary of waterfall performance"""
        if not network_requests:
            return "No network data available"

        try:
            total_requests = len(network_requests)
            total_size = sum(req.get('transferSize', 0) for req in network_requests)
            cached_requests = sum(1 for req in network_requests if req.get('fromCache', False))
            failed_requests = sum(1 for req in network_requests if req.get('statusCode', 200) >= 400)

            # Format total size
            if total_size > 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f}MB"
            elif total_size > 1024:
                size_str = f"{total_size / 1024:.1f}KB"
            else:
                size_str = f"{total_size}B"

            summary_parts = [
                f"{total_requests} total requests",
                f"{size_str} transferred",
                f"{cached_requests} cached ({cached_requests/total_requests*100:.1f}%)" if cached_requests > 0 else "0 cached"
            ]

            if failed_requests > 0:
                summary_parts.append(f"{failed_requests} failed")

            return " • ".join(summary_parts)

        except Exception as e:
            logging.debug(f"Could not generate waterfall summary: {str(e)}")
            return "Error generating summary"
