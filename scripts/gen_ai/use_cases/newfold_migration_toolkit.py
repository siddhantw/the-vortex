"""
Newfold Migration Toolkit Module
A comprehensive CSRT and RAS operations module for product lifecycle management and migration testing.

This module provides:
- Product lifecycle operations (Deactivate, Reactivate, Delete, Renew)
- CSRT API integration for product operations
- RAS (Renewal Aging System) database updates
- EDB Oracle database connectivity (STG and QAMain environments)
- Bulk SKU processing with intelligent analysis
- Real-time progress tracking and results visualization
- AI-powered insights and recommendations using Azure OpenAI
- Migration source tracking and validation
- Test result persistence and history

Core Capabilities:
1. CSRT Operations: Deactivate, Reactivate, Delete
2. RAS Operations: Renew, Deactivate, Delete
3. Product Discovery: Find eligible products based on criteria
4. Batch Processing: Process multiple SKUs efficiently
5. AI Analysis: Get intelligent recommendations and insights
"""

import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import os
import re
from enum import Enum
import time
import threading
from collections import defaultdict

# Add tests/configs to path for database config
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'tests', 'configs')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

# Try to import database config and oracledb
try:
    import database_config_variables as db_config
    DB_CONFIG_AVAILABLE = True
except ImportError:
    DB_CONFIG_AVAILABLE = False
    db_config = None

# Try to import CSRT config
try:
    import csrt_config_variables as csrt_config
    CSRT_CONFIG_AVAILABLE = True
except ImportError:
    CSRT_CONFIG_AVAILABLE = False
    csrt_config = None

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("NewfoldMigrationToolkit", level=logging.INFO, log_file="newfold_migration_toolkit.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Oracle database driver
try:
    import oracledb
    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False

# Import Azure OpenAI Client for AI-powered insights
AZURE_OPENAI_AVAILABLE = False
AzureOpenAIClient = None
try:
    from azure_openai_client import AzureOpenAIClient as _AzureOpenAIClient
    AzureOpenAIClient = _AzureOpenAIClient
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    try:
        from gen_ai.azure_openai_client import AzureOpenAIClient as _AzureOpenAIClient
        AzureOpenAIClient = _AzureOpenAIClient
        AZURE_OPENAI_AVAILABLE = True
    except ImportError:
        pass


class OperationType(Enum):
    """Operation types for product lifecycle management"""
    DEACTIVATE = "DEACTIVATE"
    REACTIVATE = "REACTIVATE"
    DELETE = "DELETE"
    RENEW = "RENEW"


class OperationOrigin(Enum):
    """Origin of the operation - RAS or CSRT"""
    RAS = "RAS"
    CSRT = "CSRT"


class NewfoldMigrationToolkit:
    """Main class for Newfold Migration Toolkit functionality"""

    def __init__(self):
        self.connection = None
        self.ai_client = None
        self.history_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "newfold_migration_history.json"
        )
        self._init_ai_client()

    def _init_ai_client(self):
        """Initialize Azure OpenAI client for AI-powered insights"""
        if AZURE_OPENAI_AVAILABLE and AzureOpenAIClient:
            try:
                self.ai_client = AzureOpenAIClient()
                logger.info("Azure OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
                self.ai_client = None
        else:
            logger.warning("Azure OpenAI not available")
            self.ai_client = None

    def connect_to_database(self, environment: str) -> Tuple[bool, str, Any]:
        """
        Connect to EDB Oracle database for the specified environment
        Fetches all configuration from database_config_variables.py - no hardcoded values

        Args:
            environment: 'QAMain', 'Stage', 'JarvisQA1', 'JarvisQA2', or 'Production'

        Returns:
            Tuple of (success, message, connection)
        """
        if not ORACLEDB_AVAILABLE:
            return False, "Oracle database driver not available. Install oracledb package.", None

        if not DB_CONFIG_AVAILABLE:
            return False, "Database configuration not available. Check database_config_variables.py", None

        try:
            # Map environment names to config keys
            env_mapping = {
                'QAMain': 'QAMain',
                'Stage': 'Staging',
                'JarvisQA1': 'JarvisQA1',
                'JarvisQA2': 'JarvisQA2',
                'Production': 'Production'
            }

            config_key = env_mapping.get(environment)
            if not config_key:
                return False, f"Invalid environment: {environment}. Valid options: QAMain, Stage, JarvisQA1, JarvisQA2, Production", None

            # Try DATABASE_CONFIG first for environments that have IP addresses
            # Then fall back to old-style variables
            # This handles cases where old-style has unresolvable hostnames but DATABASE_CONFIG has IPs
            host = None
            port = None
            service_name = None
            sid = None
            username = None
            password = None

            # First, check DATABASE_CONFIG (often has IP addresses which are more reliable)
            if hasattr(db_config, 'DATABASE_CONFIG') and config_key in db_config.DATABASE_CONFIG:
                config = db_config.DATABASE_CONFIG[config_key]
                host = config.get('host')
                port = config.get('port')
                # Handle both 'sid' and 'service_name' keys
                service_name = config.get('service_name')
                sid = config.get('sid')
                username = config.get('username')
                password = config.get('password')

            # Second, fetch from old-style variables to fill in missing values or override if better
            # For QAMain, old-style has better service_name, so override if we got SID from DATABASE_CONFIG
            if environment == 'Stage':
                if not host: host = getattr(db_config, 'stage_db_host_variable', None)
                if not port: port = getattr(db_config, 'stage_db_port_variable', None)
                # For Stage, use DATABASE_CONFIG service/sid if available, else old-style
                if not service_name and not sid:
                    service_name = getattr(db_config, 'stage_db_service_variable', None)
            elif environment == 'QAMain':
                if not host: host = getattr(db_config, 'qamain_db_host_variable', None)
                if not port: port = getattr(db_config, 'qamain_db_port_variable', None)
                # For QAMain, old-style has correct service_name, so override the wrong SID
                old_service = getattr(db_config, 'qamain_db_service_variable', None)
                if old_service:
                    service_name = old_service
                    sid = None  # Clear wrong SID
            elif environment == 'JarvisQA1':
                if not host: host = getattr(db_config, 'jarvisqa1_db_host_variable', None)
                if not port: port = getattr(db_config, 'jarvisqa1_db_port_variable', None)
                if not service_name:
                    service_name = getattr(db_config, 'jarvisqa1_db_service_variable', None)
            elif environment == 'JarvisQA2':
                if not host: host = getattr(db_config, 'jarvisqa2_db_host_variable', None)
                if not port: port = getattr(db_config, 'jarvisqa2_db_port_variable', None)
                if not service_name:
                    service_name = getattr(db_config, 'jarvisqa2_db_service_variable', None)
            elif environment == 'Production':
                return False, "Production environment connection not configured for safety reasons.", None

            # Get default credentials if not already set
            if not username:
                username = getattr(db_config, 'default_db_user_variable', None)
            if not password:
                password = getattr(db_config, 'default_db_pass_variable', None)


            # Validate we have either service_name or sid
            has_service_identifier = service_name or sid

            # Final validation - check if we have all required fields
            if not all([host, port, has_service_identifier, username, password]):
                missing = []
                if not host: missing.append('host')
                if not port: missing.append('port')
                if not has_service_identifier: missing.append('service_name or sid')
                if not username: missing.append('username')
                if not password: missing.append('password')
                return False, f"Missing configuration for {environment}: {', '.join(missing)}. Check database_config_variables.py", None

            # Convert port to int if it's a string
            if port and isinstance(port, str):
                port = int(port)

            # Create DSN - prefer service_name over SID
            if service_name:
                dsn = oracledb.makedsn(host, port, service_name=service_name)
            elif sid:
                dsn = oracledb.makedsn(host, port, sid=sid)
            else:
                return False, f"No service_name or SID found for {environment}. Check database_config_variables.py", None

            # Connect to database
            connection = oracledb.connect(user=username, password=password, dsn=dsn)

            logger.info(f"Successfully connected to {environment} database at {host}:{port}")
            return True, f"Connected to {environment} successfully", connection

        except Exception as e:
            error_msg = f"Failed to connect to {environment}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None

    def parse_skus(self, sku_input: str) -> List[str]:
        """
        Parse SKU input supporting multiple formats:
        - Newlines
        - Commas
        - Spaces
        - Mixed formats

        Args:
            sku_input: Raw SKU input string

        Returns:
            List of cleaned SKU strings
        """
        if not sku_input:
            return []

        # Replace common separators with spaces
        cleaned = re.sub(r'[,\n\r\t]+', ' ', sku_input)

        # Split by spaces and filter empty strings
        skus = [sku.strip() for sku in cleaned.split() if sku.strip()]

        return skus

    def parse_migration_sources(self, source_input: str) -> List[str]:
        """Parse migration sources input"""
        if not source_input:
            return []
        return self.parse_skus(source_input)

    def get_parent_channel_id_for_brand(self, brand: Optional[str]) -> str:
        """
        Map brand names to PARENT_CHANNEL_ID

        Args:
            brand: Brand name (e.g., 'bluehost', 'hostgator')

        Returns:
            PARENT_CHANNEL_ID string
        """
        brand_mapping = {
            'bluehost': '8',
            'hostgator': '9',
            'ipage': '17',
            'netweb': '1',
        }

        if not brand:
            return '8'  # Default to Bluehost

        brand_lower = brand.lower().strip()
        return brand_mapping.get(brand_lower, '8')

    def find_products(
        self,
        connection: Any,
        skus: List[str],
        operations: List[Dict[str, str]],
        migration_sources: List[str],
        brand: Optional[str],
        debug_mode: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find PROD_INST_IDs for given SKUs and operations

        Args:
            connection: Database connection
            skus: List of SKU codes
            operations: List of operation dicts with 'type' and 'origin'
            migration_sources: List of migration sources to filter
            brand: Brand name for PARENT_CHANNEL_ID mapping
            debug_mode: If True, stores SQL queries and parameters for debugging

        Returns:
            Dictionary mapping SKUs to list of product results
        """
        if not connection:
            return {}

        parent_channel_id = self.get_parent_channel_id_for_brand(brand)
        results = defaultdict(list)

        # Store debug info if requested
        debug_info = []

        cursor = connection.cursor()

        try:
            for sku in skus:
                for operation in operations:
                    op_type = operation['type']
                    op_origin = operation['origin']

                    # Build query based on operation type
                    lifecycle_condition = self._get_lifecycle_condition(op_type, op_origin)
                    mig_source_condition = self._get_migration_source_condition(migration_sources)
                    wallet_join, wallet_where = self._get_wallet_conditions(op_type, op_origin)

                    # Build SQL query
                    sql = f"""
                        SELECT 
                            PCB.PROD_CD,
                            PI.PARENT_CHANNEL_ID,
                            PI.PROD_INST_ID,
                            PI.MIG_SOURCE,
                            PI.SUBSCRIPTION_UNIT,
                            PI.SPECIAL_ID,
                            PI.LIFECYCLE_CD_ID,
                            PI.AUTO_RENEW_FLAG,
                            PI.EXP_DATE
                        FROM PROD_CODE_BASE PCB
                        INNER JOIN PROD_INST PI ON PI.PROD_ID = PCB.PROD_ID
                        {wallet_join}
                        WHERE PI.PARENT_CHANNEL_ID = :parent_channel_id
                        {mig_source_condition}
                        {lifecycle_condition}
                        {wallet_where}
                        AND PCB.PROD_CD = :sku
                        FETCH FIRST 100 ROWS ONLY
                    """

                    # Execute query
                    params = {'parent_channel_id': parent_channel_id, 'sku': sku}

                    # Add migration source parameters if provided
                    if migration_sources:
                        for i, mig_source in enumerate(migration_sources):
                            params[f'mig_source_{i}'] = mig_source

                    # Store debug info
                    if debug_mode:
                        debug_info.append({
                            'sku': sku,
                            'operation': f"{op_origin} {op_type}",
                            'sql': sql,
                            'params': params.copy()
                        })

                    cursor.execute(sql, params)
                    rows = cursor.fetchall()

                    if rows:
                        for row in rows:
                            results[sku].append({
                                'sku': sku,
                                'operation_type': op_type,
                                'operation_origin': op_origin,
                                'prod_inst_id': row[2],
                                'mig_source': row[3],
                                'lifecycle_cd_id': row[6],
                                'auto_renew_flag': row[7],
                                'exp_date': row[8],
                                'success': True,
                                'error': None
                            })
                        logger.info(f"Found {len(rows)} products for SKU: {sku}, operation: {op_type}")
                    else:
                        # Log the query details for debugging
                        logger.warning(f"No products found for SKU: {sku}, PARENT_CHANNEL_ID: {parent_channel_id}, operation: {op_type}")
                        logger.debug(f"Query params: {params}")

                        # Build detailed error message
                        error_details = f'No products found for SKU "{sku}" with PARENT_CHANNEL_ID={parent_channel_id}'
                        if migration_sources:
                            error_details += f' and migration sources: {", ".join(migration_sources)}'
                        error_details += f'. Operation filter: {op_type}'

                        results[sku].append({
                            'sku': sku,
                            'operation_type': op_type,
                            'operation_origin': op_origin,
                            'prod_inst_id': None,
                            'mig_source': None,
                            'lifecycle_cd_id': None,
                            'auto_renew_flag': None,
                            'exp_date': None,
                            'success': False,
                            'error': error_details
                        })

        finally:
            cursor.close()

        # Store debug info in results if debug mode enabled
        if debug_mode and debug_info:
            results['__debug_info__'] = debug_info

        return dict(results)

    def _get_lifecycle_condition(self, operation_type: str, origin: str) -> str:
        """Get SQL condition for lifecycle based on operation type"""
        if operation_type == "DELETE":
            return "AND (PI.LIFECYCLE_CD_ID = 10 OR PI.LIFECYCLE_CD_ID = 11)"
        elif operation_type == "DEACTIVATE":
            return "AND PI.LIFECYCLE_CD_ID = 10"
        elif operation_type == "RENEW":
            return "AND PI.LIFECYCLE_CD_ID = 10"
        elif operation_type == "REACTIVATE":
            return "AND PI.LIFECYCLE_CD_ID = 11"
        return ""

    def _get_migration_source_condition(self, migration_sources: List[str]) -> str:
        """Get SQL condition for migration sources"""
        if not migration_sources:
            return ""

        # Oracle bind variables for IN clause
        placeholders = ', '.join([f":mig_source_{i}" for i in range(len(migration_sources))])
        return f"AND PI.MIG_SOURCE IN ({placeholders})"

    def _get_wallet_conditions(self, operation_type: str, origin: str) -> Tuple[str, str]:
        """Get wallet join and where conditions for RENEW operations"""
        if operation_type == "RENEW" and origin == "RAS":
            wallet_join = """
                INNER JOIN ACCOUNT_WALLET_ITEM AWI ON AWI.ACCOUNT_ID = PI.ACCOUNT_ID
            """
            wallet_where = """
                AND AWI.LAST_FOUR IN ('1111', '0005', '4444')
                AND AWI.CC_EXP_DATE >= ADD_MONTHS(TRUNC(SYSDATE), 1)
            """
            return wallet_join, wallet_where
        return "", ""

    def execute_ras_operations(
        self,
        connection: Any,
        products: List[Dict[str, Any]],
        operation_type: str
    ) -> List[Dict[str, Any]]:
        """
        Execute RAS operations (database updates) for products

        Args:
            connection: Database connection
            products: List of product dictionaries
            operation_type: Type of operation (DEACTIVATE, DELETE, RENEW)

        Returns:
            List of results with success/failure status
        """
        if not connection or not products:
            return []

        cursor = connection.cursor()
        results = []

        try:
            for product in products:
                prod_inst_id = product.get('prod_inst_id')
                if not prod_inst_id:
                    results.append({
                        **product,
                        'success': False,
                        'error': 'No PROD_INST_ID available'
                    })
                    continue

                # Build UPDATE query based on operation type
                sql = self._get_ras_update_sql(operation_type)

                try:
                    cursor.execute(sql, {'prod_inst_id': prod_inst_id})
                    connection.commit()

                    results.append({
                        **product,
                        'success': True,
                        'rows_updated': cursor.rowcount,
                        'error': None
                    })

                    logger.info(f"RAS {operation_type} executed successfully for {prod_inst_id}")

                except Exception as e:
                    connection.rollback()
                    results.append({
                        **product,
                        'success': False,
                        'error': str(e)
                    })
                    logger.error(f"RAS {operation_type} failed for {prod_inst_id}: {e}")

        finally:
            cursor.close()

        return results

    def _get_ras_update_sql(self, operation_type: str) -> str:
        """Get SQL UPDATE statement for RAS operations"""
        if operation_type == "DEACTIVATE":
            return """
                UPDATE PROD_INST 
                SET AUTO_RENEW_FLAG = 'N', 
                    EXP_DATE = TRUNC(SYSDATE-21)
                WHERE PROD_INST_ID = :prod_inst_id
            """
        elif operation_type == "DELETE":
            return """
                UPDATE PROD_INST 
                SET AUTO_RENEW_FLAG = 'N', 
                    EXP_DATE = TRUNC(SYSDATE-42), 
                    LIFECYCLE_CD_ID = 11
                WHERE PROD_INST_ID = :prod_inst_id
            """
        elif operation_type == "RENEW":
            return """
                UPDATE PROD_INST 
                SET EXP_DATE = TRUNC(SYSDATE), 
                    AUTO_RENEW_FLAG = 'Y'
                WHERE PROD_INST_ID = :prod_inst_id
            """
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")

    def execute_csrt_operations(
        self,
        environment: str,
        products: List[Dict[str, Any]],
        operation_type: str
    ) -> List[Dict[str, Any]]:
        """
        Execute CSRT operations via HTTP API calls

        Args:
            environment: Environment name (QAMain, Stage, etc.)
            products: List of product dictionaries
            operation_type: Type of operation (DEACTIVATE, REACTIVATE, DELETE)

        Returns:
            List of results with success/failure status
        """
        if not CSRT_CONFIG_AVAILABLE or not csrt_config:
            logger.warning("CSRT config not available, skipping CSRT operations")
            return [{
                **product,
                'success': False,
                'error': 'CSRT configuration not available'
            } for product in products]

        if not products:
            return []

        # Get CSRT base URL for environment
        base_url = self._get_csrt_base_url(environment)
        if not base_url:
            logger.error(f"No CSRT base URL configured for environment: {environment}")
            return [{
                **product,
                'success': False,
                'error': f'No CSRT base URL for {environment}'
            } for product in products]

        # Get API endpoint for operation
        endpoint = self._get_csrt_endpoint(operation_type)
        full_url = f"{base_url}{endpoint}"

        results = []

        for product in products:
            prod_inst_id = product.get('prod_inst_id')
            sku = product.get('sku')

            if not prod_inst_id or not sku:
                results.append({
                    **product,
                    'success': False,
                    'error': 'Missing PROD_INST_ID or SKU'
                })
                continue

            # Build request payload
            payload = {
                'prodInstId': prod_inst_id,
                'sku': sku,
                'operation': operation_type
            }

            try:
                # Make HTTP POST request to CSRT API
                response = requests.post(
                    full_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )

                if response.status_code == 200:
                    results.append({
                        **product,
                        'success': True,
                        'csrt_response': response.json() if response.text else {},
                        'error': None
                    })
                    logger.info(f"CSRT {operation_type} successful for {prod_inst_id}")
                else:
                    results.append({
                        **product,
                        'success': False,
                        'error': f'HTTP {response.status_code}: {response.text[:200]}'
                    })
                    logger.error(f"CSRT {operation_type} failed for {prod_inst_id}: {response.status_code}")

            except requests.exceptions.Timeout:
                results.append({
                    **product,
                    'success': False,
                    'error': 'Request timeout (30s)'
                })
                logger.error(f"CSRT API timeout for {prod_inst_id}")

            except requests.exceptions.RequestException as e:
                results.append({
                    **product,
                    'success': False,
                    'error': f'Request failed: {str(e)}'
                })
                logger.error(f"CSRT API request failed for {prod_inst_id}: {e}")

        return results

    def _get_csrt_base_url(self, environment: str) -> Optional[str]:
        """Get CSRT base URL for environment"""
        if not CSRT_CONFIG_AVAILABLE or not csrt_config:
            return None

        env_mapping = {
            'QAMain': 'qamain_csrtools_base_url',
            'Stage': 'jarvisqa1_csrtools_base_url',  # Stage uses JarvisQA1 CSRT
            'JarvisQA1': 'jarvisqa1_csrtools_base_url',
            'JarvisQA2': 'jarvisqa2_csrtools_base_url',
            'Production': 'prod_csrtools_base_url'
        }

        attr_name = env_mapping.get(environment)
        if attr_name:
            return getattr(csrt_config, attr_name, None)
        return None

    def _get_csrt_endpoint(self, operation_type: str) -> str:
        """Get CSRT API endpoint for operation type"""
        # Default endpoint pattern - customize based on actual CSRT API
        endpoints = {
            'DEACTIVATE': '/api/product/deactivate',
            'REACTIVATE': '/api/product/reactivate',
            'DELETE': '/api/product/delete'
        }
        return endpoints.get(operation_type, '/api/product/action')

    def save_test_results(
        self,
        environment: str,
        test_name: str,
        brand: Optional[str],
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Save test results to JSON file for tracking

        Args:
            environment: STG or QAMain
            test_name: Name of the test
            brand: Brand name
            results: List of result dictionaries

        Returns:
            Filename of saved results
        """
        try:
            # Create ras_updates directory if it doesn't exist
            ras_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "Newfold Migration",
                "ras_updates"
            )
            os.makedirs(ras_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{test_name}_{timestamp}.json"
            filepath = os.path.join(ras_dir, filename)

            # Prepare data
            data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'Completed',
                'environment': environment,
                'testName': test_name,
                'brand': brand,
                'updates': results
            }

            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Test results saved to: {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return ""

    def get_ai_insights(
        self,
        results: List[Dict[str, Any]],
        test_context: Dict[str, Any]
    ) -> str:
        """
        Get AI-powered insights and recommendations for test results

        Args:
            results: List of test results
            test_context: Context information (environment, operations, etc.)

        Returns:
            AI-generated insights as markdown text
        """
        if not self.ai_client:
            return "AI insights not available (Azure OpenAI client not initialized)"

        # Check if client is properly configured
        if hasattr(self.ai_client, 'is_configured') and not self.ai_client.is_configured():
            return "AI insights not available. Please configure Azure OpenAI environment variables:\n- AZURE_OPENAI_ENDPOINT\n- AZURE_OPENAI_API_KEY\n- AZURE_OPENAI_DEPLOYMENT"

        try:
            # Prepare summary for AI
            total = len(results)
            successful = sum(1 for r in results if r.get('success'))
            failed = total - successful

            operations = test_context.get('operations', [])
            environment = test_context.get('environment', 'Unknown')

            prompt = f"""
            Analyze the following product lifecycle test execution results and provide insights:
            
            Test Context:
            - Environment: {environment}
            - Operations: {', '.join([f"{op['origin']} {op['type']}" for op in operations])}
            - Total Products: {total}
            - Successful: {successful}
            - Failed: {failed}
            
            Results Summary:
            {json.dumps(results[:10], indent=2, default=str)}  # First 10 results
            
            Please provide:
            1. Overall analysis of the test execution
            2. Key patterns or issues identified
            3. Recommendations for improvement
            4. Potential risks or concerns
            5. Best practices for similar tests
            
            Format as markdown with clear sections.
            """

            response = self.ai_client.chat_completion_create(
                messages=[
                    {"role": "system", "content": "You are an expert in product lifecycle management and database operations testing. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # Extract content from response
            if response and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content']
            else:
                return "No AI insights generated"

        except Exception as e:
            logger.error(f"Failed to get AI insights: {e}")
            return f"Failed to generate AI insights: {str(e)}"


def show_ui():
    """Main UI function for Newfold Migration Toolkit"""

    st.markdown("## 🚀 Newfold Migration Toolkit")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">CSRT & RAS Operations Platform</h3>
        <p style="color: white; margin: 5px 0 0 0; opacity: 0.9;">
            Comprehensive product lifecycle testing and migration toolkit powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize toolkit
    if 'newfold_toolkit' not in st.session_state:
        st.session_state.newfold_toolkit = NewfoldMigrationToolkit()

    # Initialize default environment on first load
    if 'nf_environment' not in st.session_state:
        st.session_state.nf_environment = None
        st.session_state.nf_connection = None

    toolkit = st.session_state.newfold_toolkit

    # Check dependencies and show initial warnings
    if not ORACLEDB_AVAILABLE:
        st.error("❌ Oracle database driver not available. Install: `pip install oracledb`")
        return

    if not DB_CONFIG_AVAILABLE:
        st.warning("⚠️ Database configuration not found. Check database_config_variables.py")

    st.markdown("---")

    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Test Execution",
        "🔍 Product Discovery",
        "📊 Results & History",
        "🤖 AI Insights",
        "⚙️ Configuration"
    ])

    # Tab 1: Test Execution
    with tab1:
        # Show System Status at the top of Test Execution tab
        st.markdown("### 📋 System Status")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = "✅" if ORACLEDB_AVAILABLE else "❌"
            st.markdown(f"**Oracle Driver:** {status}")

        with col2:
            status = "✅" if DB_CONFIG_AVAILABLE else "❌"
            st.markdown(f"**DB Config:** {status}")

        with col3:
            status = "✅" if AZURE_OPENAI_AVAILABLE else "❌"
            st.markdown(f"**Azure OpenAI:** {status}")

        with col4:
            if st.session_state.get('nf_connection'):
                env_name = st.session_state.get('nf_environment', 'Unknown')
                conn_status = f"✅ {env_name}"
            else:
                conn_status = "⭕ Select Environment"
            st.markdown(f"**Database:** {conn_status}")

        st.markdown("---")

        st.markdown("### 🎯 Execute Product Lifecycle Tests")

        col1, col2 = st.columns(2)

        with col1:
            environment = st.selectbox(
                "Database Environment",
                ["QAMain", "Stage", "JarvisQA1", "JarvisQA2", "Production"],
                help="Select the database environment to connect to",
                key="nf_env_selector"
            )

            # Auto-connect when environment changes OR when no connection exists for current environment
            current_env = st.session_state.get('nf_environment')
            should_connect = (current_env != environment) or (environment and not st.session_state.get('nf_connection'))

            if should_connect and environment:
                # Environment changed or first time with this environment, auto-connect
                with st.spinner(f"🔌 Connecting to {environment}..."):
                    success, message, connection = toolkit.connect_to_database(environment)
                    if success:
                        st.session_state.nf_connection = connection
                        st.session_state.nf_environment = environment
                        st.success(f"✅ {message}")
                    else:
                        st.session_state.nf_connection = None
                        st.session_state.nf_environment = None
                        st.error(f"❌ {message}")

            test_name = st.text_input(
                "Test Name",
                value=f"Test_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Unique identifier for this test execution"
            )

            brand = st.selectbox(
                "Brand",
                ["Bluehost", "HostGator", "Domain.com", "iPage", "NetWeb"],
                help="Select the brand for PARENT_CHANNEL_ID mapping"
            )

        with col2:
            st.markdown("#### Operations to Execute")

            # RAS Operations
            st.markdown("**RAS Operations:**")
            ras_ops = []
            if st.checkbox("RAS Deactivate", key="ras_deactivate"):
                ras_ops.append({"type": "DEACTIVATE", "origin": "RAS"})
            if st.checkbox("RAS Delete", key="ras_delete"):
                ras_ops.append({"type": "DELETE", "origin": "RAS"})
            if st.checkbox("RAS Renew", key="ras_renew"):
                ras_ops.append({"type": "RENEW", "origin": "RAS"})

            # CSRT Operations
            st.markdown("**CSRT Operations:**")
            csrt_ops = []
            if st.checkbox("CSRT Deactivate", key="csrt_deactivate"):
                csrt_ops.append({"type": "DEACTIVATE", "origin": "CSRT"})
            if st.checkbox("CSRT Reactivate", key="csrt_reactivate"):
                csrt_ops.append({"type": "REACTIVATE", "origin": "CSRT"})
            if st.checkbox("CSRT Delete", key="csrt_delete"):
                csrt_ops.append({"type": "DELETE", "origin": "CSRT"})

        # SKU Input
        st.markdown("### 📦 Product SKUs")
        sku_input = st.text_area(
            "Enter SKUs (one per line, comma-separated, or space-separated)",
            height=150,
            placeholder="ECOMDASH_PREMIUM\nSSL_WILDCARD\nBH_BASIC, BH_PRO, BH_CHOICE",
            help="Enter product SKU codes. Supports multiple formats."
        )

        # Migration Source
        migration_source_input = st.text_input(
            "Migration Sources (optional)",
            placeholder="UberSmith, NCOM, etc.",
            help="Filter products by migration source"
        )

        # Execute and Reset buttons
        col1, col2 = st.columns([5, 1])

        with col1:
            execute_disabled = not st.session_state.get('nf_connection') or not sku_input or (not ras_ops and not csrt_ops)

            if not st.session_state.get('nf_connection'):
                st.warning("⚠️ No database connection. Please select an environment above to auto-connect.")

            if st.button("▶️ Execute Test", disabled=execute_disabled, use_container_width=True):
                # Parse inputs
                skus = toolkit.parse_skus(sku_input)
                migration_sources = toolkit.parse_migration_sources(migration_source_input)
                all_operations = ras_ops + csrt_ops

                if not skus:
                    st.error("No SKUs provided")
                elif not all_operations:
                    st.error("No operations selected")
                else:
                    st.markdown("### 🔄 Execution Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Find Products
                    status_text.text("🔍 Step 1/3: Finding products...")
                    progress_bar.progress(10)

                    connection = st.session_state.nf_connection
                    product_results = toolkit.find_products(
                        connection, skus, all_operations, migration_sources, brand.lower()
                    )

                    progress_bar.progress(40)

                    # Step 2: Execute RAS Operations
                    all_ras_results = []
                    if ras_ops:
                        status_text.text("💾 Step 2a/4: Executing RAS operations...")

                        # Group products by operation type
                        ras_products_by_op = defaultdict(list)
                        for sku, products in product_results.items():
                            for prod in products:
                                if prod['operation_origin'] == 'RAS' and prod['success']:
                                    ras_products_by_op[prod['operation_type']].append(prod)

                        # Execute each RAS operation type
                        for op_type, products in ras_products_by_op.items():
                            results = toolkit.execute_ras_operations(connection, products, op_type)
                            all_ras_results.extend(results)

                        st.session_state.nf_execution_results = all_ras_results

                    progress_bar.progress(55)

                    # Step 3: Execute CSRT Operations
                    all_csrt_results = []
                    if csrt_ops:
                        status_text.text("💾 Step 2b/4: Executing CSRT operations...")

                        # Group products by operation type
                        csrt_products_by_op = defaultdict(list)
                        for sku, products in product_results.items():
                            for prod in products:
                                if prod['operation_origin'] == 'CSRT' and prod['success']:
                                    csrt_products_by_op[prod['operation_type']].append(prod)

                        # Execute each CSRT operation type
                        for op_type, products in csrt_products_by_op.items():
                            results = toolkit.execute_csrt_operations(environment, products, op_type)
                            all_csrt_results.extend(results)

                        st.session_state.nf_csrt_results = all_csrt_results

                    progress_bar.progress(70)

                    # Step 4: Save Results
                    status_text.text("💾 Step 3/4: Saving results...")

                    all_results = []
                    for sku, products in product_results.items():
                        all_results.extend(products)

                    filename = toolkit.save_test_results(environment, test_name, brand, all_results)

                    progress_bar.progress(100)
                    status_text.text("✅ Execution completed!")

                    # Display summary
                    st.success(f"Test execution completed! Results saved to: {filename}")

                    total = len(all_results)
                    successful = sum(1 for r in all_results if r.get('success'))
                    failed = total - successful

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Products", total)
                    col2.metric("Successful", successful, delta=f"{(successful/total*100):.1f}%")
                    col3.metric("Failed", failed, delta=f"-{(failed/total*100):.1f}%" if failed > 0 else "0%")

                    # Store for other tabs
                    st.session_state.nf_last_results = all_results
                    st.session_state.nf_test_context = {
                        'environment': environment,
                        'test_name': test_name,
                        'brand': brand,
                        'operations': all_operations
                    }

        with col2:
            if st.button("🔄", help="Reset Connection", use_container_width=True):
                st.session_state.nf_connection = None
                st.session_state.nf_environment = None
                st.rerun()

    # Tab 2: Product Discovery
    with tab2:
        st.markdown("### 🔍 Discover Eligible Products")
        st.info("Find products that match specific criteria without executing operations")

        if not st.session_state.get('nf_connection'):
            st.warning("⚠️ Connect to database first in the Test Execution tab")
        else:
            col1, col2 = st.columns(2)

            with col1:
                disc_skus = st.text_area(
                    "SKUs to search",
                    height=100,
                    placeholder="BH_COMMERCE_SOLUTION_HOSTING\nSSL_WILDCARD\nBH_BASIC",
                    help="Enter product SKU codes (one per line, comma-separated, or space-separated)"
                )

                disc_brand = st.selectbox(
                    "Brand",
                    ["Bluehost", "HostGator", "Domain.com", "iPage", "NetWeb"],
                    key="disc_brand",
                    help="Select brand to search within"
                )

                disc_migration_source = st.text_input(
                    "Migration Source (optional)",
                    placeholder="UberSmith, NCOM, etc.",
                    key="disc_mig_source",
                    help="Filter by migration source"
                )

            with col2:
                st.markdown("**Filter by Lifecycle Status:**")

                lifecycle_filter = st.radio(
                    "Product Status",
                    ["All Products", "Active Only (LIFECYCLE_CD_ID=10)", "Inactive Only (LIFECYCLE_CD_ID=11)", "Active or Inactive"],
                    help="Choose which lifecycle status to search for"
                )

                st.markdown("**Filter by Renewal Status:**")
                renewal_filter = st.radio(
                    "Auto-Renewal",
                    ["Any", "Auto-Renew ON (Y)", "Auto-Renew OFF (N)"],
                    help="Filter by auto-renewal flag"
                )

                # Wallet filter for renewable products
                st.markdown("**Additional Filters:**")
                check_wallet = st.checkbox(
                    "Only products with valid wallet",
                    help="Include wallet validation (for RENEW operations)"
                )

                # Debug mode
                debug_mode = st.checkbox(
                    "🔧 Debug Mode (Show SQL queries)",
                    help="Display SQL queries and parameters for troubleshooting"
                )

            if st.button("🔎 Search Products", use_container_width=True):
                if disc_skus:
                    skus = toolkit.parse_skus(disc_skus)
                    migration_sources = toolkit.parse_migration_sources(disc_migration_source) if disc_migration_source else []

                    # Build operation list based on filters
                    # We use the operation type to leverage existing query logic
                    disc_ops = []

                    if lifecycle_filter == "Active Only (LIFECYCLE_CD_ID=10)":
                        disc_ops.append({"type": "DEACTIVATE", "origin": "RAS"})  # Searches for LIFECYCLE=10
                    elif lifecycle_filter == "Inactive Only (LIFECYCLE_CD_ID=11)":
                        disc_ops.append({"type": "REACTIVATE", "origin": "CSRT"})  # Searches for LIFECYCLE=11
                    elif lifecycle_filter == "Active or Inactive":
                        disc_ops.append({"type": "DELETE", "origin": "RAS"})  # Searches for LIFECYCLE=10 or 11
                    else:  # All Products
                        # Search with multiple operation types to get all products
                        disc_ops.append({"type": "DEACTIVATE", "origin": "RAS"})
                        disc_ops.append({"type": "REACTIVATE", "origin": "CSRT"})

                    # Add wallet check if selected
                    if check_wallet:
                        disc_ops = [{"type": "RENEW", "origin": "RAS"}]  # RENEW includes wallet validation

                    with st.spinner("Searching database..."):
                        results = toolkit.find_products(
                            st.session_state.nf_connection,
                            skus,
                            disc_ops,
                            migration_sources,
                            disc_brand.lower(),
                            debug_mode=debug_mode
                        )

                    # Extract and display debug info if available
                    debug_info = results.pop('__debug_info__', None)

                    if debug_mode and debug_info:
                        with st.expander("🔧 Debug Information - SQL Queries"):
                            for idx, info in enumerate(debug_info, 1):
                                st.markdown(f"**Query {idx}: {info['sku']} - {info['operation']}**")
                                st.code(info['sql'], language='sql')
                                st.json(info['params'])
                                st.markdown("---")

                    # Display results
                    st.markdown("### 📊 Discovery Results")

                    # Summary
                    total_products = sum(len([p for p in prods if p.get('success')]) for prods in results.values())
                    total_skus = len(skus)
                    found_skus = len([sku for sku, prods in results.items() if any(p.get('success') for p in prods)])

                    col1, col2, col3 = st.columns(3)
                    col1.metric("SKUs Searched", total_skus)
                    col2.metric("SKUs Found", found_skus)
                    col3.metric("Total Products", total_products)

                    st.markdown("---")

                    for sku, products in results.items():
                        st.markdown(f"#### SKU: `{sku}`")

                        # Filter successful results
                        successful_products = [p for p in products if p.get('success')]

                        if successful_products:
                            # Apply additional filters
                            filtered_products = successful_products

                            # Filter by renewal status
                            if renewal_filter == "Auto-Renew ON (Y)":
                                filtered_products = [p for p in filtered_products if p.get('auto_renew_flag') == 'Y']
                            elif renewal_filter == "Auto-Renew OFF (N)":
                                filtered_products = [p for p in filtered_products if p.get('auto_renew_flag') == 'N']

                            if filtered_products:
                                df = pd.DataFrame(filtered_products)

                                # Select relevant columns to display
                                display_columns = ['prod_inst_id', 'sku', 'lifecycle_cd_id', 'auto_renew_flag', 'exp_date', 'mig_source']
                                available_columns = [col for col in display_columns if col in df.columns]

                                if available_columns:
                                    st.dataframe(df[available_columns], use_container_width=True)
                                else:
                                    st.dataframe(df, use_container_width=True)

                                st.success(f"✅ Found {len(filtered_products)} product(s)")
                            else:
                                st.info(f"ℹ️ No products matching the selected filters for {sku}")
                        else:
                            # Show detailed error message
                            error_messages = [p.get('error', 'Unknown error') for p in products if not p.get('success')]
                            if error_messages:
                                st.warning(f"⚠️ {error_messages[0]}")
                            else:
                                st.warning(f"⚠️ No products found for {sku}")

                            # Show troubleshooting tips
                            with st.expander("🔍 Troubleshooting Tips"):
                                st.markdown(f"""
                                **Why might this happen?**
                                
                                1. **SKU doesn't exist in database**
                                   - Verify the SKU name is correct: `{sku}`
                                   - Check if SKU exists for selected brand: `{disc_brand}`
                                   
                                2. **No products match the filters**
                                   - Current lifecycle filter: `{lifecycle_filter}`
                                   - Try selecting "All Products" to see if any products exist
                                   
                                3. **Brand mismatch**
                                   - Selected brand: `{disc_brand}` (PARENT_CHANNEL_ID={toolkit.get_parent_channel_id_for_brand(disc_brand.lower())})
                                   - Verify this SKU belongs to this brand
                                   
                                4. **Migration source filter**
                                   - {'Migration source filter active: ' + disc_migration_source if disc_migration_source else 'No migration source filter'}
                                   - Products may have different migration sources
                                
                                **What to try:**
                                - Select "All Products" to search without lifecycle restrictions
                                - Remove migration source filter
                                - Try a different brand
                                - Verify SKU exists in the database using a SQL client
                                """)
                else:
                    st.warning("⚠️ Please enter at least one SKU to search")

    # Tab 3: Results & History
    with tab3:
        st.markdown("### 📊 Test Results & History")

        if st.session_state.get('nf_last_results'):
            results = st.session_state.nf_last_results

            # Summary metrics
            st.markdown("#### Summary")
            total = len(results)
            successful = sum(1 for r in results if r.get('success'))
            failed = total - successful

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", total)
            col2.metric("Successful", successful)
            col3.metric("Failed", failed)
            col4.metric("Success Rate", f"{(successful/total*100):.1f}%")

            # Results table
            st.markdown("#### Detailed Results")
            df = pd.DataFrame(results)

            # Color code based on success
            def highlight_status(row):
                if row['success']:
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)

            styled_df = df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"newfold_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No test results available. Execute a test in the Test Execution tab.")

    # Tab 4: AI Insights
    with tab4:
        st.markdown("### 🤖 AI-Powered Insights & Recommendations")

        if not AZURE_OPENAI_AVAILABLE:
            st.warning("⚠️ Azure OpenAI not available. AI insights are disabled.")
        elif not st.session_state.get('nf_last_results'):
            st.info("No test results available. Execute a test first to get AI insights.")
        else:
            if st.button("🧠 Generate AI Insights", use_container_width=True):
                with st.spinner("Analyzing results with AI..."):
                    insights = toolkit.get_ai_insights(
                        st.session_state.nf_last_results,
                        st.session_state.get('nf_test_context', {})
                    )

                st.markdown("#### 🎯 AI Analysis")
                st.markdown(insights)

                # Store insights
                st.session_state.nf_ai_insights = insights

    # Tab 5: Configuration
    with tab5:
        st.markdown("### ⚙️ Configuration & Settings")

        st.markdown("#### 🔌 Auto-Connection Feature")
        st.info("""
        **Auto-Connection Enabled!** The database automatically connects when you select an environment.
        
        - Select from: **QAMain**, **Stage**, **JarvisQA1**, **JarvisQA2**, or **Production**
        - Connection happens automatically in the background
        - Real-time connection status displayed
        - No manual "Connect" button needed
        - Use 🔄 Reset to reconnect if needed
        """)

        st.markdown("#### Database Environments")

        st.info("📝 All configuration values are dynamically fetched from `database_config_variables.py`")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**QAMain Environment:**")
            if DB_CONFIG_AVAILABLE:
                qamain_host = getattr(db_config, 'qamain_db_host_variable', 'Not configured')
                qamain_port = getattr(db_config, 'qamain_db_port_variable', 'Not configured')
                qamain_service = getattr(db_config, 'qamain_db_service_variable', 'Not configured')
                qamain_user = getattr(db_config, 'default_db_user_variable', 'Not configured')
                st.code(f"""
Host: {qamain_host}
Port: {qamain_port}
Service: {qamain_service}
User: {qamain_user}
Source: database_config_variables.py
                """)
            else:
                st.code("Configuration file not available")

            st.markdown("**Stage Environment:**")
            if DB_CONFIG_AVAILABLE:
                stage_host = getattr(db_config, 'stage_db_host_variable', 'Not configured')
                stage_port = getattr(db_config, 'stage_db_port_variable', 'Not configured')
                stage_service = getattr(db_config, 'stage_db_service_variable', 'Not configured')
                stage_user = getattr(db_config, 'default_db_user_variable', 'Not configured')
                st.code(f"""
Host: {stage_host}
Port: {stage_port}
Service: {stage_service}
User: {stage_user}
Source: database_config_variables.py
                """)
            else:
                st.code("Configuration file not available")

            st.markdown("**JarvisQA1 Environment:**")
            if DB_CONFIG_AVAILABLE:
                jarvisqa1_host = getattr(db_config, 'jarvisqa1_db_host_variable', 'Not configured')
                jarvisqa1_port = getattr(db_config, 'jarvisqa1_db_port_variable', 'Not configured')
                jarvisqa1_service = getattr(db_config, 'jarvisqa1_db_service_variable', 'Not configured')
                jarvisqa1_user = getattr(db_config, 'default_db_user_variable', 'Not configured')
                st.code(f"""
Host: {jarvisqa1_host}
Port: {jarvisqa1_port}
Service: {jarvisqa1_service}
User: {jarvisqa1_user}
Source: database_config_variables.py
                """)
            else:
                st.code("Configuration file not available")

        with col2:
            st.markdown("**JarvisQA2 Environment:**")
            if DB_CONFIG_AVAILABLE:
                jarvisqa2_host = getattr(db_config, 'jarvisqa2_db_host_variable', 'Not configured')
                jarvisqa2_port = getattr(db_config, 'jarvisqa2_db_port_variable', 'Not configured')
                jarvisqa2_service = getattr(db_config, 'jarvisqa2_db_service_variable', 'Not configured')
                jarvisqa2_user = getattr(db_config, 'default_db_user_variable', 'Not configured')
                st.code(f"""
Host: {jarvisqa2_host}
Port: {jarvisqa2_port}
Service: {jarvisqa2_service}
User: {jarvisqa2_user}
Source: database_config_variables.py
                """)
            else:
                st.code("Configuration file not available")

            st.markdown("**Production Environment:**")
            st.code("""
Status: ⚠️ Restricted
Note: Production connections require 
      special approval for safety
Access: Contact administrator
            """)

        st.markdown("#### Operation Rules")

        st.markdown("""
        **RAS Operations:**
        - **Deactivate**: `AUTO_RENEW_FLAG='N'`, `EXP_DATE=TRUNC(SYSDATE-21)`, requires `LIFECYCLE_CD_ID=10`
        - **Delete**: `AUTO_RENEW_FLAG='N'`, `EXP_DATE=TRUNC(SYSDATE-42)`, `LIFECYCLE_CD_ID=11`
        - **Renew**: `AUTO_RENEW_FLAG='Y'`, `EXP_DATE=TRUNC(SYSDATE)`, requires `LIFECYCLE_CD_ID=10` + valid wallet
        
        **CSRT Operations:**
        - **Deactivate**: Calls CSRT API for product deactivation
        - **Reactivate**: Calls CSRT API for product reactivation, requires `LIFECYCLE_CD_ID=11`
        - **Delete**: Calls CSRT API for product deletion
        """)

        st.markdown("#### Brand Mapping")

        brand_mapping_df = pd.DataFrame([
            {"Brand": "Bluehost", "PARENT_CHANNEL_ID": "8"},
            {"Brand": "HostGator", "PARENT_CHANNEL_ID": "9"},
            {"Brand": "Domain.com", "PARENT_CHANNEL_ID": "18"},
            {"Brand": "iPage", "PARENT_CHANNEL_ID": "17"},
            {"Brand": "NetWeb", "PARENT_CHANNEL_ID": "1"},
        ])

        st.dataframe(brand_mapping_df, use_container_width=True)

        st.markdown("#### Documentation")

        with st.expander("📚 View Complete Documentation"):
            st.markdown("""
            ## Newfold Migration Toolkit Documentation
            
            ### Overview
            The Newfold Migration Toolkit is a comprehensive solution for product lifecycle management and migration testing.
            
            ### Key Features
            1. **Multi-Environment Support**: Connect to QAMain, Stage, JarvisQA1, JarvisQA2, or Production
            2. **Auto-Connection**: Automatic database connection when environment is selected
            2. **Bulk Processing**: Process multiple SKUs efficiently
            3. **RAS Operations**: Direct database updates for product lifecycle
            4. **CSRT Integration**: API-based product operations
            5. **AI Insights**: Azure OpenAI-powered analysis and recommendations
            6. **Result Tracking**: Persistent storage of test results
            
            ### Workflow
            1. Connect to target environment (STG/QAMain)
            2. Select operations (RAS and/or CSRT)
            3. Enter product SKUs
            4. Execute test
            5. Review results and AI insights
            
            ### Best Practices
            - Always test in STG before QAMain
            - Use meaningful test names for tracking
            - Review AI insights for optimization opportunities
            - Download results for record-keeping
            - Check product discovery before bulk operations
            
            ### Troubleshooting
            - **Connection Failures**: Verify network access and credentials
            - **No Products Found**: Check SKU codes and search criteria
            - **Operation Failures**: Review lifecycle conditions and wallet requirements
            - **AI Insights Unavailable**: Ensure Azure OpenAI is configured
            """)


# Entry point for the module
if __name__ == "__main__":
    show_ui()

