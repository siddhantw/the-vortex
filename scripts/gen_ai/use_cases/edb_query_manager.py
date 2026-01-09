"""
EDB Query Manager Module
Comprehensive EDB (Enterprise Database) account query and management with real database connections
and AI-powered insights
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import os

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
    st.warning("⚠️ Database configuration not available")

# Configure logging with enhanced features
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("EDBQueryManager", level=logging.INFO, log_file="edb_query_manager.log")
    ENHANCED_LOGGING = True
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ENHANCED_LOGGING = False


try:
    import oracledb
    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False

# Import AzureOpenAIClient like TestPilot does
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
        logger.warning("Azure OpenAI client not available. Install dependencies or check import paths.")


class EDBQueryManager:
    """Advanced EDB account query and management with real database connections"""

    # Class-level flag to track Oracle client initialization
    _oracle_client_initialized = False
    _use_thin_mode = False  # Flag to track if we should use thin mode

    # Import mappings from database_config_variables
    # Single source of truth: tests/configs/database_config_variables.py
    # All database-related constants are centralized there for consistency
    BRAND_MAPPING = db_config.BRAND_MAPPING
    PARENT_CHANNEL_TO_BRAND = db_config.PARENT_CHANNEL_TO_BRAND
    LIFECYCLE_MAPPING = db_config.LIFECYCLE_MAPPING
    BUILDER_TYPE_MAPPING = db_config.BUILDER_TYPE_MAPPING
    WEBBUILDER_PLATFORM_TO_BUILDER_TYPE = db_config.WEBBUILDER_PLATFORM_TO_BUILDER_TYPE
    EMAIL_PLATFORM_MAPPING = db_config.EMAIL_PLATFORM_MAPPING

    def __init__(self):
        self.connection = None
        self.query_history = []
        self.results_cache = {}
        self.current_environment = None
        self.azure_client = None  # Initialize before calling _initialize_azure_openai
        try:
            self._initialize_azure_openai()
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI in __init__: {e}")
            self.azure_client = None

        # Initialize Oracle client once at class instantiation if needed
        self._initialize_oracle_client()

    def _initialize_oracle_client(self):
        """Initialize Oracle client once per application lifecycle"""
        if not ORACLEDB_AVAILABLE or not DB_CONFIG_AVAILABLE:
            return

        # Skip if already initialized
        if EDBQueryManager._oracle_client_initialized:
            return

        try:
            # Check if we should use thick mode
            if hasattr(db_config, 'db_driver_mode_variable') and db_config.db_driver_mode_variable == 'thick':
                try:
                    oracledb.init_oracle_client()
                    EDBQueryManager._oracle_client_initialized = True
                    logger.info("Oracle client initialized in thick mode")
                except Exception as e:
                    # If thick mode fails, fall back to thin mode
                    logger.warning(f"Failed to initialize thick mode, using thin mode: {e}")
                    EDBQueryManager._use_thin_mode = True
                    EDBQueryManager._oracle_client_initialized = True
            else:
                # Use thin mode by default
                EDBQueryManager._use_thin_mode = True
                EDBQueryManager._oracle_client_initialized = True
                logger.info("Using Oracle thin mode")
        except Exception as e:
            logger.error(f"Error during Oracle client initialization: {e}")
            EDBQueryManager._use_thin_mode = True
            EDBQueryManager._oracle_client_initialized = True

    def get_db_connection(self, environment: str) -> Tuple[bool, str, Any]:
        """Establish connection to the specified database environment"""
        if not ORACLEDB_AVAILABLE:
            return False, "Oracle DB driver not installed", None

        if not DB_CONFIG_AVAILABLE:
            return False, "Database configuration not available", None

        try:
            # Get connection parameters based on environment
            if environment == "QAMain":
                host = db_config.qamain_db_host_variable
                port = db_config.qamain_db_port_variable
                service = db_config.qamain_db_service_variable
                user = db_config.qamain_db_user_variable
                password = db_config.qamain_db_pass_variable
            elif environment == "Stage":
                host = db_config.stage_db_host_variable
                port = db_config.stage_db_port_variable
                service = db_config.stage_db_service_variable
                user = db_config.stage_db_user_variable
                password = db_config.stage_db_pass_variable
            elif environment == "JarvisQA1":
                host = db_config.jarvisqa1_db_host_variable
                port = db_config.jarvisqa1_db_port_variable
                service = db_config.jarvisqa1_db_service_variable
                user = db_config.jarvisqa1_db_user_variable
                password = db_config.jarvisqa1_db_pass_variable
            elif environment == "JarvisQA2":
                host = db_config.jarvisqa2_db_host_variable
                port = db_config.jarvisqa2_db_port_variable
                service = db_config.jarvisqa2_db_service_variable
                user = db_config.jarvisqa2_db_user_variable
                password = db_config.jarvisqa2_db_pass_variable
            elif environment == "Production":
                # Production uses QAMain config as base (update if different config exists)
                host = db_config.qamain_db_host_variable
                port = db_config.qamain_db_port_variable
                service = db_config.qamain_db_service_variable
                user = db_config.qamain_db_user_variable
                password = db_config.qamain_db_pass_variable
            else:
                return False, f"Unknown environment: {environment}", None

            # Create DSN
            dsn = oracledb.makedsn(host, port, service_name=service)

            # Establish connection (timeout parameter not supported in all versions)
            connection = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn
            )
            self.current_environment = environment

            logger.info(f"Successfully connected to {environment} database")
            return True, f"Connected to {environment}", connection

        except Exception as e:
            logger.error(f"Failed to connect to {environment}: {e}")
            return False, f"Connection failed: {str(e)}", None

    def verify_table_access(self, connection, table_name: str) -> Tuple[bool, str]:
        """
        Verify if a table exists and is accessible.

        Args:
            connection: Active database connection
            table_name: Base table name to check

        Returns:
            Tuple of (success, message)
        """
        try:
            cursor = connection.cursor()

            # Try to query just one row to verify access - no schema prefix
            query = f"SELECT * FROM {table_name} WHERE ROWNUM <= 1"
            cursor.execute(query)
            cursor.fetchall()
            cursor.close()

            return True, f"Table {table_name} is accessible"
        except Exception as e:
            error_msg = str(e)

            # Check for specific error codes
            if "ORA-00942" in error_msg:
                return False, f"Table '{table_name}' does not exist or is not accessible"
            elif "ORA-00980" in error_msg:
                return False, f"Synonym '{table_name}' translation no longer valid"
            elif "ORA-01031" in error_msg:
                return False, f"Insufficient privileges to access '{table_name}'"
            else:
                return False, f"Error accessing table '{table_name}': {error_msg}"

    def detect_product_table(self, connection) -> Optional[str]:
        """
        Dynamically detect which product table exists in the database.
        Different environments may have different table names.

        Args:
            connection: Active database connection

        Returns:
            Table name if found, None otherwise
        """
        # List of possible product table names to try (with schema prefixes!)
        possible_tables = [
            'EDB.PROD_CODE_BASE',  # Most common - in EDB schema
            'prod_code_base',      # Without schema (if current user has access)
            'EDB.PRODUCT',         # Alternative in EDB schema
            'product',             # Without schema
            'PROD_CATALOG',        # Another possible name
            'prod_catalog'         # Without schema
        ]

        for table_name in possible_tables:
            accessible, message = self.verify_table_access(connection, table_name)
            if accessible:
                logger.info(f"Detected product table: {table_name}")
                return table_name

        logger.error("Could not find any accessible product table")
        return None

    def get_available_tables(self, environment: str) -> List[str]:
        """
        Get list of tables accessible to the current user.
        Useful for diagnostics when tables are not found.

        Args:
            environment: Database environment

        Returns:
            List of accessible table names
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Cannot get tables: {message}")
            return []

        try:
            cursor = connection.cursor()

            # Query user's accessible tables - no schema prefix needed
            query = """
                SELECT table_name 
                FROM user_tables 
                ORDER BY table_name
            """

            cursor.execute(query)
            tables = [row[0] for row in cursor.fetchall()]

            cursor.close()
            connection.close()

            logger.info(f"Found {len(tables)} accessible tables in {environment}")
            logger.info(f"Tables: {tables}")  # Log all tables for debugging

            return tables

        except Exception as e:
            logger.error(f"Error fetching table list: {e}")
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def diagnose_database_schema(self, environment: str) -> Dict[str, Any]:
        """
        Comprehensive database schema diagnosis for debugging.
        Returns detailed information about tables and potential product-related tables.

        Args:
            environment: Database environment

        Returns:
            Dictionary with diagnostic information
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            return {"success": False, "error": message}

        try:
            cursor = connection.cursor()
            diagnosis = {
                "success": True,
                "environment": environment,
                "all_tables": [],
                "product_related_tables": [],
                "prod_related_tables": [],
                "table_details": {}
            }

            # Get all tables
            cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
            all_tables = [row[0] for row in cursor.fetchall()]
            diagnosis["all_tables"] = all_tables

            # Find product-related tables
            for table in all_tables:
                if 'PRODUCT' in table.upper() or 'PROD' in table.upper():
                    diagnosis["product_related_tables"].append(table)

                    # Get columns for this table
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM user_tab_columns 
                        WHERE table_name = :table_name
                        ORDER BY column_id
                    """, table_name=table)

                    columns = [(row[0], row[1]) for row in cursor.fetchall()]
                    diagnosis["table_details"][table] = {
                        "columns": columns,
                        "column_names": [col[0] for col in columns]
                    }

            cursor.close()
            connection.close()

            logger.info(f"Database diagnosis complete for {environment}")
            logger.info(f"Total tables: {len(all_tables)}")
            logger.info(f"Product-related tables: {diagnosis['product_related_tables']}")

            return diagnosis

        except Exception as e:
            logger.error(f"Error during database diagnosis: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return {"success": False, "error": str(e)}

    def get_table_details(self, environment: str) -> List[Dict[str, Any]]:
        """
        Get detailed information about all accessible tables including row counts.

        Args:
            environment: Database environment

        Returns:
            List of dictionaries with table details
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Cannot get table details: {message}")
            return []

        try:
            cursor = connection.cursor()

            # Query table metadata
            query = """
                SELECT 
                    table_name,
                    num_rows,
                    blocks,
                    avg_row_len,
                    last_analyzed
                FROM user_tables 
                ORDER BY table_name
            """

            cursor.execute(query)

            table_details = []
            for row in cursor.fetchall():
                table_details.append({
                    'table_name': row[0],
                    'num_rows': row[1] if row[1] else 0,
                    'blocks': row[2] if row[2] else 0,
                    'avg_row_len': row[3] if row[3] else 0,
                    'last_analyzed': row[4].isoformat() if row[4] else None
                })

            cursor.close()
            connection.close()

            logger.info(f"Found {len(table_details)} table details in {environment}")
            return table_details

        except Exception as e:
            logger.error(f"Error fetching table details: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_table_columns(self, environment: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a specific table.

        Args:
            environment: Database environment
            table_name: Name of the table

        Returns:
            List of dictionaries with column details
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Cannot get table columns: {message}")
            return []

        try:
            cursor = connection.cursor()

            # Query column metadata
            query = """
                SELECT 
                    column_name,
                    data_type,
                    data_length,
                    nullable,
                    column_id
                FROM user_tab_columns 
                WHERE table_name = :table_name
                ORDER BY column_id
            """

            cursor.execute(query, table_name=table_name.upper())

            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'column_name': row[0],
                    'data_type': row[1],
                    'data_length': row[2],
                    'nullable': row[3],
                    'column_id': row[4]
                })

            cursor.close()
            connection.close()

            return columns

        except Exception as e:
            logger.error(f"Error fetching columns for {table_name}: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_table_relationships(self, environment: str) -> List[Dict[str, Any]]:
        """
        Get foreign key relationships between tables.

        Args:
            environment: Database environment

        Returns:
            List of dictionaries with relationship details
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Cannot get table relationships: {message}")
            return []

        try:
            cursor = connection.cursor()

            # Query foreign key constraints - simplified to avoid complex joins
            query = """
                SELECT 
                    uc.constraint_name,
                    uc.table_name as child_table,
                    ucc.column_name as child_column,
                    uc_pk.table_name as parent_table,
                    ucc_pk.column_name as parent_column
                FROM user_constraints uc
                JOIN user_cons_columns ucc ON uc.constraint_name = ucc.constraint_name
                JOIN user_constraints uc_pk ON uc.r_constraint_name = uc_pk.constraint_name
                JOIN user_cons_columns ucc_pk ON uc_pk.constraint_name = ucc_pk.constraint_name
                WHERE uc.constraint_type = 'R'
                    AND ucc.position = ucc_pk.position
                ORDER BY uc.table_name, uc.constraint_name
            """

            cursor.execute(query)

            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    'constraint_name': row[0],
                    'child_table': row[1],
                    'child_column': row[2],
                    'parent_table': row[3],
                    'parent_column': row[4]
                })

            cursor.close()
            connection.close()

            logger.info(f"Found {len(relationships)} relationships in {environment}")
            return relationships

        except Exception as e:
            logger.error(f"Error fetching table relationships: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_product_codes(self, environment: str, search_term: str = "") -> List[str]:
        """Fetch available product codes from the database"""
        logger.info(f"Fetching product codes from {environment} with search term: '{search_term}'")

        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Failed to get product codes: {message}")
            return []

        try:
            cursor = connection.cursor()

            # Use exact same query as database.py - it uses prod_cd column from prod_code_base table
            # Try with EDB schema prefix first, then without
            query_tried = []
            results = []

            # Try 1: With EDB schema prefix
            try:
                if search_term:
                    query = """
                        SELECT DISTINCT prod_cd
                        FROM EDB.prod_code_base
                        WHERE UPPER(prod_cd) LIKE UPPER(:search_term)
                        ORDER BY prod_cd
                    """
                    cursor.execute(query, {'search_term': f'%{search_term}%'})
                else:
                    query = """
                        SELECT DISTINCT prod_cd
                        FROM EDB.prod_code_base
                        ORDER BY prod_cd
                    """
                    cursor.execute(query)

                results = [row[0] for row in cursor.fetchall() if row[0]]
                logger.info(f"Successfully queried EDB.prod_code_base")

            except Exception as e1:
                query_tried.append(f"EDB.prod_code_base: {str(e1)}")

                # Try 2: Without schema prefix (if synonyms exist)
                try:
                    if search_term:
                        query = """
                            SELECT DISTINCT prod_cd
                            FROM prod_code_base
                            WHERE UPPER(prod_cd) LIKE UPPER(:search_term)
                            ORDER BY prod_cd
                        """
                        cursor.execute(query, {'search_term': f'%{search_term}%'})
                    else:
                        query = """
                            SELECT DISTINCT prod_cd
                            FROM prod_code_base
                            ORDER BY prod_cd
                        """
                        cursor.execute(query)

                    results = [row[0] for row in cursor.fetchall() if row[0]]
                    logger.info(f"Successfully queried prod_code_base (without schema prefix)")

                except Exception as e2:
                    query_tried.append(f"prod_code_base: {str(e2)}")
                    logger.error(f"Failed to query product codes. Tried: {query_tried}")

            if results:
                logger.info(f"Found {len(results)} product codes")
            else:
                logger.warning(f"No product codes found. Queries tried: {query_tried}")

            cursor.close()
            connection.close()

            return results

        except Exception as e:
            logger.error(f"Error fetching product codes: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_mig_sources(self, environment: str, search_term: str = "") -> List[str]:
        """
        Fetches available migration sources from the database for autocomplete.

        Args:
            environment: Database environment (QAMain, Stage, JarvisQA1, JarvisQA2, Production)
            search_term: Optional search term to filter migration sources

        Returns:
            List of migration source strings
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Failed to connect for mig sources: {message}")
            return []

        try:
            cursor = connection.cursor()

            if search_term:
                query = """
                SELECT DISTINCT mig_source 
                FROM EDB.PROD_INST 
                WHERE mig_source IS NOT NULL
                AND UPPER(mig_source) LIKE UPPER(:search_term)
                ORDER BY mig_source
                """
                cursor.execute(query, {'search_term': f'%{search_term}%'})
            else:
                query = """
                SELECT DISTINCT mig_source 
                FROM EDB.PROD_INST 
                WHERE mig_source IS NOT NULL
                ORDER BY mig_source
                """
                cursor.execute(query)

            # Fetch results - filter out None values
            results = [row[0] for row in cursor.fetchall() if row[0]]

            cursor.close()
            connection.close()
            logger.info(f"Found {len(results)} migration source(s) for environment {environment}")
            return results

        except Exception as e:
            logger.error(f"Error fetching migration sources: {str(e)}")
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_email_platforms(self, environment: str, search_term: str = "") -> List[str]:
        """
        Fetches available email platforms from the database for autocomplete.

        Args:
            environment: Database environment to query
            search_term: Optional search term to filter email platforms

        Returns:
            List of email platform names
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Failed to connect to database: {message}")
            return []

        try:
            cursor = connection.cursor()

            if search_term:
                query = """
                SELECT DISTINCT ed.mail_platform
                FROM EDB.EMAIL_DOMPTR ed
                WHERE ed.mail_platform IS NOT NULL
                ORDER BY ed.mail_platform
                """
                cursor.execute(query)
            else:
                query = """
                SELECT DISTINCT ed.mail_platform
                FROM EDB.EMAIL_DOMPTR ed
                WHERE ed.mail_platform IS NOT NULL
                ORDER BY ed.mail_platform
                """
                cursor.execute(query)

            # Fetch numeric platform IDs and convert to names using EMAIL_PLATFORM_MAPPING
            results = []
            reverse_email_mapping = {v: k for k, v in self.EMAIL_PLATFORM_MAPPING.items()}

            for row in cursor.fetchall():
                platform_id = row[0]
                if platform_id in reverse_email_mapping:
                    platform_name = reverse_email_mapping[platform_id]
                    # Apply search filter if provided
                    if not search_term or search_term.upper() in platform_name.upper():
                        results.append(platform_name)

            cursor.close()
            connection.close()

            logger.info(f"Found {len(results)} email platform(s) for environment {environment}")
            return sorted(set(results))  # Remove duplicates and sort

        except Exception as e:
            logger.error(f"Error fetching email platforms: {str(e)}")
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def quick_search_account(
        self,
        environment: str,
        account_id: str = None,
        user_login_name: str = None,
        person_org_id: str = None
    ) -> Dict[str, Any]:
        """
        Quick search for account by Account ID, User Login Name, or Person Org ID.

        Args:
            environment: Database environment
            account_id: Account ID to search for
            user_login_name: User login name to search for
            person_org_id: Person Org ID to search for

        Returns:
            Dict with search results
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            return {
                "success": False,
                "error": message,
                "results": [],
                "count": 0
            }

        try:
            cursor = connection.cursor()

            # Build WHERE clause based on provided search criteria
            where_conditions = []
            params = {}

            if account_id:
                where_conditions.append("pi.account_id = :account_id")
                params['account_id'] = account_id

            if user_login_name:
                where_conditions.append("UPPER(po.user_login_name) = UPPER(:user_login_name)")
                params['user_login_name'] = user_login_name

            if person_org_id:
                where_conditions.append("poax.person_org_id = :person_org_id")
                params['person_org_id'] = person_org_id

            if not where_conditions:
                return {
                    "success": False,
                    "error": "No search criteria provided",
                    "results": [],
                    "count": 0
                }

            where_clause = " OR ".join(where_conditions)

            query = f"""
                SELECT DISTINCT 
                    pi.account_id,
                    poax.person_org_id,
                    pi.prod_inst_id,
                    pi.created_date,
                    pi.lifecycle_cd_id,
                    pcb.prod_cd,
                    po.user_login_name,
                    pi.mig_source,
                    pi.exp_date,
                    pi.subscription_unit,
                    pi.auto_renew_flag,
                    pi.parent_channel_id,
                    pi.webbuilder_platform_ind
                FROM EDB.PROD_INST pi
                INNER JOIN EDB.PROD_CODE_BASE pcb ON pcb.prod_id = pi.prod_id
                INNER JOIN EDB.PERSON_ORG_ACCOUNT_XREF poax ON poax.account_id = pi.account_id
                INNER JOIN EDB.PERSON_ORG po ON po.person_org_id = poax.person_org_id
                WHERE {where_clause}
                ORDER BY pi.created_date DESC
            """

            cursor.execute(query, params)

            # Fetch and process results (reuse the same logic as query_accounts_by_filters)
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                cursor.close()
                connection.close()
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "message": "No accounts found matching the search criteria"
                }

            # Process results into structured format
            accounts_dict = {}
            for row in rows:
                row_dict = dict(zip(columns, row))
                acct_id = row_dict['account_id']

                # Map parent_channel_id to brand
                parent_channel_id = row_dict.get('parent_channel_id')
                if parent_channel_id and parent_channel_id in self.PARENT_CHANNEL_TO_BRAND:
                    brand = self.PARENT_CHANNEL_TO_BRAND[parent_channel_id]
                elif parent_channel_id:
                    brand = str(parent_channel_id)
                else:
                    brand = 'N/A'

                if acct_id not in accounts_dict:
                    accounts_dict[acct_id] = {
                        "account_id": acct_id,
                        "person_org_id": row_dict.get('person_org_id'),
                        "user_login_name": row_dict.get('user_login_name'),
                        "brand_code": brand,
                        "prod_instances": []
                    }

                # Map lifecycle_cd_id to status
                lifecycle_cd_id = row_dict.get('lifecycle_cd_id')
                lifecycle_status_map = {
                    10: "ACTIVE",
                    11: "INACTIVE",
                    12: "SUSPENDED",
                    13: "CANCELLED"
                }
                lifecycle_status = lifecycle_status_map.get(lifecycle_cd_id, "UNKNOWN")

                # Get builder type from webbuilder_platform_ind
                webbuilder_platform_ind = row_dict.get('webbuilder_platform_ind')
                builder_type_name = None
                if webbuilder_platform_ind is not None:
                    try:
                        if isinstance(webbuilder_platform_ind, str) and webbuilder_platform_ind.strip():
                            webbuilder_platform_ind = int(webbuilder_platform_ind)
                        elif isinstance(webbuilder_platform_ind, (int, float)):
                            webbuilder_platform_ind = int(webbuilder_platform_ind)
                        else:
                            webbuilder_platform_ind = None
                    except (ValueError, AttributeError):
                        webbuilder_platform_ind = None

                    if webbuilder_platform_ind is not None and webbuilder_platform_ind in self.WEBBUILDER_PLATFORM_TO_BUILDER_TYPE:
                        builder_type_name = self.WEBBUILDER_PLATFORM_TO_BUILDER_TYPE[webbuilder_platform_ind]

                accounts_dict[acct_id]["prod_instances"].append({
                    "prod_inst_id": row_dict.get('prod_inst_id'),
                    "prod_cd": row_dict.get('prod_cd'),
                    "prod_code": row_dict.get('prod_cd'),
                    "prod_name": row_dict.get('prod_cd'),
                    "created_date": row_dict['created_date'].isoformat() if row_dict.get('created_date') else None,
                    "lifecycle_cd_id": lifecycle_cd_id,
                    "lifecycle_status": lifecycle_status,
                    "auto_renew_flag": row_dict.get('auto_renew_flag', 'N'),
                    "auto_renew": row_dict.get('auto_renew_flag', 'N'),
                    "exp_date": row_dict['exp_date'].isoformat() if row_dict.get('exp_date') else None,
                    "subscription_unit": row_dict.get('subscription_unit'),
                    "mig_source": row_dict.get('mig_source'),
                    "parent_channel_id": parent_channel_id,
                    "brand": brand,
                    "webbuilder_platform_ind": row_dict.get('webbuilder_platform_ind'),
                    "builder_type": builder_type_name
                })

            results = list(accounts_dict.values())

            cursor.close()
            connection.close()

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in quick search: {str(e)}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass

            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }

    def get_active_products_count(self, environment: str, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gets the count of active products (lifecycle_cd_id = 10) grouped by SKU.

        Args:
            environment: Database environment (QAMain, Stage, JarvisQA1, JarvisQA2, Production)
            offset: Number of rows to skip (for pagination)
            limit: Maximum number of rows to return

        Returns:
            List of dictionaries containing SKU and product count
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            logger.error(f"Failed to connect for active products count: {message}")
            return []

        try:
            cursor = connection.cursor()

            query = """
            SELECT COUNT(pi.prod_inst_id) as PRODUCT_COUNT, pcb.prod_cd as PROD_CD
            FROM EDB.PROD_INST pi
            INNER JOIN EDB.PROD_CODE_BASE pcb ON pcb.prod_id = pi.prod_id
            WHERE pi.lifecycle_cd_id = 10
            GROUP BY pcb.prod_cd
            ORDER BY COUNT(pi.prod_inst_id) DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """

            cursor.execute(query, {'offset': offset, 'limit': limit})

            # Fetch results
            columns = [col[0] for col in cursor.description]
            results = []
            for row in cursor.fetchall():
                result_dict = dict(zip(columns, row))
                results.append(result_dict)

            cursor.close()
            connection.close()
            logger.info(f"Found {len(results)} SKU(s) with active products for environment {environment}")
            return results

        except Exception as e:
            logger.error(f"Error fetching active products count: {str(e)}")
            if connection:
                try:
                    connection.close()
                except:
                    pass
            return []

    def get_random_account_by_sku(self, environment: str, sku: str) -> Dict[str, Any]:
        """
        Gets a random account that contains the specified SKU.

        Args:
            environment: Database environment (QAMain, Stage, JarvisQA1, JarvisQA2, Production)
            sku: Product code (SKU) to search for

        Returns:
            Dictionary containing account information (same structure as query_accounts_by_filters)
        """
        success, message, connection = self.get_db_connection(environment)

        if not success:
            return {
                "success": False,
                "error": message,
                "results": [],
                "count": 0
            }

        try:
            cursor = connection.cursor()

            # Optimized query to get a random active account with the specified SKU
            query = """
            SELECT pi.account_id, poax.person_org_id, pi.prod_inst_id, pi.created_date, 
                   pi.lifecycle_cd_id, pcb.prod_cd, po.user_login_name, pi.mig_source, 
                   pi.exp_date, pi.subscription_unit, pi.auto_renew_flag, pi.parent_channel_id, 
                   pi.webbuilder_platform_ind
            FROM EDB.PROD_INST pi
            INNER JOIN EDB.PROD_CODE_BASE pcb ON pcb.prod_id = pi.prod_id
            INNER JOIN EDB.PERSON_ORG_ACCOUNT_XREF poax ON poax.account_id = pi.account_id AND poax.relationship_id = 101
            INNER JOIN EDB.PERSON_ORG po ON po.person_org_id = poax.person_org_id
            WHERE pcb.prod_cd = :sku
            AND pi.lifecycle_cd_id = 10
            ORDER BY DBMS_RANDOM.VALUE
            FETCH FIRST 1 ROW ONLY
            """

            cursor.execute(query, {'sku': sku})

            # Fetch results
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                cursor.close()
                connection.close()
                return {
                    "success": True,
                    "results": [],
                    "count": 0
                }

            # Process results into structured format
            accounts_dict = {}
            for row in rows:
                row_dict = dict(zip(columns, row))
                acct_id = row_dict['account_id']

                # Map parent_channel_id to brand
                parent_channel_id = row_dict.get('parent_channel_id')
                if parent_channel_id and parent_channel_id in self.PARENT_CHANNEL_TO_BRAND:
                    brand = self.PARENT_CHANNEL_TO_BRAND[parent_channel_id]
                elif parent_channel_id:
                    brand = str(parent_channel_id)
                else:
                    brand = 'N/A'

                if acct_id not in accounts_dict:
                    accounts_dict[acct_id] = {
                        "account_id": acct_id,
                        "person_org_id": row_dict.get('person_org_id'),
                        "user_login_name": row_dict.get('user_login_name'),
                        "brand_code": brand,
                        "prod_instances": []
                    }

                # Map lifecycle_cd_id to status
                lifecycle_cd_id = row_dict.get('lifecycle_cd_id')
                lifecycle_status_map = {
                    10: "ACTIVE",
                    11: "INACTIVE",
                    12: "SUSPENDED",
                    13: "CANCELLED"
                }
                lifecycle_status = lifecycle_status_map.get(lifecycle_cd_id, "UNKNOWN")

                # Get builder type from webbuilder_platform_ind
                webbuilder_platform_ind = row_dict.get('webbuilder_platform_ind')
                builder_type_name = None
                if webbuilder_platform_ind is not None:
                    try:
                        if isinstance(webbuilder_platform_ind, str) and webbuilder_platform_ind.strip():
                            webbuilder_platform_ind = int(webbuilder_platform_ind)
                        elif isinstance(webbuilder_platform_ind, (int, float)):
                            webbuilder_platform_ind = int(webbuilder_platform_ind)
                        else:
                            webbuilder_platform_ind = None
                    except (ValueError, AttributeError):
                        webbuilder_platform_ind = None

                    if webbuilder_platform_ind is not None and webbuilder_platform_ind in self.WEBBUILDER_PLATFORM_TO_BUILDER_TYPE:
                        builder_type_name = self.WEBBUILDER_PLATFORM_TO_BUILDER_TYPE[webbuilder_platform_ind]

                accounts_dict[acct_id]["prod_instances"].append({
                    "prod_inst_id": row_dict.get('prod_inst_id'),
                    "prod_cd": row_dict.get('prod_cd'),
                    "prod_code": row_dict.get('prod_cd'),
                    "prod_name": row_dict.get('prod_cd'),
                    "created_date": row_dict['created_date'].isoformat() if row_dict.get('created_date') else None,
                    "lifecycle_cd_id": lifecycle_cd_id,
                    "lifecycle_status": lifecycle_status,
                    "auto_renew_flag": row_dict.get('auto_renew_flag', 'N'),
                    "auto_renew": row_dict.get('auto_renew_flag', 'N'),
                    "exp_date": row_dict['exp_date'].isoformat() if row_dict.get('exp_date') else None,
                    "subscription_unit": row_dict.get('subscription_unit'),
                    "mig_source": row_dict.get('mig_source'),
                    "parent_channel_id": parent_channel_id,
                    "brand": brand,
                    "webbuilder_platform_ind": row_dict.get('webbuilder_platform_ind'),
                    "builder_type": builder_type_name
                })

            results = list(accounts_dict.values())

            cursor.close()
            connection.close()

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching random account for SKU {sku}: {str(e)}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass

            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }

    def query_accounts_by_filters(
        self,
        environment: str,
        sku_list: List[str],
        brand: str = "All",
        show_only: str = "Active",
        sort_order: str = "newest",
        auto_renew: str = "All",
        subscription_term: str = "All",
        builder_type: str = "All",
        max_accounts: int = 100,
        every_sku: bool = False,
        mig_source_list: List[str] = None,
        every_mig_source: bool = False,
        email_platform_list: List[str] = None,
        every_email_platform: bool = False,
        created_date_from: str = None,
        created_date_to: str = None,
        exp_date_from: str = None,
        exp_date_to: str = None,
        selected_brands: List[str] = None,
        min_products: int = 0,
        max_products: int = 0
    ) -> Dict[str, Any]:
        """Query accounts from EDB based on filters using correct table names from database.py

        New filters:
            created_date_from: Filter accounts created after this date
            created_date_to: Filter accounts created before this date
            exp_date_from: Filter products expiring after this date
            exp_date_to: Filter products expiring before this date
            selected_brands: List of brands for multi-brand filter
            min_products: Minimum number of products per account
            max_products: Maximum number of products per account (0 = no limit)
        """

        success, message, connection = self.get_db_connection(environment)

        if not success:
            return {
                "success": False,
                "error": message,
                "results": [],
                "count": 0
            }

        try:
            cursor = connection.cursor()

            # Determine if we need to join with EMAIL_DOMPTR table
            need_email_join = not every_email_platform and email_platform_list

            # Build the query based on filters - using EDB schema prefix!
            if need_email_join:
                base_query = """
                    SELECT DISTINCT 
                        pi.account_id,
                        poax.person_org_id,
                        pi.prod_inst_id,
                        pi.created_date,
                        pi.lifecycle_cd_id,
                        pcb.prod_cd,
                        po.user_login_name,
                        pi.mig_source,
                        pi.exp_date,
                        pi.subscription_unit,
                        pi.auto_renew_flag,
                        pi.parent_channel_id,
                        pi.webbuilder_platform_ind
                    FROM EDB.PROD_INST pi
                    INNER JOIN EDB.PROD_CODE_BASE pcb ON pcb.prod_id = pi.prod_id
                    INNER JOIN EDB.PERSON_ORG_ACCOUNT_XREF poax ON poax.account_id = pi.account_id
                    INNER JOIN EDB.PERSON_ORG po ON po.person_org_id = poax.person_org_id
                    INNER JOIN EDB.EMAIL_DOMPTR ed ON pi.prod_inst_id = ed.prod_inst_id
                    WHERE 1=1
                """
            else:
                base_query = """
                    SELECT DISTINCT 
                        pi.account_id,
                        poax.person_org_id,
                        pi.prod_inst_id,
                        pi.created_date,
                        pi.lifecycle_cd_id,
                        pcb.prod_cd,
                        po.user_login_name,
                        pi.mig_source,
                        pi.exp_date,
                        pi.subscription_unit,
                        pi.auto_renew_flag,
                        pi.parent_channel_id,
                        pi.webbuilder_platform_ind
                    FROM EDB.PROD_INST pi
                    INNER JOIN EDB.PROD_CODE_BASE pcb ON pcb.prod_id = pi.prod_id
                    INNER JOIN EDB.PERSON_ORG_ACCOUNT_XREF poax ON poax.account_id = pi.account_id
                    INNER JOIN EDB.PERSON_ORG po ON po.person_org_id = poax.person_org_id
                    WHERE 1=1
                """

            params = {}

            # Add SKU filter only if every_sku is False
            if not every_sku and sku_list:
                sku_placeholders = ','.join([f":sku{i}" for i in range(len(sku_list))])
                base_query += f" AND pcb.prod_cd IN ({sku_placeholders})"
                for i, sku in enumerate(sku_list):
                    params[f"sku{i}"] = sku

            # Add brand filter (using parent_channel_id)
            # Support both single brand and multi-brand selection
            if selected_brands and len(selected_brands) > 0:
                # Multi-brand filter
                brand_mapping = {"NSI": 2, "BH": 263, "HG": 110}
                brand_ids = [brand_mapping.get(b) for b in selected_brands if b in brand_mapping]
                if brand_ids:
                    brand_placeholders = ','.join([f":brand_id{i}" for i in range(len(brand_ids))])
                    base_query += f" AND pi.parent_channel_id IN ({brand_placeholders})"
                    for i, brand_id in enumerate(brand_ids):
                        params[f"brand_id{i}"] = brand_id
            elif brand and brand != "All":
                # Single brand filter (backward compatibility)
                brand_mapping = {"NSI": 2, "BH": 263, "HG": 110}
                if brand in brand_mapping:
                    base_query += " AND pi.parent_channel_id = :brand_id"
                    params["brand_id"] = brand_mapping[brand]

            # Add lifecycle status filter (using lifecycle_cd_id)
            if show_only == "Active":
                base_query += " AND pi.lifecycle_cd_id = 10"
            elif show_only == "Inactive":
                base_query += " AND pi.lifecycle_cd_id = 11"
            elif show_only == "Active/Inactive":
                base_query += " AND pi.lifecycle_cd_id IN (10, 11)"
            elif show_only == "Expired":
                base_query += " AND pi.exp_date IS NOT NULL AND TRUNC(pi.exp_date) < TRUNC(SYSDATE)"
            elif show_only == "To Be Expired":
                base_query += " AND pi.exp_date IS NOT NULL AND TRUNC(pi.exp_date) > TRUNC(SYSDATE) AND TRUNC(pi.exp_date) < TRUNC(SYSDATE + 10)"

            # Add auto-renew filter
            if auto_renew == "Enabled":
                base_query += " AND pi.auto_renew_flag = 'Y'"
            elif auto_renew == "Disabled":
                base_query += " AND pi.auto_renew_flag = 'N'"

            # Add subscription term filter
            if subscription_term and subscription_term != "All":
                subscription_mapping = {
                    "Annual": "YR",
                    "Monthly": "MO",
                    "One Time Shipped": "OT_SHIPPED",
                    "One Time": "OT",
                    "Four Week": "4WK"
                }
                if subscription_term in subscription_mapping:
                    base_query += " AND pi.subscription_unit = :sub_unit"
                    params["sub_unit"] = subscription_mapping[subscription_term]

            # Add builder type filter (webbuilder_platform_ind)
            if builder_type and builder_type != "All":
                builder_mapping = {
                    "ImageCafe": "IMGCAFE",
                    "LeapCafe": "LEAPCAFE",
                    "Matrix": "MATRIX",
                    "Neo": "NEO",
                    "Nexus": "NEXUS",
                    "Siteplus": "SITEPLUS"
                }
                if builder_type in builder_mapping:
                    base_query += " AND pi.webbuilder_platform_ind = :builder"
                    params["builder"] = builder_mapping[builder_type]

            # Add migration source filter only if every_mig_source is False
            if not every_mig_source and mig_source_list:
                mig_source_placeholders = ','.join([f":mig_source{i}" for i in range(len(mig_source_list))])
                base_query += f" AND pi.mig_source IN ({mig_source_placeholders})"
                for i, mig_source in enumerate(mig_source_list):
                    params[f"mig_source{i}"] = mig_source

            # Add email platform filter only if every_email_platform is False
            if not every_email_platform and email_platform_list:
                # Convert platform names to IDs
                email_platform_ids = []
                for platform_name in email_platform_list:
                    if platform_name in self.EMAIL_PLATFORM_MAPPING:
                        email_platform_ids.append(self.EMAIL_PLATFORM_MAPPING[platform_name])

                if email_platform_ids:
                    email_placeholders = ','.join([f":email_platform{i}" for i in range(len(email_platform_ids))])
                    base_query += f" AND ed.mail_platform IN ({email_placeholders})"
                    for i, platform_id in enumerate(email_platform_ids):
                        params[f"email_platform{i}"] = platform_id

            # Add date range filters
            if created_date_from:
                base_query += " AND TRUNC(pi.created_date) >= TO_DATE(:created_from, 'YYYY-MM-DD')"
                params["created_from"] = str(created_date_from)

            if created_date_to:
                base_query += " AND TRUNC(pi.created_date) <= TO_DATE(:created_to, 'YYYY-MM-DD')"
                params["created_to"] = str(created_date_to)

            if exp_date_from:
                base_query += " AND TRUNC(pi.exp_date) >= TO_DATE(:exp_from, 'YYYY-MM-DD')"
                params["exp_from"] = str(exp_date_from)

            if exp_date_to:
                base_query += " AND TRUNC(pi.exp_date) <= TO_DATE(:exp_to, 'YYYY-MM-DD')"
                params["exp_to"] = str(exp_date_to)

            # Add sort order
            if sort_order == "newest":
                base_query += " ORDER BY pi.created_date DESC"
            elif sort_order == "oldest":
                base_query += " ORDER BY pi.created_date ASC"
            elif sort_order == "randomize":
                base_query += " ORDER BY DBMS_RANDOM.VALUE"
            else:
                base_query += " ORDER BY pi.created_date DESC"  # Default to newest

            # Add row limit using FETCH FIRST (Oracle 12c+)
            base_query = f"""
                SELECT * FROM (
                    {base_query}
                ) WHERE ROWNUM <= :max_accounts
            """
            params["max_accounts"] = max_accounts

            # Execute query
            logger.info(f"Executing accounts query: {base_query}")
            logger.info(f"Parameters: {params}")

            cursor.execute(base_query, params)

            # Fetch results
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()

            # Process results into structured format
            accounts_dict = {}
            for row in rows:
                row_dict = dict(zip(columns, row))
                acct_id = row_dict['account_id']

                # Map parent_channel_id to brand with comprehensive mapping
                parent_channel_id = row_dict.get('parent_channel_id')
                brand_map = {
                    1: "NSI",
                    21: "BH",
                    40: "HG",
                    # Add more mappings as discovered
                    # Some environments may use different IDs
                }

                if parent_channel_id and parent_channel_id in brand_map:
                    brand = brand_map.get(parent_channel_id)
                elif parent_channel_id:
                    # Show actual ID for debugging when not in mapping
                    brand = f"Unknown(ID:{parent_channel_id})"
                else:
                    brand = "Unknown"

                if acct_id not in accounts_dict:
                    accounts_dict[acct_id] = {
                        "account_id": acct_id,
                        "person_org_id": row_dict.get('person_org_id'),
                        "user_login_name": row_dict.get('user_login_name'),
                        "brand_code": brand,  # Add brand_code at account level
                        "prod_instances": []
                    }

                # Map lifecycle_cd_id to status
                lifecycle_cd_id = row_dict.get('lifecycle_cd_id')
                lifecycle_status_map = {
                    10: "ACTIVE",
                    11: "INACTIVE",
                    12: "SUSPENDED",
                    13: "CANCELLED"
                }
                lifecycle_status = lifecycle_status_map.get(lifecycle_cd_id, "UNKNOWN")

                # Map auto_renew_flag to readable format
                auto_renew = row_dict.get('auto_renew_flag', 'N')

                accounts_dict[acct_id]["prod_instances"].append({
                    "prod_inst_id": row_dict.get('prod_inst_id'),
                    "prod_cd": row_dict.get('prod_cd'),
                    "prod_code": row_dict.get('prod_cd'),  # Add for backward compatibility
                    "prod_name": row_dict.get('prod_cd'),  # Use prod_cd as name for now
                    "created_date": row_dict['created_date'].isoformat() if row_dict.get('created_date') else None,
                    "lifecycle_cd_id": lifecycle_cd_id,
                    "lifecycle_status": lifecycle_status,
                    "auto_renew_flag": auto_renew,
                    "auto_renew": auto_renew,  # Add for backward compatibility
                    "exp_date": row_dict['exp_date'].isoformat() if row_dict.get('exp_date') else None,
                    "subscription_unit": row_dict.get('subscription_unit'),
                    "mig_source": row_dict.get('mig_source'),
                    "parent_channel_id": parent_channel_id,
                    "brand": brand,
                    "webbuilder_platform_ind": row_dict.get('webbuilder_platform_ind')
                })

            results = list(accounts_dict.values())

            # Apply product count filters if specified
            if min_products > 0 or max_products > 0:
                filtered_results = []
                for account in results:
                    product_count = len(account.get("prod_instances", []))

                    # Check minimum
                    if min_products > 0 and product_count < min_products:
                        continue

                    # Check maximum (0 means no limit)
                    if max_products > 0 and product_count > max_products:
                        continue

                    filtered_results.append(account)

                results = filtered_results

            cursor.close()
            connection.close()

            # Log query to history
            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "environment": environment,
                "sku_list": sku_list,
                "brand": brand,
                "show_only": show_only,
                "sort_order": sort_order,
                "result_count": len(results),
                "success": True
            })

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error querying accounts: {e}", exc_info=True)
            if connection:
                try:
                    connection.close()
                except:
                    pass

            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "environment": environment,
                "error": str(e),
                "success": False
            })

            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }

    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get the query history"""
        return self.query_history

    def clear_query_history(self):
        """Clear the query history"""
        self.query_history = []
        logger.info("Query history cleared")

    def _initialize_azure_openai(self):
        """Initialize Azure OpenAI client using AzureOpenAIClient"""
        if not AZURE_OPENAI_AVAILABLE or AzureOpenAIClient is None:
            logger.warning("Azure OpenAI client not available")
            return

        try:
            # Use AzureOpenAIClient which auto-configures from environment variables
            self.azure_client = AzureOpenAIClient()

            # Check if it's properly configured
            if self.azure_client.is_configured():
                logger.info("Azure OpenAI client initialized and configured successfully")
            else:
                logger.warning("Azure OpenAI client created but not fully configured. Check environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
                self.azure_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            self.azure_client = None

    def generate_ai_insights(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights from query results"""
        if not self.azure_client:
            return {
                "success": False,
                "error": "Azure OpenAI not available"
            }

        try:
            # Prepare data summary for AI analysis
            results = query_results.get("results", [])
            if not results:
                return {
                    "success": False,
                    "error": "No results to analyze"
                }

            # Create summary statistics
            total_accounts = len(results)
            total_products = sum(len(acc["prod_instances"]) for acc in results)
            brands = {}
            lifecycle_statuses = {}
            auto_renew_stats = {"Y": 0, "N": 0}
            product_types = {}

            for account in results:
                brand = account.get("brand_code", "Unknown")
                brands[brand] = brands.get(brand, 0) + 1

                for prod in account.get("prod_instances", []):
                    status = prod.get("lifecycle_status", "Unknown")
                    lifecycle_statuses[status] = lifecycle_statuses.get(status, 0) + 1

                    auto_renew = prod.get("auto_renew", "N")
                    auto_renew_stats[auto_renew] = auto_renew_stats.get(auto_renew, 0) + 1

                    prod_code = prod.get("prod_code", "Unknown")
                    product_types[prod_code] = product_types.get(prod_code, 0) + 1

            # Create prompt for AI analysis
            prompt = f"""Analyze the following EDB query results and provide actionable insights:

Environment: {query_results.get('environment', 'Unknown')}
Total Accounts: {total_accounts}
Total Products: {total_products}

Brand Distribution:
{json.dumps(brands, indent=2)}

Lifecycle Status Distribution:
{json.dumps(lifecycle_statuses, indent=2)}

Auto-Renew Statistics:
Enabled: {auto_renew_stats.get('Y', 0)}
Disabled: {auto_renew_stats.get('N', 0)}

Top Product Types:
{json.dumps(dict(sorted(product_types.items(), key=lambda x: x[1], reverse=True)[:10]), indent=2)}

Please provide:
1. Key Insights: What patterns or trends do you see?
2. Recommendations: What actions should be taken?
3. Predictions: What might happen based on this data?
4. Data Quality: Any concerns about the data?
5. Business Impact: How might this affect business operations?

Be specific, actionable, and business-focused."""

            # Call Azure OpenAI using AzureOpenAIClient
            response = self.azure_client.chat_completion_create(
                messages=[
                    {"role": "system", "content": "You are an expert database analyst and business intelligence specialist. Provide clear, actionable insights based on data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            # Extract insights from response dictionary
            insights_text = response["choices"][0]["message"]["content"]

            return {
                "success": True,
                "insights": insights_text,
                "statistics": {
                    "total_accounts": total_accounts,
                    "total_products": total_products,
                    "brands": brands,
                    "lifecycle_statuses": lifecycle_statuses,
                    "auto_renew_stats": auto_renew_stats,
                    "top_products": dict(sorted(product_types.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_account_health_score(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a health score for an account based on various factors"""
        try:
            score = 100
            issues = []
            recommendations = []

            # Check auto-renew status
            prod_instances = account_data.get("prod_instances", [])
            auto_renew_disabled = sum(1 for p in prod_instances if p.get("auto_renew") == "N")
            if auto_renew_disabled > 0:
                score -= 15
                issues.append(f"{auto_renew_disabled} product(s) have auto-renew disabled")
                recommendations.append("Enable auto-renew for products to ensure continuity")

            # Check for inactive products
            inactive_products = sum(1 for p in prod_instances if p.get("lifecycle_status") == "INACTIVE")
            if inactive_products > 0:
                score -= 20
                issues.append(f"{inactive_products} inactive product(s)")
                recommendations.append("Review and reactivate or remove inactive products")

            # Check for expired products
            expired_products = sum(1 for p in prod_instances if p.get("lifecycle_status") == "EXPIRED")
            if expired_products > 0:
                score -= 25
                issues.append(f"{expired_products} expired product(s)")
                recommendations.append("Immediate action needed for expired products")

            # Check product diversity
            if len(prod_instances) == 1:
                score -= 5
                recommendations.append("Consider upselling additional products")

            # Determine health status
            if score >= 90:
                status = "Excellent"
                color = "green"
            elif score >= 75:
                status = "Good"
                color = "blue"
            elif score >= 60:
                status = "Fair"
                color = "orange"
            else:
                status = "Needs Attention"
                color = "red"

            return {
                "score": max(0, score),
                "status": status,
                "color": color,
                "issues": issues,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {
                "score": 0,
                "status": "Error",
                "color": "gray",
                "issues": [str(e)],
                "recommendations": []
            }

    def get_expiring_products_alert(self, results: Dict[str, Any], days: int = 30) -> Dict[str, Any]:
        """Identify products expiring within specified days"""
        try:
            from datetime import datetime, timedelta

            expiring_soon = []
            cutoff_date = datetime.now() + timedelta(days=days)

            for account in results.get("results", []):
                for prod in account.get("prod_instances", []):
                    # Check if product has expiration date
                    created_date = prod.get("created_date")
                    if created_date:
                        try:
                            created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                            # Assuming annual products expire 1 year from creation
                            expiration_dt = created_dt + timedelta(days=365)

                            if expiration_dt <= cutoff_date and expiration_dt > datetime.now():
                                expiring_soon.append({
                                    "account_id": account["account_id"],
                                    "prod_inst_id": prod.get("prod_inst_id"),
                                    "prod_code": prod.get("prod_code", prod.get("prod_cd", "N/A")),
                                    "prod_name": prod.get("prod_name", prod.get("prod_cd", "N/A")),
                                    "expiration_date": expiration_dt.isoformat(),
                                    "days_until_expiration": (expiration_dt - datetime.now()).days
                                })
                        except:
                            pass

            return {
                "success": True,
                "expiring_products": expiring_soon,
                "count": len(expiring_soon),
                "days_threshold": days
            }

        except Exception as e:
            logger.error(f"Error checking expiring products: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_revenue_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate potential revenue insights from query results"""
        try:
            # Sample pricing (should be configurable)
            product_pricing = {
                "DOM_COM": 12.99,
                "DOM_NET": 12.99,
                "DOM_ORG": 12.99,
                "E_BASIC": 5.99,
                "E_PRO": 9.99,
                "HOST_SHARED": 6.99,
                "HOST_VPS": 29.99,
                "SSL_BASIC": 49.99
            }

            total_revenue = 0
            active_revenue = 0
            at_risk_revenue = 0  # Products without auto-renew

            revenue_by_brand = {}
            revenue_by_product = {}

            for account in results.get("results", []):
                brand = account.get("brand_code", "Unknown")

                for prod in account.get("prod_instances", []):
                    prod_code = prod.get("prod_code", "")
                    status = prod.get("lifecycle_status", "")
                    auto_renew = prod.get("auto_renew", "N")

                    price = product_pricing.get(prod_code, 10.0) # Default price

                    total_revenue += price

                    if status == "ACTIVE":
                        active_revenue += price

                        if auto_renew == "N":
                            at_risk_revenue += price

                    # Track by brand
                    revenue_by_brand[brand] = revenue_by_brand.get(brand, 0) + price

                    # Track by product
                    revenue_by_product[prod_code] = revenue_by_product.get(prod_code, 0) + price

            return {
                "success": True,
                "total_revenue": round(total_revenue, 2),
                "active_revenue": round(active_revenue, 2),
                "at_risk_revenue": round(at_risk_revenue, 2),
                "revenue_by_brand": revenue_by_brand,
                "revenue_by_product": dict(sorted(revenue_by_product.items(), key=lambda x: x[1], reverse=True)[:10])
            }

        except Exception as e:
            logger.error(f"Error calculating revenue insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def export_results(self, results: Dict[str, Any], format: str = "json") -> str:
        """Export query results to various formats"""
        try:
            if format == "json":
                return json.dumps(results, indent=2, default=str)
            elif format == "csv":
                # Flatten results for CSV export
                flat_data = []
                for account in results.get("results", []):
                    for prod in account.get("prod_instances", []):
                        flat_data.append({
                            "Account ID": account.get("account_id", ""),
                            "Person Org ID": account.get("person_org_id", ""),
                            "Brand": account.get("brand_code", prod.get("brand", "")),
                            "Account Created": account.get("account_created_date", ""),
                            "Product Instance ID": prod.get("prod_inst_id", ""),
                            "Product Code": prod.get("prod_code", prod.get("prod_cd", "")),
                            "Product Name": prod.get("prod_name", prod.get("prod_cd", "")),
                            "Product Created": prod.get("created_date", ""),
                            "Lifecycle Status": prod.get("lifecycle_status", ""),
                            "Auto Renew": prod.get("auto_renew", prod.get("auto_renew_flag", ""))
                        })
                df = pd.DataFrame(flat_data)
                return df.to_csv(index=False)
            return ""
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""


def show_ui():
    """Main UI for EDB Query Manager"""

    # Initialize session state
    if 'edb_manager' not in st.session_state:
        st.session_state.edb_manager = EDBQueryManager()

    if 'selected_skus' not in st.session_state:
        st.session_state.selected_skus = []

    if 'selected_mig_sources' not in st.session_state:
        st.session_state.selected_mig_sources = []

    if 'available_mig_sources' not in st.session_state:
        st.session_state.available_mig_sources = []

    if 'selected_email_platforms' not in st.session_state:
        st.session_state.selected_email_platforms = []

    if 'available_email_platforms' not in st.session_state:
        st.session_state.available_email_platforms = []

    if 'query_results' not in st.session_state:
        st.session_state.query_results = None

    if 'results_page' not in st.session_state:
        st.session_state.results_page = 0

    if 'results_per_page' not in st.session_state:
        st.session_state.results_per_page = 10

    if 'filter_presets' not in st.session_state:
        st.session_state.filter_presets = {}

    if 'current_preset_name' not in st.session_state:
        st.session_state.current_preset_name = ""

    # Custom CSS for better visibility and styling
    st.markdown("""
    <style>
    /* Fix Tips section visibility */
    .tips-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tips-section h4 {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .tips-section ul {
        color: #374151;
        line-height: 1.8;
    }
    
    .tips-section li {
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .tips-section strong {
        color: #1f2937;
        font-weight: 600;
    }
    
    /* Enhance metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1f2937;
    }
    
    /* Better table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Status badges */
    .status-active {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-inactive {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 EDB Query Manager")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: white; margin: 0;'>Enterprise Database Account Query & Management</h3>
        <p style='color: #e0e7ff; margin: 0.5rem 0 0 0;'>
            Query and manage account information from EDB with real database connections
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if required dependencies are available
    if not ORACLEDB_AVAILABLE or not DB_CONFIG_AVAILABLE:
        st.error("⚠️ Required dependencies are missing.")

        if not ORACLEDB_AVAILABLE:
            st.error("❌ Oracle DB driver not installed.")
            st.code("pip install oracledb")

        if not DB_CONFIG_AVAILABLE:
            st.error("❌ Database configuration not available.")
            st.info("Ensure `tests/configs/database_config_variables.py` exists and is accessible.")

        return

    # Show info about optional AI features
    if not AZURE_OPENAI_AVAILABLE:
        st.info("ℹ️ Azure OpenAI is not installed. AI Insights features will be limited. Check azure_openai_client.py availability.")
    elif not hasattr(st.session_state.edb_manager, 'azure_client') or not st.session_state.edb_manager.azure_client:
        st.warning("⚠️ Azure OpenAI credentials not configured. AI Insights will not be available. Set environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Query Accounts",
        "📊 Active Products Count",
        "🗄️ Schema Explorer",
        "🤖 AI Insights",
        "📊 Query History",
        "ℹ️ Help & Tips"
    ])

    # Tab 1: Query Accounts
    with tab1:
        st.subheader("Query Accounts by Product Codes and Filters")

        # Environment Selection
        st.markdown("### 🌐 Select Environment")
        col1, col2, col3 = st.columns(3)

        with col1:
            environment = st.selectbox(
                "Database Environment",
                ["QAMain", "Stage", "JarvisQA1", "JarvisQA2", "Production"],
                help="Select the database environment to query"
            )

        with col2:
            every_sku = st.checkbox(
                "Every SKU",
                value=False,
                help="Check to query all SKUs (disables SKU selection)"
            )
            every_migration_source = st.checkbox(
                "Every Migration Source",
                value=False,
                help="Check to query all migration sources (disables Migration Source selection)"
            )
            every_email_platform = st.checkbox(
                "Every Email Platform",
                value=False,
                help="Check to query all email platforms (disables Email Platform selection)"
            )

        with col3:
            max_accounts = st.number_input(
                "Max Accounts to Return",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Maximum number of accounts to return"
            )

        st.markdown("---")

        # Quick Search Section
        st.markdown("### 🔎 Quick Search")
        st.markdown("Search directly by Account ID, User Login Name, or Person Org ID (bypasses other filters)")

        col1, col2, col3 = st.columns(3)
        with col1:
            quick_account_id = st.text_input(
                "Account ID",
                placeholder="e.g., 12345678",
                help="Enter Account ID for direct lookup"
            )
        with col2:
            quick_user_login = st.text_input(
                "User Login Name",
                placeholder="e.g., user@example.com",
                help="Enter User Login Name for direct lookup"
            )
        with col3:
            quick_person_org_id = st.text_input(
                "Person Org ID",
                placeholder="e.g., 98765432",
                help="Enter Person Org ID for direct lookup"
            )

        if quick_account_id or quick_user_login or quick_person_org_id:
            st.info("💡 Quick search is active - other filters will be bypassed")

        st.markdown("---")

        # SKU Selection with Search
        st.markdown("### 🎯 Product Code Selection (SKU)")

        # Disable SKU section if "Every SKU" is checked
        if not every_sku:
            col1, col2 = st.columns([3, 1])

            with col1:
                sku_search = st.text_input(
                    "Search Product Codes",
                    placeholder="Type to search for product codes (e.g., DOM_COM, E_BASIC)...",
                    help="Start typing to search for product codes"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Search SKUs", use_container_width=True):
                    with st.spinner("Fetching product codes..."):
                        try:
                            codes = st.session_state.edb_manager.get_product_codes(environment, sku_search)
                            if codes:
                                st.session_state.available_skus = codes
                                st.success(f"✅ Found {len(codes)} product code(s)")
                            else:
                                # No codes found - provide helpful message
                                if sku_search:
                                    st.warning(f"⚠️ No product codes found matching '{sku_search}'. Try:")
                                    st.markdown("- Searching with fewer characters (e.g., 'DOM' instead of 'DOM_COM')")
                                    st.markdown("- Leaving search empty to see all product codes")
                                    st.markdown("- Checking if the environment has data")
                                else:
                                    st.error("❌ No product codes found in database. Possible issues:")
                                    st.markdown("- Database connection issue")
                                    st.markdown("- Product table is empty")
                                    st.markdown("- Insufficient permissions")
                                    st.info("💡 Check the terminal logs for detailed error messages")
                        except Exception as e:
                            st.error(f"❌ Error fetching product codes: {str(e)}")
                            st.info("💡 Check database connectivity and credentials")

            # Display available SKUs for selection
            if 'available_skus' in st.session_state and st.session_state.available_skus:
                st.markdown("**Available Product Codes:**")

                # Multi-select with checkboxes
                selected_skus = st.multiselect(
                    "Select Product Codes (Multi-select enabled)",
                    options=st.session_state.available_skus,
                    default=st.session_state.selected_skus,
                    help="Select one or more product codes to query"
                )
                st.session_state.selected_skus = selected_skus

                if selected_skus:
                    st.info(f"✓ Selected {len(selected_skus)} product code(s): {', '.join(selected_skus)}")
        else:
            st.info("🔓 **Every SKU** is enabled - All product codes will be queried")
            # Clear selected SKUs when "Every SKU" is checked
            st.session_state.selected_skus = []

        st.markdown("---")

        # Migration Source Selection with Search
        st.markdown("### 🌐 Migration Source Selection")

        # Disable Migration Source section if "Every Migration Source" is checked
        if not every_migration_source:
            col1, col2 = st.columns([3, 1])

            with col1:
                mig_source_search = st.text_input(
                    "Search Migration Sources",
                    placeholder="Type to search for migration sources (e.g., BLUEHOST, HOSTGATOR)...",
                    help="Start typing to search for migration sources"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Search Migration Sources", use_container_width=True):
                    with st.spinner("Fetching migration sources..."):
                        try:
                            sources = st.session_state.edb_manager.get_mig_sources(environment, mig_source_search)
                            if sources:
                                st.session_state.available_mig_sources = sources
                                st.success(f"✅ Found {len(sources)} migration source(s)")
                            else:
                                # No sources found - provide helpful message
                                if mig_source_search:
                                    st.warning(f"⚠️ No migration sources found matching '{mig_source_search}'. Try:")
                                    st.markdown("- Searching with fewer characters")
                                    st.markdown("- Leaving search empty to see all migration sources")
                                    st.markdown("- Checking if the environment has migration data")
                                else:
                                    st.warning("⚠️ No migration sources found in database.")
                                    st.info("💡 This may be normal if no accounts have migration source data")
                        except Exception as e:
                            st.error(f"❌ Error fetching migration sources: {str(e)}")
                            st.info("💡 Check database connectivity and credentials")

            # Display available Migration Sources for selection
            if 'available_mig_sources' in st.session_state and st.session_state.available_mig_sources:
                st.markdown("**Available Migration Sources:**")

                # Multi-select with checkboxes
                selected_mig_sources = st.multiselect(
                    "Select Migration Sources (Multi-select enabled)",
                    options=st.session_state.available_mig_sources,
                    default=st.session_state.selected_mig_sources,
                    help="Select one or more migration sources to query"
                )
                st.session_state.selected_mig_sources = selected_mig_sources

                if selected_mig_sources:
                    st.info(f"✓ Selected {len(selected_mig_sources)} migration source(s): {', '.join(selected_mig_sources)}")
        else:
            st.info("🔓 **Every Migration Source** is enabled - All migration sources will be queried")
            # Clear selected migration sources when "Every Migration Source" is checked
            st.session_state.selected_mig_sources = []

        st.markdown("---")

        # Email Platform Selection with Search
        st.markdown("### 📧 Email Platform Selection")

        # Disable Email Platform section if "Every Email Platform" is checked
        if not every_email_platform:
            col1, col2 = st.columns([3, 1])

            with col1:
                email_platform_search = st.text_input(
                    "Search Email Platforms",
                    placeholder="Type to search for email platforms (e.g., OX Cloud, MS365, GWS)...",
                    help="Start typing to search for email platforms"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Search Email Platforms", use_container_width=True):
                    with st.spinner("Fetching email platforms..."):
                        try:
                            platforms = st.session_state.edb_manager.get_email_platforms(environment, email_platform_search)
                            if platforms:
                                st.session_state.available_email_platforms = platforms
                                st.success(f"✅ Found {len(platforms)} email platform(s)")
                            else:
                                # No platforms found - provide helpful message
                                if email_platform_search:
                                    st.warning(f"⚠️ No email platforms found matching '{email_platform_search}'. Try:")
                                    st.markdown("- Searching with fewer characters")
                                    st.markdown("- Leaving search empty to see all email platforms")
                                    st.markdown("- Checking if the environment has email data")
                                else:
                                    st.warning("⚠️ No email platforms found in database.")
                                    st.info("💡 This may be normal if no accounts have email platform data")
                        except Exception as e:
                            st.error(f"❌ Error fetching email platforms: {str(e)}")
                            st.info("💡 Check database connectivity and credentials")

            # Display available Email Platforms for selection
            if 'available_email_platforms' in st.session_state and st.session_state.available_email_platforms:
                st.markdown("**Available Email Platforms:**")

                # Multi-select with checkboxes
                selected_email_platforms = st.multiselect(
                    "Select Email Platforms (Multi-select enabled)",
                    options=st.session_state.available_email_platforms,
                    default=st.session_state.selected_email_platforms,
                    help="Select one or more email platforms to query"
                )
                st.session_state.selected_email_platforms = selected_email_platforms

                if selected_email_platforms:
                    st.info(f"✓ Selected {len(selected_email_platforms)} email platform(s): {', '.join(selected_email_platforms)}")
        else:
            st.info("🔓 **Every Email Platform** is enabled - All email platforms will be queried")
            # Clear selected email platforms when "Every Email Platform" is checked
            st.session_state.selected_email_platforms = []

        st.markdown("---")

        # Filter Options
        st.markdown("### 🎛️ Filter Options")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            brand = st.selectbox(
                "Brand",
                ["All", "NSI", "BH", "HG"],
                help="Filter by brand code"
            )

        with col2:
            show_only = st.selectbox(
                "Show Only",
                ["Active", "Inactive", "Active/Inactive", "All", "Expired", "To Be Expired"],
                help="Filter by lifecycle status"
            )

        with col3:
            sort_order = st.selectbox(
                "Sort By",
                ["Newest", "Oldest", "Randomize"],
                help="Sort accounts by creation date"
            )

        with col4:
            auto_renew = st.selectbox(
                "Auto-Renew",
                ["All", "Enabled", "Disabled"],
                help="Filter by auto-renew status"
            )

        # Additional filters
        col1, col2 = st.columns(2)

        with col1:
            subscription_term = st.selectbox(
                "Subscription Term",
                ["All", "Annual", "Monthly", "One Time Shipped", "One Time", "Four Week"],
                help="Filter by subscription term"
            )

        with col2:
            builder_type = st.selectbox(
                "Builder Type",
                ["All", "ImageCafe", "LeapCafe", "Matrix", "Neo", "Nexus", "Siteplus"],
                help="Filter by website builder type"
            )

        # Advanced filters (expandable section)
        with st.expander("🔧 Advanced Filters", expanded=False):
            st.markdown("#### Date Range Filters")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Created Date Range**")
                enable_created_date = st.checkbox("Enable Created Date Filter", key="enable_created_date")
                if enable_created_date:
                    created_date_from = st.date_input("From Date", key="created_from")
                    created_date_to = st.date_input("To Date", key="created_to")
                else:
                    created_date_from = None
                    created_date_to = None

            with col2:
                st.markdown("**Expiration Date Range**")
                enable_exp_date = st.checkbox("Enable Expiration Date Filter", key="enable_exp_date")
                if enable_exp_date:
                    exp_date_from = st.date_input("From Date", key="exp_from")
                    exp_date_to = st.date_input("To Date", key="exp_to")
                else:
                    exp_date_from = None
                    exp_date_to = None

            st.markdown("---")
            st.markdown("#### Multiple Brand Selection")
            enable_multi_brand = st.checkbox("Enable Multi-Brand Filter", key="enable_multi_brand")
            if enable_multi_brand:
                selected_brands = st.multiselect(
                    "Select Brands",
                    ["NSI", "BH", "HG"],
                    help="Select multiple brands to include in query"
                )
            else:
                selected_brands = None

            st.markdown("---")
            st.markdown("#### Account Filters")
            col1, col2 = st.columns(2)

            with col1:
                min_products = st.number_input(
                    "Minimum Products per Account",
                    min_value=0,
                    max_value=100,
                    value=0,
                    help="Filter accounts with at least this many products"
                )

            with col2:
                max_products = st.number_input(
                    "Maximum Products per Account",
                    min_value=0,
                    max_value=100,
                    value=0,
                    help="Filter accounts with at most this many products (0 = no limit)"
                )

        st.markdown("---")

        # Filter Presets
        st.markdown("### 💾 Filter Presets")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            preset_name = st.text_input(
                "Preset Name",
                placeholder="e.g., Active BH Domains",
                key="preset_name_input"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Current Filters", use_container_width=True):
                if preset_name:
                    # Save current filter configuration
                    st.session_state.filter_presets[preset_name] = {
                        "environment": environment,
                        "selected_skus": st.session_state.selected_skus.copy(),
                        "brand": brand,
                        "show_only": show_only,
                        "sort_order": sort_order,
                        "auto_renew": auto_renew,
                        "subscription_term": subscription_term,
                        "builder_type": builder_type,
                        "selected_mig_sources": st.session_state.selected_mig_sources.copy(),
                        "selected_email_platforms": st.session_state.selected_email_platforms.copy(),
                        "every_sku": every_sku,
                        "every_migration_source": every_migration_source,
                        "every_email_platform": every_email_platform
                    }
                    st.success(f"✅ Saved preset: {preset_name}")
                else:
                    st.error("⚠️ Please enter a preset name")

        with col3:
            if st.session_state.filter_presets:
                selected_preset = st.selectbox(
                    "Load Preset",
                    [""] + list(st.session_state.filter_presets.keys()),
                    key="preset_selector"
                )
                if selected_preset:
                    st.session_state.current_preset_name = selected_preset

        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Load Selected Preset", use_container_width=True):
                if st.session_state.current_preset_name:
                    preset = st.session_state.filter_presets[st.session_state.current_preset_name]
                    # Note: This requires form rerun to update UI values
                    st.info(f"ℹ️ Preset loaded: {st.session_state.current_preset_name}. Click Query to apply.")
                    # Store loaded values for next query
                    st.session_state.loaded_preset = preset
                else:
                    st.warning("⚠️ Please select a preset to load")

        st.markdown("---")

        # Query and Reset Buttons
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("🔄 Reset All Filters", use_container_width=True, type="secondary"):
                # Clear all session state filters
                st.session_state.selected_skus = []
                st.session_state.selected_mig_sources = []
                st.session_state.selected_email_platforms = []
                st.session_state.available_skus = []
                st.session_state.available_mig_sources = []
                st.session_state.available_email_platforms = []
                st.session_state.results_page = 0
                if 'loaded_preset' in st.session_state:
                    del st.session_state.loaded_preset
                st.success("✅ All filters reset!")
                st.rerun()

        with col2:
            query_button = st.button(
                "🚀 Query Accounts",
                use_container_width=True,
                type="primary"
            )

        with col3:
            st.markdown("")  # Placeholder for symmetry

        # Execute Query
        if query_button:
            # Check if quick search is active
            if quick_account_id or quick_user_login or quick_person_org_id:
                # Use quick search
                with st.spinner(f"Searching {environment} database..."):
                    import time
                    start_time = time.time()

                    results = st.session_state.edb_manager.quick_search_account(
                        environment=environment,
                        account_id=quick_account_id if quick_account_id else None,
                        user_login_name=quick_user_login if quick_user_login else None,
                        person_org_id=quick_person_org_id if quick_person_org_id else None
                    )

                    execution_time = time.time() - start_time
                    if results.get("success"):
                        results["execution_time"] = execution_time

                    st.session_state.query_results = results
            elif not every_sku and not st.session_state.selected_skus:
                st.error("⚠️ Please select at least one product code (SKU) or enable 'Every SKU'")
            else:
                # Use normal filter query
                with st.spinner(f"Querying {environment} database..."):
                    import time
                    start_time = time.time()

                    results = st.session_state.edb_manager.query_accounts_by_filters(
                        environment=environment,
                        sku_list=st.session_state.selected_skus,
                        brand=brand,
                        show_only=show_only,
                        sort_order=sort_order.lower(),
                        auto_renew=auto_renew,
                        subscription_term=subscription_term,
                        builder_type=builder_type,
                        max_accounts=max_accounts,
                        every_sku=every_sku,
                        mig_source_list=st.session_state.selected_mig_sources,
                        every_mig_source=every_migration_source,
                        email_platform_list=st.session_state.selected_email_platforms,
                        every_email_platform=every_email_platform,
                        created_date_from=created_date_from,
                        created_date_to=created_date_to,
                        exp_date_from=exp_date_from,
                        exp_date_to=exp_date_to,
                        selected_brands=selected_brands,
                        min_products=min_products,
                        max_products=max_products
                    )

                    execution_time = time.time() - start_time
                    if results.get("success"):
                        results["execution_time"] = execution_time

                    st.session_state.query_results = results

        # Display Results
        if st.session_state.query_results:
            results = st.session_state.query_results

            st.markdown("---")
            st.markdown("### 📊 Query Results")

            # Show applied filters summary
            with st.expander("🔍 Applied Filters Summary", expanded=False):
                filter_summary = []

                if quick_account_id:
                    filter_summary.append(f"**Quick Search:** Account ID = {quick_account_id}")
                if quick_user_login:
                    filter_summary.append(f"**Quick Search:** User Login = {quick_user_login}")
                if quick_person_org_id:
                    filter_summary.append(f"**Quick Search:** Person Org ID = {quick_person_org_id}")

                if not (quick_account_id or quick_user_login or quick_person_org_id):
                    # Normal filters
                    filter_summary.append(f"**Environment:** {environment}")

                    if every_sku:
                        filter_summary.append("**SKUs:** All (Every SKU enabled)")
                    elif st.session_state.selected_skus:
                        filter_summary.append(f"**SKUs:** {', '.join(st.session_state.selected_skus[:5])}" +
                                            (f" (+{len(st.session_state.selected_skus)-5} more)" if len(st.session_state.selected_skus) > 5 else ""))

                    if enable_multi_brand and selected_brands:
                        filter_summary.append(f"**Brands:** {', '.join(selected_brands)}")
                    elif brand != "All":
                        filter_summary.append(f"**Brand:** {brand}")

                    filter_summary.append(f"**Lifecycle:** {show_only}")
                    filter_summary.append(f"**Sort:** {sort_order}")

                    if auto_renew != "All":
                        filter_summary.append(f"**Auto-Renew:** {auto_renew}")

                    if subscription_term != "All":
                        filter_summary.append(f"**Subscription:** {subscription_term}")

                    if builder_type != "All":
                        filter_summary.append(f"**Builder:** {builder_type}")

                    if every_migration_source:
                        filter_summary.append("**Migration Source:** All")
                    elif st.session_state.selected_mig_sources:
                        filter_summary.append(f"**Migration Source:** {', '.join(st.session_state.selected_mig_sources[:3])}" +
                                            (f" (+{len(st.session_state.selected_mig_sources)-3} more)" if len(st.session_state.selected_mig_sources) > 3 else ""))

                    if every_email_platform:
                        filter_summary.append("**Email Platform:** All")
                    elif st.session_state.selected_email_platforms:
                        filter_summary.append(f"**Email Platform:** {', '.join(st.session_state.selected_email_platforms[:3])}" +
                                            (f" (+{len(st.session_state.selected_email_platforms)-3} more)" if len(st.session_state.selected_email_platforms) > 3 else ""))

                    if enable_created_date and (created_date_from or created_date_to):
                        date_str = f"Created: {created_date_from or 'Any'} to {created_date_to or 'Any'}"
                        filter_summary.append(f"**Date Range:** {date_str}")

                    if enable_exp_date and (exp_date_from or exp_date_to):
                        date_str = f"Expiration: {exp_date_from or 'Any'} to {exp_date_to or 'Any'}"
                        filter_summary.append(f"**Date Range:** {date_str}")

                    if min_products > 0 or max_products > 0:
                        filter_summary.append(f"**Product Count:** {min_products}-{max_products if max_products > 0 else '∞'}")

                    filter_summary.append(f"**Max Results:** {max_accounts}")

                for item in filter_summary:
                    st.markdown(f"- {item}")

            if results.get("success"):
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Accounts", results["count"])

                with col2:
                    total_products = sum(len(acc["prod_instances"]) for acc in results["results"])
                    st.metric("Total Products", total_products)

                with col3:
                    st.metric("Environment", results["environment"])

                with col4:
                    query_time = datetime.fromisoformat(results["timestamp"]).strftime("%H:%M:%S")
                    st.metric("Query Time", query_time)

                with col5:
                    exec_time = results.get("execution_time", 0)
                    st.metric("Execution", f"{exec_time:.2f}s",
                             delta="Fast" if exec_time < 2 else "Slow",
                             delta_color="normal" if exec_time < 2 else "inverse")

                if results["count"] > 0:
                    st.markdown("---")

                    # Pagination controls
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        results_per_page = st.selectbox(
                            "Results per page",
                            [5, 10, 20, 50],
                            index=1,
                            key="results_per_page_select"
                        )
                        st.session_state.results_per_page = results_per_page

                    with col2:
                        total_pages = (results["count"] + results_per_page - 1) // results_per_page
                        current_page = st.session_state.results_page

                        st.markdown(f"<div style='text-align: center; padding: 1rem;'>"
                                  f"<strong>Page {current_page + 1} of {total_pages}</strong>"
                                  f"</div>", unsafe_allow_html=True)

                    with col3:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("⬅️ Previous", disabled=current_page == 0, key="prev_page"):
                                st.session_state.results_page = max(0, current_page - 1)
                                st.rerun()
                        with col_b:
                            if st.button("Next ➡️", disabled=current_page >= total_pages - 1, key="next_page"):
                                st.session_state.results_page = min(total_pages - 1, current_page + 1)
                                st.rerun()

                    st.markdown("---")

                    # Calculate pagination slice
                    start_idx = current_page * results_per_page
                    end_idx = min(start_idx + results_per_page, results["count"])
                    paginated_results = results["results"][start_idx:end_idx]

                    # Query Statistics Summary
                    with st.expander("📊 Query Statistics", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)

                        # Calculate statistics
                        all_brands = {}
                        all_statuses = {}
                        all_auto_renew = {"Y": 0, "N": 0}
                        all_subscriptions = {}
                        all_builder_types = {}
                        all_mig_sources = {}

                        for account in results["results"]:
                            brand = account.get("brand_code", "Unknown")
                            all_brands[brand] = all_brands.get(brand, 0) + 1

                            for prod in account.get("prod_instances", []):
                                status = prod.get("lifecycle_status", "Unknown")
                                all_statuses[status] = all_statuses.get(status, 0) + 1

                                auto_renew = prod.get("auto_renew", "N")
                                all_auto_renew[auto_renew] = all_auto_renew.get(auto_renew, 0) + 1

                                sub = prod.get("subscription_unit", "Unknown")
                                all_subscriptions[sub] = all_subscriptions.get(sub, 0) + 1

                                builder = prod.get("builder_type")
                                if builder:
                                    all_builder_types[builder] = all_builder_types.get(builder, 0) + 1

                                mig = prod.get("mig_source")
                                if mig:
                                    all_mig_sources[mig] = all_mig_sources.get(mig, 0) + 1

                        with col1:
                            st.markdown("**Brand Distribution**")
                            for brand, count in sorted(all_brands.items(), key=lambda x: x[1], reverse=True):
                                st.text(f"{brand}: {count}")

                        with col2:
                            st.markdown("**Lifecycle Status**")
                            for status, count in sorted(all_statuses.items(), key=lambda x: x[1], reverse=True):
                                st.text(f"{status}: {count}")

                        with col3:
                            st.markdown("**Auto-Renew**")
                            st.text(f"Enabled: {all_auto_renew.get('Y', 0)}")
                            st.text(f"Disabled: {all_auto_renew.get('N', 0)}")
                            renewal_rate = (all_auto_renew.get('Y', 0) / max(sum(all_auto_renew.values()), 1)) * 100
                            st.text(f"Rate: {renewal_rate:.1f}%")

                        with col4:
                            st.markdown("**Top Subscriptions**")
                            for sub, count in sorted(all_subscriptions.items(), key=lambda x: x[1], reverse=True)[:3]:
                                st.text(f"{sub}: {count}")

                        if all_builder_types:
                            st.markdown("---")
                            st.markdown("**Builder Types Distribution**")
                            builder_df = pd.DataFrame([
                                {"Builder": k, "Count": v}
                                for k, v in sorted(all_builder_types.items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.bar_chart(builder_df.set_index("Builder"))

                        if all_mig_sources:
                            st.markdown("---")
                            st.markdown("**Top Migration Sources**")
                            mig_df = pd.DataFrame([
                                {"Source": k, "Count": v}
                                for k, v in sorted(all_mig_sources.items(), key=lambda x: x[1], reverse=True)[:10]
                            ])
                            st.dataframe(mig_df, use_container_width=True, hide_index=True)

                    # Display each account with health score
                    for idx, account in enumerate(paginated_results, start_idx + 1):
                        # Calculate health score
                        health_data = st.session_state.edb_manager.get_account_health_score(account)

                        # Health score badge
                        health_badge = f"""
                        <span style='background: {health_data['color']}; color: white; 
                                     padding: 0.25rem 0.75rem; border-radius: 12px; 
                                     font-size: 0.875rem; font-weight: 500;'>
                            {health_data['status']} ({health_data['score']}/100)
                        </span>
                        """

                        with st.expander(
                            f"**Account #{idx}: {account['account_id']}** - {len(account['prod_instances'])} Product(s) | Health: ",
                            expanded=(idx == 1)
                        ):
                            # Display health badge
                            st.markdown(health_badge, unsafe_allow_html=True)
                            st.markdown("")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Account ID", account['account_id'])

                            with col2:
                                st.metric("Person Org ID", account['person_org_id'])

                            with col3:
                                st.metric("Brand", account.get('brand_code', 'N/A'))

                            with col4:
                                st.metric("Health Score", f"{health_data['score']}/100")

                            # Show health issues and recommendations
                            if health_data['issues']:
                                st.warning("**Issues Detected:**")
                                for issue in health_data['issues']:
                                    st.markdown(f"- {issue}")

                            if health_data['recommendations']:
                                st.info("**Recommendations:**")
                                for rec in health_data['recommendations']:
                                    st.markdown(f"- {rec}")

                            # Product instances table
                            st.markdown("**Product Instances:**")

                            prod_data = []
                            for prod in account['prod_instances']:
                                prod_data.append({
                                    "Product Instance ID": prod.get('prod_inst_id', 'N/A'),
                                    "Product Code": prod.get('prod_code', prod.get('prod_cd', 'N/A')),
                                    "Product Name": prod.get('prod_name', prod.get('prod_cd', 'N/A')),
                                    "Created Date": prod.get('created_date', 'N/A'),
                                    "Exp Date": prod.get('exp_date', 'N/A'),
                                    "Status": prod.get('lifecycle_status', 'N/A'),
                                    "Auto-Renew": prod.get('auto_renew', prod.get('auto_renew_flag', 'N/A')),
                                    "Subscription": prod.get('subscription_unit', 'N/A'),
                                    "Builder Type": prod.get('builder_type', 'N/A'),
                                    "Migration Source": prod.get('mig_source', 'N/A')
                                })

                            prod_df = pd.DataFrame(prod_data)
                            st.dataframe(prod_df, use_container_width=True, hide_index=True)

                    # Export options
                    st.markdown("---")
                    st.markdown("### 💾 Export Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        json_export = st.session_state.edb_manager.export_results(results, "json")
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_export,
                            file_name=f"edb_query_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    with col2:
                        csv_export = st.session_state.edb_manager.export_results(results, "csv")
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_export,
                            file_name=f"edb_query_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col3:
                        if st.button("🗑️ Clear Results", use_container_width=True):
                            st.session_state.query_results = None
                            st.rerun()

                else:
                    st.info("ℹ️ No accounts found matching the specified criteria")

            else:
                st.error(f"❌ Query failed: {results.get('error', 'Unknown error')}")
                st.markdown("**Troubleshooting Tips:**")
                st.markdown("- Check database connectivity")
                st.markdown("- Verify database credentials")
                st.markdown("- Ensure the selected environment is accessible")

    # Tab 2: Active Products Count
    with tab2:
        st.subheader("📊 See Active Products Count")
        st.markdown("View the count of active products grouped by SKU")

        # Environment Selection
        col1, col2 = st.columns([2, 1])
        with col1:
            active_products_env = st.selectbox(
                "Select Environment",
                ["QAMain", "Stage", "JarvisQA1", "JarvisQA2", "Production"],
                key="active_products_env",
                help="Select the database environment to query"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            load_active_products_btn = st.button("📊 Load Active Products", use_container_width=True, type="primary")

        # Pagination controls
        if 'active_products_offset' not in st.session_state:
            st.session_state.active_products_offset = 0
        if 'active_products_limit' not in st.session_state:
            st.session_state.active_products_limit = 50

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            items_per_page = st.selectbox(
                "Items per page",
                [10, 20, 50, 100],
                index=2,
                key="items_per_page"
            )
            st.session_state.active_products_limit = items_per_page

        if load_active_products_btn or 'active_products_data' in st.session_state:
            if load_active_products_btn:
                st.session_state.active_products_offset = 0  # Reset offset on new load

            with st.spinner(f"Loading active products from {active_products_env}..."):
                active_products = st.session_state.edb_manager.get_active_products_count(
                    active_products_env,
                    st.session_state.active_products_offset,
                    st.session_state.active_products_limit
                )
                st.session_state.active_products_data = active_products
                st.session_state.active_products_current_env = active_products_env

            if st.session_state.active_products_data:
                st.markdown("---")
                st.markdown("### 📊 Active Products by SKU")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SKUs Shown", len(st.session_state.active_products_data))
                with col2:
                    total_active = sum([p.get('PRODUCT_COUNT', 0) for p in st.session_state.active_products_data])
                    st.metric("Total Active Products", f"{total_active:,}")
                with col3:
                    st.metric("Environment", st.session_state.active_products_current_env)

                st.markdown("---")

                # Display active products in a table
                active_df = pd.DataFrame(st.session_state.active_products_data)
                active_df = active_df.rename(columns={
                    'PROD_CD': 'SKU',
                    'PRODUCT_COUNT': 'Active Products Count'
                })

                # Add row numbers
                active_df.insert(0, '#', range(st.session_state.active_products_offset + 1,
                                              st.session_state.active_products_offset + len(active_df) + 1))

                st.dataframe(active_df, use_container_width=True, hide_index=True, height=400)

                # Add click functionality to get random account
                st.markdown("---")
                st.markdown("### 🎲 Get Random Account by SKU")

                sku_options = [p.get('PROD_CD') for p in st.session_state.active_products_data]
                selected_sku = st.selectbox(
                    "Select SKU to get a random account",
                    options=sku_options,
                    key="selected_sku_for_random"
                )

                if st.button("🎲 Get Random Account", type="primary"):
                    with st.spinner(f"Fetching random account for {selected_sku}..."):
                        random_account = st.session_state.edb_manager.get_random_account_by_sku(
                            st.session_state.active_products_current_env,
                            selected_sku
                        )
                        st.session_state.random_account_result = random_account

                # Display random account result
                if 'random_account_result' in st.session_state and st.session_state.random_account_result:
                    result = st.session_state.random_account_result

                    if result.get("success") and result.get("count") > 0:
                        st.success(f"✅ Found random account for SKU: {selected_sku}")

                        account = result["results"][0]

                        # Display account details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Account ID", account['account_id'])
                        with col2:
                            st.metric("Person Org ID", account['person_org_id'])
                        with col3:
                            st.metric("Brand", account.get('brand_code', 'N/A'))

                        # Display product instances
                        st.markdown("**Product Instances:**")
                        prod_data = []
                        for prod in account['prod_instances']:
                            prod_data.append({
                                "Product Instance ID": prod.get('prod_inst_id', 'N/A'),
                                "Product Code": prod.get('prod_code', prod.get('prod_cd', 'N/A')),
                                "Created Date": prod.get('created_date', 'N/A'),
                                "Status": prod.get('lifecycle_status', 'N/A'),
                                "Auto-Renew": prod.get('auto_renew', prod.get('auto_renew_flag', 'N/A')),
                                "Builder Type": prod.get('builder_type', 'N/A')
                            })

                        prod_df = pd.DataFrame(prod_data)
                        st.dataframe(prod_df, use_container_width=True, hide_index=True)

                    elif result.get("success") and result.get("count") == 0:
                        st.warning(f"⚠️ No active accounts found for SKU: {selected_sku}")
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

                # Pagination controls
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    if st.button("⬅️ Previous Page", disabled=st.session_state.active_products_offset == 0):
                        st.session_state.active_products_offset = max(0, st.session_state.active_products_offset - st.session_state.active_products_limit)
                        st.rerun()

                with col2:
                    page_num = (st.session_state.active_products_offset // st.session_state.active_products_limit) + 1
                    st.markdown(f"<div style='text-align: center; padding: 0.5rem;'><strong>Page {page_num}</strong></div>", unsafe_allow_html=True)

                with col3:
                    has_more = len(st.session_state.active_products_data) == st.session_state.active_products_limit
                    if st.button("Next Page ➡️", disabled=not has_more):
                        st.session_state.active_products_offset += st.session_state.active_products_limit
                        st.rerun()

                # Export button
                st.markdown("---")
                csv = active_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Active Products List",
                    data=csv,
                    file_name=f"active_products_{active_products_env}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No active products found")

    # Tab 3: Schema Explorer
    with tab3:
        st.subheader("🗄️ Database Schema Explorer")
        st.markdown("Explore database tables, columns, and relationships")

        # Environment Selection for Schema Explorer
        col1, col2 = st.columns([2, 1])
        with col1:
            schema_env = st.selectbox(
                "Select Environment for Schema Exploration",
                ["QAMain", "Stage", "JarvisQA1", "JarvisQA2", "Production"],
                key="schema_env_select",
                help="Select the database environment to explore"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            load_schema_btn = st.button("🔄 Load Schema", use_container_width=True, type="primary")

        if load_schema_btn:
            with st.spinner(f"Loading schema from {schema_env}..."):
                # Load tables
                tables = st.session_state.edb_manager.get_available_tables(schema_env)
                table_details = st.session_state.edb_manager.get_table_details(schema_env)
                relationships = st.session_state.edb_manager.get_table_relationships(schema_env)

                st.session_state.schema_tables = tables
                st.session_state.schema_table_details = table_details
                st.session_state.schema_relationships = relationships
                st.session_state.schema_env_loaded = schema_env

        # Display Schema Information
        if 'schema_tables' in st.session_state and st.session_state.schema_tables:
            st.markdown("---")

            # Summary metrics
            st.markdown("### 📊 Schema Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Tables", len(st.session_state.schema_tables))

            with col2:
                total_rows = sum([t.get('num_rows', 0) for t in st.session_state.schema_table_details])
                st.metric("Total Rows", f"{total_rows:,}")

            with col3:
                st.metric("Relationships", len(st.session_state.schema_relationships))

            with col4:
                st.metric("Environment", st.session_state.schema_env_loaded)

            # Create sub-tabs for different views
            schema_tab1, schema_tab2, schema_tab3 = st.tabs([
                "📋 Tables List",
                "🔗 Relationships (ER Diagram)",
                "🔍 Table Details"
            ])

            # Sub-tab 1: Tables List
            with schema_tab1:
                st.markdown("#### Database Tables")

                if st.session_state.schema_table_details:
                    # Create DataFrame for tables
                    tables_df = pd.DataFrame(st.session_state.schema_table_details)
                    tables_df = tables_df.rename(columns={
                        'table_name': 'Table Name',
                        'num_rows': 'Row Count',
                        'blocks': 'Blocks',
                        'avg_row_len': 'Avg Row Length',
                        'last_analyzed': 'Last Analyzed'
                    })

                    # Add search filter
                    table_search = st.text_input("🔍 Search tables", placeholder="Type to filter tables...")

                    if table_search:
                        tables_df = tables_df[tables_df['Table Name'].str.contains(table_search.upper(), na=False)]

                    st.dataframe(
                        tables_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )

                    # Download button
                    csv = tables_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Tables List",
                        data=csv,
                        file_name=f"tables_list_{schema_env}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No table details available")

            # Sub-tab 2: Relationships (ER Diagram)
            with schema_tab2:
                st.markdown("#### Entity Relationship Diagram")

                if st.session_state.schema_relationships:
                    # Display relationships as a table
                    st.markdown("##### Foreign Key Relationships")
                    rel_df = pd.DataFrame(st.session_state.schema_relationships)
                    rel_df = rel_df.rename(columns={
                        'constraint_name': 'Constraint',
                        'child_table': 'Child Table',
                        'child_column': 'Child Column',
                        'parent_table': 'Parent Table',
                        'parent_column': 'Parent Column'
                    })

                    # Add filter
                    rel_filter = st.text_input("🔍 Filter relationships", placeholder="Type to filter...")

                    if rel_filter:
                        rel_df = rel_df[
                            rel_df['Child Table'].str.contains(rel_filter.upper(), na=False) |
                            rel_df['Parent Table'].str.contains(rel_filter.upper(), na=False)
                        ]

                    st.dataframe(rel_df, use_container_width=True, hide_index=True, height=400)

                    # Generate visual ER diagram using Mermaid
                    st.markdown("---")
                    st.markdown("##### Visual ER Diagram (Mermaid)")

                    # Build mermaid diagram
                    mermaid_code = "erDiagram\n"

                    # Group relationships by tables
                    table_rels = {}
                    for rel in st.session_state.schema_relationships:
                        parent = rel['parent_table']
                        child = rel['child_table']
                        parent_col = rel['parent_column']
                        child_col = rel['child_column']

                        if parent not in table_rels:
                            table_rels[parent] = []
                        table_rels[parent].append({
                            'child': child,
                            'parent_col': parent_col,
                            'child_col': child_col
                        })

                    # Generate relationships (limit to prevent overwhelming diagram)
                    rel_count = 0
                    max_rels = 50  # Limit for readability

                    for parent, rels in table_rels.items():
                        for rel in rels:
                            if rel_count >= max_rels:
                                break
                            child = rel['child']
                            mermaid_code += f'    {parent} ||--o{{ {child} : "{rel["parent_col"]} -> {rel["child_col"]}"\n'
                            rel_count += 1
                        if rel_count >= max_rels:
                            break

                    if rel_count >= max_rels:
                        st.info(f"ℹ️ Showing first {max_rels} relationships. Use the filter above to focus on specific tables.")

                    # Display mermaid diagram
                    st.code(mermaid_code, language="mermaid")

                    st.markdown("""
                    **How to view:**
                    1. Copy the above Mermaid code
                    2. Paste it into [Mermaid Live Editor](https://mermaid.live/)
                    3. Or use a Mermaid preview extension in your IDE
                    """)

                    # Download relationships
                    csv = rel_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Relationships",
                        data=csv,
                        file_name=f"relationships_{schema_env}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No foreign key relationships found in this schema")

            # Sub-tab 3: Table Details
            with schema_tab3:
                st.markdown("#### Detailed Table Information")

                # Select a table to view details
                selected_table = st.selectbox(
                    "Select a table to view columns",
                    options=st.session_state.schema_tables,
                    key="selected_table_detail"
                )

                if selected_table:
                    with st.spinner(f"Loading columns for {selected_table}..."):
                        columns = st.session_state.edb_manager.get_table_columns(
                            st.session_state.schema_env_loaded,
                            selected_table
                        )

                    if columns:
                        st.markdown(f"##### Columns in `{selected_table}`")

                        # Display columns
                        cols_df = pd.DataFrame(columns)
                        cols_df = cols_df.rename(columns={
                            'column_name': 'Column Name',
                            'data_type': 'Data Type',
                            'data_length': 'Length',
                            'nullable': 'Nullable',
                            'column_id': 'Position'
                        })

                        st.dataframe(cols_df, use_container_width=True, hide_index=True)

                        # Show related tables
                        st.markdown("---")
                        st.markdown("##### Related Tables")

                        # Find relationships for this table
                        parent_rels = [r for r in st.session_state.schema_relationships
                                     if r['parent_table'] == selected_table]
                        child_rels = [r for r in st.session_state.schema_relationships
                                    if r['child_table'] == selected_table]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("###### Tables referencing this table")
                            if parent_rels:
                                for rel in parent_rels:
                                    st.markdown(f"- `{rel['child_table']}` → `{rel['child_column']}`")
                            else:
                                st.info("No child tables")

                        with col2:
                            st.markdown("###### Tables referenced by this table")
                            if child_rels:
                                for rel in child_rels:
                                    st.markdown(f"- `{rel['parent_table']}` ← `{rel['parent_column']}`")
                            else:
                                st.info("No parent tables")

                        # Download columns
                        csv = cols_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {selected_table} Columns",
                            data=csv,
                            file_name=f"columns_{selected_table}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No columns found for table {selected_table}")
        else:
            st.info("👆 Click 'Load Schema' button to explore the database schema")

    # Tab 4: AI Insights
    with tab4:
        st.subheader("🤖 AI-Powered Insights & Analysis")

        if not AZURE_OPENAI_AVAILABLE:
            st.warning("⚠️ Azure OpenAI client is not available. AI Insights features are disabled.")
            st.info("Azure OpenAI client auto-connects when properly configured. Set environment variables:")
            st.code("""
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
            """)
            st.info("The client will automatically connect on next restart if variables are set correctly.")

        if st.session_state.query_results and st.session_state.query_results.get("success"):
            results = st.session_state.query_results

            st.markdown("### 📊 Query Summary")

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)

            total_products = sum(len(acc["prod_instances"]) for acc in results["results"])
            active_products = sum(
                sum(1 for p in acc["prod_instances"] if p.get("lifecycle_status") == "ACTIVE")
                for acc in results["results"]
            )
            auto_renew_enabled = sum(
                sum(1 for p in acc["prod_instances"] if p.get("auto_renew", p.get("auto_renew_flag")) == "Y")
                for acc in results["results"]
            )

            with col1:
                st.metric("Total Accounts", results["count"])

            with col2:
                st.metric("Total Products", total_products)

            with col3:
                st.metric("Active Products", active_products)

            with col4:
                st.metric("Auto-Renew Enabled", auto_renew_enabled)

            st.markdown("---")

            # Revenue Insights
            st.markdown("### 💰 Revenue Insights")
            with st.spinner("Calculating revenue insights..."):
                revenue_data = st.session_state.edb_manager.get_revenue_insights(results)

                if revenue_data.get("success"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Revenue",
                            f"${revenue_data['total_revenue']:,.2f}",
                            help="Total estimated revenue from all products"
                        )

                    with col2:
                        st.metric(
                            "Active Revenue",
                            f"${revenue_data['active_revenue']:,.2f}",
                            help="Revenue from active products only"
                        )

                    with col3:
                        st.metric(
                            "At-Risk Revenue",
                            f"${revenue_data['at_risk_revenue']:,.2f}",
                            help="Revenue from products without auto-renew",
                            delta=f"-{revenue_data['at_risk_revenue']:.2f}",
                            delta_color="inverse"
                        )

                    # Revenue by brand
                    if revenue_data.get("revenue_by_brand"):
                        st.markdown("#### Revenue by Brand")
                        brand_rev_df = pd.DataFrame([
                            {"Brand": k, "Revenue": f"${v:,.2f}"}
                            for k, v in revenue_data["revenue_by_brand"].items()
                        ])
                        st.dataframe(brand_rev_df, use_container_width=True, hide_index=True)

                    # Top revenue products
                    if revenue_data.get("revenue_by_product"):
                        st.markdown("#### Top Revenue Products")
                        prod_rev_df = pd.DataFrame([
                            {"Product": k, "Revenue": f"${v:,.2f}"}
                            for k, v in revenue_data["revenue_by_product"].items()
                        ])
                        st.dataframe(prod_rev_df, use_container_width=True, hide_index=True)

            # Expiring Products Alert
            st.markdown("---")
            st.markdown("### ⚠️ Expiring Products Alert")

            col1, col2 = st.columns([2, 1])
            with col1:
                days_threshold = st.slider(
                    "Check products expiring within (days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=7
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                check_expiring_btn = st.button("🔍 Check Expiring Products", use_container_width=True)

            if check_expiring_btn or 'expiring_products_data' not in st.session_state:
                with st.spinner("Checking for expiring products..."):
                    expiring_data = st.session_state.edb_manager.get_expiring_products_alert(results, days_threshold)
                    st.session_state.expiring_products_data = expiring_data

            if 'expiring_products_data' in st.session_state:
                exp_data = st.session_state.expiring_products_data

                if exp_data.get("success") and exp_data.get("count", 0) > 0:
                    st.warning(f"⚠️ {exp_data['count']} product(s) expiring within {exp_data['days_threshold']} days!")

                    exp_df = pd.DataFrame(exp_data["expiring_products"])
                    st.dataframe(exp_df, use_container_width=True, hide_index=True)

                    # Download expiring products list
                    exp_csv = exp_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Expiring Products",
                        data=exp_csv,
                        file_name=f"expiring_products_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                elif exp_data.get("success"):
                    st.success(f"✅ No products expiring within {exp_data['days_threshold']} days")
                else:
                    st.error(f"Error checking expiring products: {exp_data.get('error')}")

            st.markdown("---")

            # Generate AI Insights button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Check if azure_client is available
                azure_available = (
                    hasattr(st.session_state.edb_manager, 'azure_client') and
                    st.session_state.edb_manager.azure_client is not None
                )

                generate_insights_btn = st.button(
                    "🤖 Generate AI Insights",
                    use_container_width=True,
                    type="primary",
                    disabled=not azure_available
                )

            if generate_insights_btn:
                with st.spinner("🤖 AI is analyzing your data..."):
                    insights = st.session_state.edb_manager.generate_ai_insights(results)
                    st.session_state.ai_insights = insights

            # Display AI Insights
            if 'ai_insights' in st.session_state and st.session_state.ai_insights.get("success"):
                insights = st.session_state.ai_insights

                st.markdown("### 🎯 AI-Generated Insights")
                st.markdown(insights["insights"])

                st.markdown("---")
                st.markdown("### 📈 Detailed Statistics")

                # Brand distribution
                if insights["statistics"]["brands"]:
                    st.markdown("#### Brand Distribution")
                    brand_df = pd.DataFrame([
                        {"Brand": k, "Count": v}
                        for k, v in insights["statistics"]["brands"].items()
                    ])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(brand_df, use_container_width=True, hide_index=True)
                    with col2:
                        st.bar_chart(brand_df.set_index("Brand"))

                # Lifecycle status distribution
                if insights["statistics"]["lifecycle_statuses"]:
                    st.markdown("#### Lifecycle Status Distribution")
                    status_df = pd.DataFrame([
                        {"Status": k, "Count": v}
                        for k, v in insights["statistics"]["lifecycle_statuses"].items()
                    ])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(status_df, use_container_width=True, hide_index=True)
                    with col2:
                        st.bar_chart(status_df.set_index("Status"))

                # Auto-renew statistics
                st.markdown("#### Auto-Renew Status")
                auto_renew_stats = insights["statistics"]["auto_renew_stats"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Auto-Renew Enabled", auto_renew_stats.get("Y", 0))
                with col2:
                    st.metric("Auto-Renew Disabled", auto_renew_stats.get("N", 0))

                # Top products
                if insights["statistics"]["top_products"]:
                    st.markdown("#### Top 10 Product Types")
                    top_prod_df = pd.DataFrame([
                        {"Product Code": k, "Count": v}
                        for k, v in insights["statistics"]["top_products"].items()
                    ])
                    st.dataframe(top_prod_df, use_container_width=True, hide_index=True)
                    st.bar_chart(top_prod_df.set_index("Product Code"))

                # Export insights
                st.markdown("---")
                insights_json = json.dumps(insights, indent=2, default=str)
                st.download_button(
                    label="📥 Download AI Insights",
                    data=insights_json,
                    file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            elif 'ai_insights' in st.session_state and not st.session_state.ai_insights.get("success"):
                st.error(f"Failed to generate insights: {st.session_state.ai_insights.get('error')}")

        else:
            st.info("ℹ️ Run a query first to generate AI insights")
            st.markdown("""
            AI Insights will provide:
            - **Key Patterns & Trends**: Identify important patterns in your data
            - **Actionable Recommendations**: Get specific actions to improve operations
            - **Predictive Analysis**: Forecast potential issues and opportunities
            - **Data Quality Assessment**: Identify inconsistencies and data issues
            - **Business Impact Analysis**: Understand how data affects your business
            """)

    # Tab 5: Query History
    with tab5:
        st.subheader("Query History")

        history = st.session_state.edb_manager.get_query_history()

        if history:
            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Queries", len(history))

            with col2:
                successful = sum(1 for q in history if q.get("success"))
                st.metric("Successful", successful)

            with col3:
                failed = len(history) - successful
                st.metric("Failed", failed)

            st.markdown("---")

            # Display history
            for idx, query in enumerate(reversed(history), 1):
                status_icon = "✅" if query.get("success") else "❌"
                timestamp = datetime.fromisoformat(query["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

                with st.expander(f"{status_icon} Query #{idx} - {timestamp}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Environment:** {query.get('environment', 'N/A')}")
                        st.write(f"**Timestamp:** {timestamp}")
                        st.write(f"**Status:** {'Success' if query.get('success') else 'Failed'}")

                    with col2:
                        if query.get("success"):
                            st.write(f"**SKUs:** {', '.join(query.get('sku_list', []))}")
                            st.write(f"**Brand:** {query.get('brand', 'N/A')}")
                            st.write(f"**Results:** {query.get('result_count', 0)} accounts")
                        else:
                            st.write(f"**Error:** {query.get('error', 'Unknown error')}")

            # Export history
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                json_history = json.dumps(history, indent=2, default=str)
                st.download_button(
                    label="📥 Download Query History",
                    data=json_history,
                    file_name=f"edb_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                    st.session_state.edb_manager.clear_query_history()
                    st.success("✅ Query history cleared!")
                    st.rerun()

        else:
            st.info("ℹ️ No queries have been executed yet. Start by running a query in the Query Accounts tab.")

    # Tab 6: Help & Tips
    with tab6:
        st.subheader("Help & Tips")

        st.markdown("""
        <div class="tips-section">
            <h4>💡 Tips for Using EDB Query Manager</h4>
            <ul>
                <li><strong>Environment Selection:</strong> Choose from 5 database environments: QAMain, Stage, JarvisQA1, JarvisQA2, or Production</li>
                <li><strong>Quick Search:</strong> Use Account ID, User Login Name, or Person Org ID for instant direct lookup (bypasses all filters)</li>
                <li><strong>Product Code Search:</strong> Use the search box to find specific SKUs. The search is case-insensitive and supports partial matches</li>
                <li><strong>Multi-Select SKUs:</strong> You can select multiple product codes to query accounts that have any of the selected products</li>
                <li><strong>Migration Source Filter:</strong> Filter accounts by their migration origin (e.g., BLUEHOST, HOSTGATOR, GODADDY)</li>
                <li><strong>Email Platform Filter:</strong> Filter accounts by email platform (OX Cloud, MS365, GWS, etc.) - requires EMAIL_DOMPTR table</li>
                <li><strong>Brand Filter:</strong> Filter accounts by brand code - NSI (Network Solutions), BH (Bluehost), or HG (HostGator)</li>
                <li><strong>Multi-Brand Selection:</strong> Enable in Advanced Filters to select multiple brands at once</li>
                <li><strong>Lifecycle Status:</strong> Choose from Active, Inactive, Active/Inactive, All, Expired, or To Be Expired (expiring within 10 days)</li>
                <li><strong>Date Range Filters:</strong> In Advanced Filters, filter by Created Date or Expiration Date ranges</li>
                <li><strong>Product Count Filter:</strong> In Advanced Filters, filter accounts by minimum/maximum product count</li>
                <li><strong>Subscription Terms:</strong> Filter by Annual, Monthly, One Time Shipped, One Time, or Four Week subscriptions</li>
                <li><strong>Builder Types:</strong> Filter by ImageCafe, LeapCafe, Matrix, Neo, Nexus, or Siteplus website builders</li>
                <li><strong>Sort Order:</strong> Sort results by newest, oldest, or randomize account creation date</li>
                <li><strong>Auto-Renew Filter:</strong> Filter products based on their auto-renewal setting</li>
                <li><strong>Filter Presets:</strong> Save your frequently used filter combinations and reload them later for quick access</li>
                <li><strong>Pagination:</strong> Navigate through large result sets with configurable results per page (5, 10, 20, 50)</li>
                <li><strong>Query Performance:</strong> View execution time for each query to monitor database performance</li>
                <li><strong>Health Scores:</strong> Each account displays a health score (0-100) based on product status, auto-renew settings, and other factors</li>
                <li><strong>Enhanced Data Display:</strong> View migration source, expiration date, subscription term, and builder type in results</li>
                <li><strong>AI Insights:</strong> Use Azure OpenAI to get intelligent analysis, predictions, and recommendations from your query results</li>
                <li><strong>Export Results:</strong> Download query results in JSON or CSV format for further analysis</li>
                <li><strong>Query History:</strong> All queries are logged for auditing. Use the Clear History button to reset</li>
                <li><strong>Result Limits:</strong> Use the "Max Accounts to Return" setting to control the size of result sets</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🆕 New Features Guide")

        with st.expander("🔎 Quick Search", expanded=False):
            st.markdown("""
            **Direct Account Lookup:**
            - Search by Account ID for instant access
            - Search by User Login Name (email)
            - Search by Person Org ID
            - Bypasses all other filters for fastest results
            - Uses OR logic (any match returns the account)
            
            **Example:** Enter Account ID "12345678" and click Query to see all products for that account.
            """)

        with st.expander("🌐 Migration Source Filter", expanded=False):
            st.markdown("""
            **Track Account Origins:**
            - Filter accounts by where they migrated from
            - Common sources: BLUEHOST, HOSTGATOR, GODADDY, NETWORK_SOLUTIONS
            - Search for specific migration sources
            - Multi-select to include multiple sources
            - Enable "Every Migration Source" to query all
            
            **Use Case:** Find all accounts migrated from GoDaddy with active hosting products.
            """)

        with st.expander("📧 Email Platform Filter", expanded=False):
            st.markdown("""
            **Filter by Email Service:**
            - OX Cloud, MS365, GWS (Google Workspace)
            - Inquent on prem, Pro Mail, Hostopia
            - HostedExchange, Titan, Roundcube
            - Requires EMAIL_DOMPTR table in database
            - Automatically joins when filter is active
            
            **Use Case:** Find all accounts using MS365 email with annual subscriptions.
            """)

        with st.expander("📅 Date Range Filters", expanded=False):
            st.markdown("""
            **Advanced Date Filtering:**
            - Created Date Range: Filter accounts by creation date
            - Expiration Date Range: Filter products by expiration
            - Enable checkbox to activate filter
            - Select From and To dates
            - Useful for finding accounts in specific time periods
            
            **Use Case:** Find accounts created in last 30 days that expire in next 60 days.
            """)

        with st.expander("🎯 Multi-Brand Selection", expanded=False):
            st.markdown("""
            **Query Multiple Brands:**
            - Select NSI, BH, and HG simultaneously
            - Enable in Advanced Filters section
            - Replaces single brand dropdown when enabled
            - Useful for cross-brand analysis
            
            **Use Case:** Compare active products across BH and HG brands.
            """)

        with st.expander("📊 Product Count Filter", expanded=False):
            st.markdown("""
            **Filter by Account Size:**
            - Minimum Products: Find accounts with at least X products
            - Maximum Products: Find accounts with at most Y products
            - Set 0 for no limit on maximum
            - Applied after main query for flexibility
            
            **Use Case:** Find high-value accounts with 5+ active products.
            """)

        with st.expander("💾 Filter Presets", expanded=False):
            st.markdown("""
            **Save & Reuse Filters:**
            1. Configure all desired filters
            2. Enter a preset name (e.g., "Active BH Domains")
            3. Click "Save Current Filters"
            4. Later, select from dropdown and click "Load"
            
            **Benefits:**
            - Quick access to common queries
            - Share filter configurations with team
            - Consistent reporting queries
            """)

        with st.expander("📄 Pagination", expanded=False):
            st.markdown("""
            **Navigate Large Result Sets:**
            - Choose results per page: 5, 10, 20, or 50
            - Previous/Next buttons for navigation
            - Current page indicator
            - Improves performance with large queries
            
            **Tip:** Use 10-20 results per page for optimal performance.
            """)

        with st.expander("⚡ Query Performance", expanded=False):
            st.markdown("""
            **Execution Time Monitoring:**
            - View query execution time in seconds
            - "Fast" indicator for queries under 2 seconds
            - "Slow" warning for queries over 2 seconds
            - Use to optimize filter combinations
            
            **Optimization Tips:**
            - Use specific SKUs instead of "Every SKU"
            - Apply lifecycle status filters
            - Limit max accounts returned
            """)

        st.markdown("---")

        st.markdown("### 🔧 Database Configuration")
        st.info("""
        Database connections are pre-configured using the settings in:
        `tests/configs/database_config_variables.py`
        
        Available environments:
        - **QAMain**: Production-like QA environment
        - **Stage**: Staging environment for pre-production testing
        - **JarvisQA1**: Jarvis QA environment 1
        - **JarvisQA2**: Jarvis QA environment 2
        - **Production**: Production environment (use with caution)
        """)

        st.markdown("---")

        st.markdown("### 🤖 AI Insights Configuration")
        st.info("""
        AI Insights powered by Azure OpenAI provide intelligent analysis of your query results.
        
        **Auto-Connection:** The Azure OpenAI client automatically connects when environment variables are properly configured.
        
        Required environment variables:
        ```bash
        export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
        export AZURE_OPENAI_API_KEY="your-api-key-here"
        export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
        ```
        
        After setting these variables, restart the application and Azure OpenAI will automatically connect.
        
        AI Insights Features:
        - **Pattern Detection**: Identify trends and anomalies in your data
        - **Predictive Analysis**: Forecast potential issues and opportunities
        - **Actionable Recommendations**: Get specific suggestions for improvement
        - **Data Quality Assessment**: Identify inconsistencies and data issues
        - **Business Impact Analysis**: Understand how data affects operations
        """)

        st.markdown("---")

        st.markdown("### 🏥 Account Health Scores")
        st.markdown("""
        Each account is assigned a health score (0-100) based on:
        - **Auto-Renew Status**: Accounts with products that have auto-renew disabled lose points
        - **Lifecycle Status**: Inactive or expired products reduce the score
        - **Product Diversity**: Single-product accounts get recommendations for upselling
        
        Health Score Ranges:
        - **90-100**: Excellent (Green) - Account is in great health
        - **75-89**: Good (Blue) - Account is healthy with minor improvements possible
        - **60-74**: Fair (Orange) - Account needs attention
        - **0-59**: Needs Attention (Red) - Immediate action required
        """)

        st.markdown("---")

        st.markdown("### 📚 Common Product Codes (SKUs)")
        st.markdown("""
        - **DOM_COM**: .COM Domain
        - **DOM_NET**: .NET Domain
        - **DOM_ORG**: .ORG Domain
        - **E_BASIC**: Basic Email Package
        - **E_PRO**: Professional Email Package
        - **HOST_SHARED**: Shared Hosting
        - **HOST_VPS**: VPS Hosting
        - **SSL_BASIC**: Basic SSL Certificate
        - **WB_DIY**: DIY Website Builder
        - **WB_ECOM**: eCommerce Website Builder
        """)

        st.markdown("---")

        st.markdown("### ⚠️ Troubleshooting")
        with st.expander("Connection Issues"):
            st.markdown("""
            If you encounter connection issues:
            1. Verify you have network access to the database server
            2. Check that oracledb is installed: `pip install oracledb`
            3. Ensure database credentials are correct in the configuration
            4. Check firewall rules and VPN connection if required
            5. Verify the database service is running
            """)

        with st.expander("No Results Found"):
            st.markdown("""
            If your query returns no results:
            1. Try broadening your search criteria
            2. Check if the selected SKUs exist in the database
            3. Verify lifecycle status filter isn't too restrictive
            4. Try querying a different environment
            5. Check the Query History for error messages
            """)

        with st.expander("Performance Tips"):
            st.markdown("""
            To optimize query performance:
            1. Limit the number of accounts returned using "Max Accounts to Return"
            2. Use specific SKU selections rather than querying all products
            3. Apply appropriate filters to narrow down results
            4. Close the application when not in use to free database connections
            5. Export large result sets for offline analysis
            """)

