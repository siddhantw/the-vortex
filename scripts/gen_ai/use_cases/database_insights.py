"""
Database Insights Module
Comprehensive database analysis and insights generation with AI-powered recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Database connection imports (install as needed)
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

try:
    import pyodbc
    MSSQL_AVAILABLE = True
except ImportError:
    MSSQL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseAnalyzer:
    """Advanced database analysis and insights generator"""
    
    def __init__(self):
        self.engine = None
        self.connection = None
        self.db_type = None
        self.schema_info = {}
        self.analysis_results = {}
        
    def connect(self, connection_params: Dict) -> Tuple[bool, str]:
        """Connect to database using provided parameters"""
        try:
            db_type = connection_params.get('db_type', '').lower()
            host = connection_params.get('host')
            port = connection_params.get('port')
            database = connection_params.get('database')
            username = connection_params.get('username')
            password = connection_params.get('password')
            
            # Build connection string based on database type
            if db_type == 'postgresql':
                if not POSTGRES_AVAILABLE:
                    return False, "PostgreSQL driver not available. Install psycopg2."
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                
            elif db_type == 'mysql':
                if not MYSQL_AVAILABLE:
                    return False, "MySQL driver not available. Install pymysql."
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                
            elif db_type == 'oracle':
                if not ORACLE_AVAILABLE:
                    return False, "Oracle driver not available. Install cx_Oracle."
                connection_string = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"
                
            elif db_type == 'sqlserver':
                if not MSSQL_AVAILABLE:
                    return False, "SQL Server driver not available. Install pyodbc."
                connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
                
            elif db_type == 'sqlite':
                connection_string = f"sqlite:///{database}"
                
            else:
                return False, f"Unsupported database type: {db_type}"
            
            # Create engine and test connection
            self.engine = create_engine(connection_string, echo=False)
            self.connection = self.engine.connect()
            self.db_type = db_type
            
            # Test with a simple query
            result = self.connection.execute(text("SELECT 1")).fetchone()
            
            return True, "Connected successfully"
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def analyze_schema(self) -> Dict[str, Any]:
        """Comprehensive schema analysis"""
        schema_analysis = {
            "tables": [],
            "total_tables": 0,
            "total_columns": 0,
            "indexes": [],
            "foreign_keys": [],
            "constraints": [],
            "views": [],
            "procedures": [],
            "triggers": [],
            "schema_health": {},
            "recommendations": []
        }
        
        try:
            inspector = inspect(self.engine)
            
            # Get all table names
            table_names = inspector.get_table_names()
            schema_analysis["total_tables"] = len(table_names)
            
            total_columns = 0
            
            for table_name in table_names:
                table_info = {
                    "name": table_name,
                    "columns": [],
                    "indexes": [],
                    "foreign_keys": [],
                    "primary_keys": [],
                    "row_count": 0,
                    "size_mb": 0,
                    "last_modified": None
                }
                
                # Get column information
                columns = inspector.get_columns(table_name)
                for column in columns:
                    table_info["columns"].append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get("nullable", True),
                        "default": column.get("default"),
                        "autoincrement": column.get("autoincrement", False)
                    })
                    total_columns += 1
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                for index in indexes:
                    index_info = {
                        "name": index["name"],
                        "columns": index["column_names"],
                        "unique": index.get("unique", False)
                    }
                    table_info["indexes"].append(index_info)
                    schema_analysis["indexes"].append({
                        "table": table_name,
                        "index": index_info
                    })
                
                # Get foreign keys
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    fk_info = {
                        "name": fk.get("name"),
                        "constrained_columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    }
                    table_info["foreign_keys"].append(fk_info)
                    schema_analysis["foreign_keys"].append({
                        "table": table_name,
                        "foreign_key": fk_info
                    })
                
                # Get primary keys
                primary_keys = inspector.get_pk_constraint(table_name)
                if primary_keys:
                    table_info["primary_keys"] = primary_keys.get("constrained_columns", [])
                
                # Get row count (with error handling)
                try:
                    result = self.connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                    table_info["row_count"] = result[0] if result else 0
                except Exception as e:
                    table_info["row_count"] = "Error"
                    logger.warning(f"Could not get row count for {table_name}: {e}")
                
                # Try to get table size (database-specific)
                try:
                    table_info["size_mb"] = self._get_table_size(table_name)
                except Exception as e:
                    table_info["size_mb"] = "Unknown"
                    logger.warning(f"Could not get size for {table_name}: {e}")
                
                schema_analysis["tables"].append(table_info)
            
            schema_analysis["total_columns"] = total_columns
            
            # Get views
            try:
                view_names = inspector.get_view_names()
                schema_analysis["views"] = [{"name": view} for view in view_names]
            except Exception as e:
                logger.warning(f"Could not get views: {e}")
            
            # Generate schema health analysis
            schema_analysis["schema_health"] = self._analyze_schema_health(schema_analysis)
            
            # Generate recommendations
            schema_analysis["recommendations"] = self._generate_schema_recommendations(schema_analysis)
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            schema_analysis["error"] = str(e)
        
        return schema_analysis
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Database performance analysis"""
        performance_analysis = {
            "slow_queries": [],
            "index_usage": {},
            "table_scans": [],
            "lock_analysis": {},
            "connection_stats": {},
            "buffer_pool_stats": {},
            "io_stats": {},
            "recommendations": [],
            "performance_score": 0
        }
        
        try:
            # Database-specific performance queries
            if self.db_type == 'postgresql':
                performance_analysis.update(self._analyze_postgres_performance())
            elif self.db_type == 'mysql':
                performance_analysis.update(self._analyze_mysql_performance())
            elif self.db_type == 'oracle':
                performance_analysis.update(self._analyze_oracle_performance())
            elif self.db_type == 'sqlserver':
                performance_analysis.update(self._analyze_sqlserver_performance())
            else:
                performance_analysis.update(self._analyze_generic_performance())
            
            # Calculate performance score
            performance_analysis["performance_score"] = self._calculate_performance_score(performance_analysis)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            performance_analysis["error"] = str(e)
        
        return performance_analysis
    
    def analyze_data_quality(self, sample_tables: List[str] = None) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        data_quality = {
            "table_analyses": [],
            "overall_quality_score": 0,
            "data_profiling": {},
            "anomalies": [],
            "completeness": {},
            "consistency": {},
            "validity": {},
            "recommendations": []
        }
        
        try:
            inspector = inspect(self.engine)
            table_names = sample_tables or inspector.get_table_names()[:10]  # Limit to 10 tables
            
            total_quality_score = 0
            analyzed_tables = 0
            
            for table_name in table_names:
                try:
                    table_analysis = self._analyze_table_data_quality(table_name)
                    data_quality["table_analyses"].append(table_analysis)
                    
                    if "quality_score" in table_analysis:
                        total_quality_score += table_analysis["quality_score"]
                        analyzed_tables += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
                    data_quality["table_analyses"].append({
                        "table_name": table_name,
                        "error": str(e)
                    })
            
            # Calculate overall quality score
            if analyzed_tables > 0:
                data_quality["overall_quality_score"] = total_quality_score / analyzed_tables
            
            # Generate recommendations
            data_quality["recommendations"] = self._generate_data_quality_recommendations(data_quality)
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            data_quality["error"] = str(e)
        
        return data_quality
    
    def analyze_security(self) -> Dict[str, Any]:
        """Database security analysis"""
        security_analysis = {
            "users": [],
            "privileges": [],
            "roles": [],
            "security_issues": [],
            "compliance_checks": {},
            "encryption_status": {},
            "audit_status": {},
            "security_score": 0,
            "recommendations": []
        }
        
        try:
            # Database-specific security analysis
            if self.db_type == 'postgresql':
                security_analysis.update(self._analyze_postgres_security())
            elif self.db_type == 'mysql':
                security_analysis.update(self._analyze_mysql_security())
            elif self.db_type == 'oracle':
                security_analysis.update(self._analyze_oracle_security())
            elif self.db_type == 'sqlserver':
                security_analysis.update(self._analyze_sqlserver_security())
            else:
                security_analysis.update(self._analyze_generic_security())
            
            # Calculate security score
            security_analysis["security_score"] = self._calculate_security_score(security_analysis)
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            security_analysis["error"] = str(e)
        
        return security_analysis
    
    def _get_table_size(self, table_name: str) -> float:
        """Get table size in MB (database-specific)"""
        try:
            if self.db_type == 'postgresql':
                query = f"SELECT pg_total_relation_size('{table_name}') / 1024.0 / 1024.0 AS size_mb"
            elif self.db_type == 'mysql':
                query = f"""
                SELECT ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
                FROM information_schema.TABLES 
                WHERE table_name = '{table_name}'
                """
            elif self.db_type == 'oracle':
                query = f"""
                SELECT ROUND(SUM(bytes) / 1024 / 1024, 2) AS size_mb
                FROM user_segments 
                WHERE segment_name = UPPER('{table_name}')
                """
            elif self.db_type == 'sqlserver':
                query = f"""
                SELECT CAST(SUM(reserved_page_count) * 8.0 / 1024 AS DECIMAL(10,2)) AS size_mb
                FROM sys.dm_db_partition_stats 
                WHERE object_id = OBJECT_ID('{table_name}')
                """
            else:
                return 0.0
            
            result = self.connection.execute(text(query)).fetchone()
            return float(result[0]) if result and result[0] else 0.0
            
        except Exception as e:
            logger.warning(f"Could not get size for table {table_name}: {e}")
            return 0.0
    
    def _analyze_schema_health(self, schema_info: Dict) -> Dict[str, Any]:
        """Analyze overall schema health"""
        health = {
            "missing_indexes": 0,
            "missing_foreign_keys": 0,
            "missing_primary_keys": 0,
            "large_tables": 0,
            "empty_tables": 0,
            "health_score": 100
        }
        
        try:
            for table in schema_info.get("tables", []):
                # Check for missing primary keys
                if not table.get("primary_keys"):
                    health["missing_primary_keys"] += 1
                    health["health_score"] -= 5
                
                # Check for empty tables
                if table.get("row_count") == 0:
                    health["empty_tables"] += 1
                    health["health_score"] -= 2
                
                # Check for large tables without indexes
                if (isinstance(table.get("row_count"), int) and table["row_count"] > 10000 
                    and len(table.get("indexes", [])) < 2):
                    health["missing_indexes"] += 1
                    health["health_score"] -= 10
                
                # Check for large tables
                if isinstance(table.get("size_mb"), (int, float)) and table["size_mb"] > 1000:
                    health["large_tables"] += 1
            
            # Ensure score doesn't go below 0
            health["health_score"] = max(0, health["health_score"])
            
        except Exception as e:
            logger.error(f"Schema health analysis failed: {e}")
        
        return health
    
    def _generate_schema_recommendations(self, schema_info: Dict) -> List[str]:
        """Generate schema improvement recommendations"""
        recommendations = []
        
        try:
            health = schema_info.get("schema_health", {})
            
            if health.get("missing_primary_keys", 0) > 0:
                recommendations.append(f"Add primary keys to {health['missing_primary_keys']} tables for better performance and data integrity")
            
            if health.get("missing_indexes", 0) > 0:
                recommendations.append(f"Consider adding indexes to {health['missing_indexes']} large tables to improve query performance")
            
            if health.get("empty_tables", 0) > 0:
                recommendations.append(f"Review {health['empty_tables']} empty tables - consider removing if not needed")
            
            if health.get("large_tables", 0) > 0:
                recommendations.append(f"Monitor {health['large_tables']} large tables for performance - consider partitioning")
            
            # Check for naming conventions
            table_names = [table["name"] for table in schema_info.get("tables", [])]
            inconsistent_naming = len(set([name.lower() for name in table_names])) != len(table_names)
            if inconsistent_naming:
                recommendations.append("Standardize table naming conventions for better maintainability")
            
        except Exception as e:
            logger.error(f"Failed to generate schema recommendations: {e}")
        
        return recommendations
    
    def _analyze_postgres_performance(self) -> Dict[str, Any]:
        """PostgreSQL-specific performance analysis"""
        performance = {}
        
        try:
            # Slow queries
            slow_query = """
            SELECT query, mean_time, calls, total_time
            FROM pg_stat_statements 
            WHERE mean_time > 100
            ORDER BY mean_time DESC 
            LIMIT 10
            """
            
            try:
                result = self.connection.execute(text(slow_query)).fetchall()
                performance["slow_queries"] = [
                    {
                        "query": row[0][:200] + "..." if len(row[0]) > 200 else row[0],
                        "mean_time": row[1],
                        "calls": row[2],
                        "total_time": row[3]
                    }
                    for row in result
                ]
            except Exception:
                performance["slow_queries"] = []
                performance["note"] = "pg_stat_statements extension not available"
            
        except Exception as e:
            performance["error"] = str(e)
        
        return performance
    
    def _analyze_mysql_performance(self) -> Dict[str, Any]:
        """MySQL-specific performance analysis"""
        performance = {}
        
        try:
            # Get MySQL status variables
            status_query = "SHOW STATUS WHERE Variable_name IN ('Slow_queries', 'Connections', 'Uptime')"
            result = self.connection.execute(text(status_query)).fetchall()
            
            performance["status_variables"] = {row[0]: row[1] for row in result}
            
        except Exception as e:
            performance["error"] = str(e)
        
        return performance
    
    def _analyze_oracle_performance(self) -> Dict[str, Any]:
        """Oracle-specific performance analysis"""
        performance = {}
        
        try:
            # Basic Oracle performance queries would go here
            performance["note"] = "Oracle performance analysis requires specific privileges"
            
        except Exception as e:
            performance["error"] = str(e)
        
        return performance
    
    def _analyze_sqlserver_performance(self) -> Dict[str, Any]:
        """SQL Server-specific performance analysis"""
        performance = {}
        
        try:
            # Basic SQL Server performance queries would go here
            performance["note"] = "SQL Server performance analysis requires specific permissions"
            
        except Exception as e:
            performance["error"] = str(e)
        
        return performance
    
    def _analyze_generic_performance(self) -> Dict[str, Any]:
        """Generic performance analysis for unsupported databases"""
        return {
            "note": f"Specific performance analysis not available for {self.db_type}",
            "recommendations": ["Monitor query execution times manually", "Review database-specific performance tools"]
        }
    
    def _analyze_table_data_quality(self, table_name: str) -> Dict[str, Any]:
        """Analyze data quality for a specific table"""
        analysis = {
            "table_name": table_name,
            "row_count": 0,
            "column_analysis": [],
            "null_percentages": {},
            "duplicates": 0,
            "quality_score": 100,
            "issues": []
        }
        
        try:
            # Get basic table info
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get row count
            result = self.connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
            analysis["row_count"] = result[0] if result else 0
            
            if analysis["row_count"] == 0:
                analysis["quality_score"] = 0
                analysis["issues"].append("Table is empty")
                return analysis
            
            # Analyze each column
            for column in columns[:10]:  # Limit to first 10 columns
                col_name = column["name"]
                col_type = str(column["type"])
                
                col_analysis = {
                    "name": col_name,
                    "type": col_type,
                    "null_count": 0,
                    "null_percentage": 0,
                    "unique_values": 0,
                    "issues": []
                }
                
                try:
                    # Get null count
                    null_query = f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL"
                    result = self.connection.execute(text(null_query)).fetchone()
                    col_analysis["null_count"] = result[0] if result else 0
                    col_analysis["null_percentage"] = (col_analysis["null_count"] / analysis["row_count"]) * 100
                    
                    # Get unique value count (sample for large tables)
                    if analysis["row_count"] > 10000:
                        unique_query = f"SELECT COUNT(DISTINCT {col_name}) FROM (SELECT {col_name} FROM {table_name} LIMIT 10000) AS sample"
                    else:
                        unique_query = f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}"
                    
                    result = self.connection.execute(text(unique_query)).fetchone()
                    col_analysis["unique_values"] = result[0] if result else 0
                    
                    # Data quality checks
                    if col_analysis["null_percentage"] > 50:
                        col_analysis["issues"].append("High null percentage")
                        analysis["quality_score"] -= 10
                    
                    if col_analysis["unique_values"] == 1 and analysis["row_count"] > 1:
                        col_analysis["issues"].append("All values are the same")
                        analysis["quality_score"] -= 5
                        
                except Exception as e:
                    col_analysis["issues"].append(f"Analysis failed: {str(e)}")
                
                analysis["column_analysis"].append(col_analysis)
            
            # Check for duplicates (simplified check)
            try:
                if len(columns) > 0:
                    first_few_cols = [col["name"] for col in columns[:3]]
                    cols_str = ", ".join(first_few_cols)
                    dup_query = f"""
                    SELECT COUNT(*) - COUNT(DISTINCT {cols_str}) as duplicates
                    FROM {table_name}
                    """
                    result = self.connection.execute(text(dup_query)).fetchone()
                    analysis["duplicates"] = result[0] if result and result[0] else 0
                    
                    if analysis["duplicates"] > 0:
                        analysis["issues"].append(f"{analysis['duplicates']} potential duplicate rows")
                        analysis["quality_score"] -= min(20, analysis["duplicates"] / analysis["row_count"] * 100)
                        
            except Exception as e:
                logger.warning(f"Duplicate check failed for {table_name}: {e}")
            
            # Ensure score doesn't go below 0
            analysis["quality_score"] = max(0, analysis["quality_score"])
            
        except Exception as e:
            analysis["issues"].append(f"Table analysis failed: {str(e)}")
            analysis["quality_score"] = 0
        
        return analysis
    
    def _generate_data_quality_recommendations(self, data_quality: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        try:
            overall_score = data_quality.get("overall_quality_score", 0)
            
            if overall_score < 70:
                recommendations.append("Overall data quality is below acceptable levels - immediate attention required")
            
            # Analyze table-specific issues
            high_null_tables = []
            empty_tables = []
            duplicate_tables = []
            
            for table_analysis in data_quality.get("table_analyses", []):
                if "error" in table_analysis:
                    continue
                    
                table_name = table_analysis.get("table_name", "")
                
                # Check for high null percentages
                for col in table_analysis.get("column_analysis", []):
                    if col.get("null_percentage", 0) > 50:
                        high_null_tables.append(f"{table_name}.{col['name']}")
                
                # Check for duplicates
                if table_analysis.get("duplicates", 0) > 0:
                    duplicate_tables.append(table_name)
                
                # Check for empty tables
                if table_analysis.get("row_count", 0) == 0:
                    empty_tables.append(table_name)
            
            if high_null_tables:
                recommendations.append(f"Address high null percentages in columns: {', '.join(high_null_tables[:5])}")
            
            if duplicate_tables:
                recommendations.append(f"Remove or investigate duplicate data in tables: {', '.join(duplicate_tables[:5])}")
            
            if empty_tables:
                recommendations.append(f"Review empty tables: {', '.join(empty_tables[:5])}")
            
            recommendations.append("Implement data validation rules and constraints")
            recommendations.append("Set up regular data quality monitoring")
            
        except Exception as e:
            logger.error(f"Failed to generate data quality recommendations: {e}")
        
        return recommendations
    
    def _analyze_postgres_security(self) -> Dict[str, Any]:
        """PostgreSQL-specific security analysis"""
        security = {}
        
        try:
            # Get user information
            users_query = "SELECT usename, usesuper, usecreatedb FROM pg_user"
            result = self.connection.execute(text(users_query)).fetchall()
            security["users"] = [
                {
                    "username": row[0],
                    "is_superuser": row[1],
                    "can_create_db": row[2]
                }
                for row in result
            ]
            
        except Exception as e:
            security["error"] = str(e)
        
        return security
    
    def _analyze_mysql_security(self) -> Dict[str, Any]:
        """MySQL-specific security analysis"""
        security = {}
        
        try:
            # Basic MySQL security checks would go here
            security["note"] = "MySQL security analysis requires specific privileges"
            
        except Exception as e:
            security["error"] = str(e)
        
        return security
    
    def _analyze_oracle_security(self) -> Dict[str, Any]:
        """Oracle-specific security analysis"""
        security = {}
        
        try:
            security["note"] = "Oracle security analysis requires DBA privileges"
            
        except Exception as e:
            security["error"] = str(e)
        
        return security
    
    def _analyze_sqlserver_security(self) -> Dict[str, Any]:
        """SQL Server-specific security analysis"""
        security = {}
        
        try:
            security["note"] = "SQL Server security analysis requires specific permissions"
            
        except Exception as e:
            security["error"] = str(e)
        
        return security
    
    def _analyze_generic_security(self) -> Dict[str, Any]:
        """Generic security analysis"""
        return {
            "note": f"Specific security analysis not available for {self.db_type}",
            "recommendations": ["Review database user permissions", "Ensure proper access controls", "Enable audit logging"]
        }
    
    def _calculate_performance_score(self, performance: Dict) -> int:
        """Calculate overall performance score"""
        score = 100
        
        # Deduct points based on performance issues
        slow_queries = len(performance.get("slow_queries", []))
        if slow_queries > 0:
            score -= min(30, slow_queries * 5)
        
        return max(0, score)
    
    def _calculate_security_score(self, security: Dict) -> int:
        """Calculate overall security score"""
        score = 100
        
        # Deduct points for security issues
        security_issues = len(security.get("security_issues", []))
        if security_issues > 0:
            score -= min(50, security_issues * 10)
        
        return max(0, score)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive database analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_type": self.db_type,
            "schema_analysis": {},
            "performance_analysis": {},
            "data_quality_analysis": {},
            "security_analysis": {},
            "overall_health": {},
            "executive_summary": {},
            "recommendations": []
        }
        
        try:
            # Run all analyses
            report["schema_analysis"] = self.analyze_schema()
            report["performance_analysis"] = self.analyze_performance()
            report["data_quality_analysis"] = self.analyze_data_quality()
            report["security_analysis"] = self.analyze_security()
            
            # Calculate overall health
            schema_score = report["schema_analysis"].get("schema_health", {}).get("health_score", 0)
            performance_score = report["performance_analysis"].get("performance_score", 0)
            quality_score = report["data_quality_analysis"].get("overall_quality_score", 0)
            security_score = report["security_analysis"].get("security_score", 0)
            
            overall_score = (schema_score + performance_score + quality_score + security_score) / 4
            
            report["overall_health"] = {
                "overall_score": overall_score,
                "schema_score": schema_score,
                "performance_score": performance_score,
                "quality_score": quality_score,
                "security_score": security_score,
                "grade": self._get_health_grade(overall_score)
            }
            
            # Generate executive summary
            report["executive_summary"] = self._generate_executive_summary(report)
            
            # Compile all recommendations
            all_recommendations = []
            all_recommendations.extend(report["schema_analysis"].get("recommendations", []))
            all_recommendations.extend(report["performance_analysis"].get("recommendations", []))
            all_recommendations.extend(report["data_quality_analysis"].get("recommendations", []))
            all_recommendations.extend(report["security_analysis"].get("recommendations", []))
            
            report["recommendations"] = list(set(all_recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            report["error"] = str(e)
        
        return report
    
    def _get_health_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_executive_summary(self, report: Dict) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        summary = {
            "database_overview": "",
            "key_findings": [],
            "critical_issues": [],
            "recommendations_summary": ""
        }
        
        try:
            # Database overview
            total_tables = report["schema_analysis"].get("total_tables", 0)
            total_columns = report["schema_analysis"].get("total_columns", 0)
            overall_score = report["overall_health"].get("overall_score", 0)
            
            summary["database_overview"] = f"""
            Database contains {total_tables} tables with {total_columns} total columns.
            Overall health score: {overall_score:.1f}/100 (Grade: {report["overall_health"].get("grade", "F")})
            """
            
            # Key findings
            if overall_score >= 80:
                summary["key_findings"].append("Database is in good health overall")
            elif overall_score >= 60:
                summary["key_findings"].append("Database has moderate issues requiring attention")
            else:
                summary["key_findings"].append("Database has significant issues requiring immediate attention")
            
            # Critical issues
            schema_health = report["schema_analysis"].get("schema_health", {})
            if schema_health.get("missing_primary_keys", 0) > 0:
                summary["critical_issues"].append(f"{schema_health['missing_primary_keys']} tables missing primary keys")
            
            quality_score = report["data_quality_analysis"].get("overall_quality_score", 100)
            if quality_score < 70:
                summary["critical_issues"].append("Data quality below acceptable levels")
            
            security_score = report["security_analysis"].get("security_score", 100)
            if security_score < 80:
                summary["critical_issues"].append("Security vulnerabilities detected")
            
            # Recommendations summary
            total_recommendations = len(report.get("recommendations", []))
            summary["recommendations_summary"] = f"Generated {total_recommendations} recommendations for improvement"
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            summary["error"] = str(e)
        
        return summary

def show_ui():
    """Display the Database Insights UI"""
    st.title("🗄️ Database Insights")
    st.markdown("Comprehensive database analysis and insights generation with AI-powered recommendations")
    
    # Check for required dependencies
    if not SQLALCHEMY_AVAILABLE:
        st.error("SQLAlchemy is required for database connections. Please install: pip install sqlalchemy")
        return
    
    # Database Connection Section
    st.header("🔌 Database Connection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "Oracle", "SQL Server", "SQLite"],
            help="Select your database type"
        )
        
        if db_type != "SQLite":
            host = st.text_input("Host", value="localhost", help="Database server hostname or IP")
            port = st.number_input(
                "Port", 
                value=5432 if db_type == "PostgreSQL" else 3306 if db_type == "MySQL" else 1521 if db_type == "Oracle" else 1433,
                help="Database server port"
            )
        else:
            host = port = None
        
    with col2:
        if db_type != "SQLite":
            database = st.text_input("Database Name", help="Name of the database to connect to")
            username = st.text_input("Username", help="Database username")
            password = st.text_input("Password", type="password", help="Database password")
        else:
            database = st.text_input("Database File Path", help="Path to SQLite database file")
            username = password = None
    
    # Test Connection
    if st.button("🔗 Test Connection"):
        if db_type != "SQLite" and not all([host, database, username, password]):
            st.error("Please fill in all connection details")
            return
        elif db_type == "SQLite" and not database:
            st.error("Please provide SQLite database file path")
            return
        
        connection_params = {
            "db_type": db_type.lower().replace(" ", ""),
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password
        }
        
        analyzer = DatabaseAnalyzer()
        
        with st.spinner("Testing connection..."):
            success, message = analyzer.connect(connection_params)
            
            if success:
                st.success(f"✅ {message}")
                st.session_state.db_analyzer = analyzer
                st.session_state.connection_params = connection_params
                st.session_state.connected = True
            else:
                st.error(f"❌ {message}")
                st.session_state.connected = False
    
    # Analysis Section
    if st.session_state.get('connected', False):
        st.header("📊 Database Analysis")
        
        analysis_options = st.multiselect(
            "Select Analysis Types",
            ["Schema Analysis", "Performance Analysis", "Data Quality", "Security Analysis", "Comprehensive Report"],
            default=["Schema Analysis", "Data Quality"],
            help="Choose which types of analysis to perform"
        )
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                sample_tables_only = st.checkbox(
                    "Sample Tables Only", 
                    value=True,
                    help="Analyze only a sample of tables for faster processing"
                )
                
                max_tables = st.number_input(
                    "Max Tables to Analyze",
                    1, 100, 10,
                    help="Maximum number of tables to analyze"
                )
                
            with col2:
                include_empty_tables = st.checkbox(
                    "Include Empty Tables",
                    value=False,
                    help="Include empty tables in analysis"
                )
                
                detailed_analysis = st.checkbox(
                    "Detailed Analysis",
                    value=True,
                    help="Perform detailed analysis (may take longer)"
                )
        
        # Start Analysis
        if st.button("🚀 Start Analysis", type="primary"):
            if not analysis_options:
                st.error("Please select at least one analysis type")
                return
            
            analyzer = st.session_state.db_analyzer
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            total_analyses = len(analysis_options)
            
            try:
                for i, analysis_type in enumerate(analysis_options):
                    status_text.text(f"Running {analysis_type}...")
                    progress_bar.progress((i + 1) / total_analyses)
                    
                    if analysis_type == "Schema Analysis":
                        results["schema"] = analyzer.analyze_schema()
                    elif analysis_type == "Performance Analysis":
                        results["performance"] = analyzer.analyze_performance()
                    elif analysis_type == "Data Quality":
                        sample_tables = None
                        if sample_tables_only:
                            # Get table names and limit them
                            try:
                                inspector = inspect(analyzer.engine)
                                all_tables = inspector.get_table_names()
                                sample_tables = all_tables[:max_tables]
                            except Exception:
                                sample_tables = None
                        results["data_quality"] = analyzer.analyze_data_quality(sample_tables)
                    elif analysis_type == "Security Analysis":
                        results["security"] = analyzer.analyze_security()
                    elif analysis_type == "Comprehensive Report":
                        results["comprehensive"] = analyzer.generate_comprehensive_report()
                
                progress_bar.progress(1.0)
                status_text.text("Analysis completed!")
                
                # Display Results
                st.header("📈 Analysis Results")
                
                # Executive Summary (if comprehensive report)
                if "comprehensive" in results:
                    comprehensive = results["comprehensive"]
                    
                    st.subheader("📋 Executive Summary")
                    
                    # Overall Health Metrics
                    health = comprehensive.get("overall_health", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        overall_score = health.get("overall_score", 0)
                        grade = health.get("grade", "F")
                        grade_color = "🟢" if grade in ["A", "B"] else "🟡" if grade == "C" else "🔴"
                        st.metric("Overall Health", f"{grade_color} {overall_score:.1f}/100 ({grade})")
                    
                    with col2:
                        schema_score = health.get("schema_score", 0)
                        st.metric("Schema Health", f"{schema_score:.1f}/100")
                    
                    with col3:
                        quality_score = health.get("quality_score", 0)
                        st.metric("Data Quality", f"{quality_score:.1f}/100")
                    
                    with col4:
                        security_score = health.get("security_score", 0)
                        st.metric("Security Score", f"{security_score:.1f}/100")
                    
                    # Executive Summary Text
                    exec_summary = comprehensive.get("executive_summary", {})
                    if exec_summary.get("database_overview"):
                        st.write("**Database Overview:**")
                        st.write(exec_summary["database_overview"])
                    
                    if exec_summary.get("critical_issues"):
                        st.write("**🚨 Critical Issues:**")
                        for issue in exec_summary["critical_issues"]:
                            st.write(f"- {issue}")
                
                # Schema Analysis Results
                if "schema" in results:
                    schema = results["schema"]
                    
                    with st.expander("🏗️ Schema Analysis", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Tables", schema.get("total_tables", 0))
                        with col2:
                            st.metric("Total Columns", schema.get("total_columns", 0))
                        with col3:
                            st.metric("Total Indexes", len(schema.get("indexes", [])))
                        
                        # Schema Health
                        health = schema.get("schema_health", {})
                        if health:
                            st.write("**Schema Health Issues:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"- Missing Primary Keys: {health.get('missing_primary_keys', 0)}")
                                st.write(f"- Missing Indexes: {health.get('missing_indexes', 0)}")
                            with col2:
                                st.write(f"- Empty Tables: {health.get('empty_tables', 0)}")
                                st.write(f"- Large Tables: {health.get('large_tables', 0)}")
                        
                        # Table Details
                        if schema.get("tables"):
                            st.write("**Table Overview:**")
                            
                            table_data = []
                            for table in schema["tables"][:20]:  # Show first 20 tables
                                table_data.append({
                                    "Table Name": table["name"],
                                    "Columns": len(table["columns"]),
                                    "Rows": table.get("row_count", "Unknown"),
                                    "Size (MB)": table.get("size_mb", "Unknown"),
                                    "Primary Key": "Yes" if table.get("primary_keys") else "No",
                                    "Indexes": len(table.get("indexes", []))
                                })
                            
                            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                
                # Data Quality Results
                if "data_quality" in results:
                    quality = results["data_quality"]
                    
                    with st.expander("📊 Data Quality Analysis", expanded=True):
                        overall_quality = quality.get("overall_quality_score", 0)
                        quality_color = "🟢" if overall_quality >= 80 else "🟡" if overall_quality >= 60 else "🔴"
                        st.metric("Overall Data Quality", f"{quality_color} {overall_quality:.1f}/100")
                        
                        # Table Quality Overview
                        if quality.get("table_analyses"):
                            st.write("**Table Quality Scores:**")
                            
                            quality_data = []
                            for table_analysis in quality["table_analyses"]:
                                if "error" not in table_analysis:
                                    quality_data.append({
                                        "Table": table_analysis["table_name"],
                                        "Quality Score": f"{table_analysis.get('quality_score', 0):.1f}",
                                        "Row Count": table_analysis.get("row_count", 0),
                                        "Issues": len(table_analysis.get("issues", [])),
                                        "Duplicates": table_analysis.get("duplicates", 0)
                                    })
                            
                            if quality_data:
                                st.dataframe(pd.DataFrame(quality_data), use_container_width=True)
                            
                            # Data Quality Issues
                            all_issues = []
                            for table_analysis in quality["table_analyses"]:
                                if "error" not in table_analysis:
                                    for issue in table_analysis.get("issues", []):
                                        all_issues.append(f"{table_analysis['table_name']}: {issue}")
                            
                            if all_issues:
                                st.write("**🚨 Data Quality Issues:**")
                                for issue in all_issues[:10]:  # Show first 10 issues
                                    st.write(f"- {issue}")
                
                # Performance Analysis Results
                if "performance" in results:
                    performance = results["performance"]
                    
                    with st.expander("⚡ Performance Analysis"):
                        perf_score = performance.get("performance_score", 0)
                        perf_color = "🟢" if perf_score >= 80 else "🟡" if perf_score >= 60 else "🔴"
                        st.metric("Performance Score", f"{perf_color} {perf_score}/100")
                        
                        # Slow Queries
                        slow_queries = performance.get("slow_queries", [])
                        if slow_queries:
                            st.write("**🐌 Slow Queries:**")
                            for i, query in enumerate(slow_queries[:5], 1):
                                with st.expander(f"Query {i} - Avg Time: {query.get('mean_time', 'N/A')}ms"):
                                    st.code(query.get("query", ""), language="sql")
                        else:
                            st.info("No slow queries detected or query analysis not available")
                
                # Security Analysis Results
                if "security" in results:
                    security = results["security"]
                    
                    with st.expander("🔒 Security Analysis"):
                        sec_score = security.get("security_score", 0)
                        sec_color = "🟢" if sec_score >= 80 else "🟡" if sec_score >= 60 else "🔴"
                        st.metric("Security Score", f"{sec_color} {sec_score}/100")
                        
                        # Users
                        users = security.get("users", [])
                        if users:
                            st.write("**Database Users:**")
                            user_data = []
                            for user in users:
                                user_data.append({
                                    "Username": user.get("username", ""),
                                    "Is Superuser": user.get("is_superuser", False),
                                    "Can Create DB": user.get("can_create_db", False)
                                })
                            
                            if user_data:
                                st.dataframe(pd.DataFrame(user_data), use_container_width=True)
                        
                        # Security Issues
                        sec_issues = security.get("security_issues", [])
                        if sec_issues:
                            st.write("**🚨 Security Issues:**")
                            for issue in sec_issues:
                                st.write(f"- {issue}")
                
                # Recommendations
                recommendations = []
                for result_type, result_data in results.items():
                    if isinstance(result_data, dict):
                        recommendations.extend(result_data.get("recommendations", []))
                
                if recommendations:
                    st.subheader("💡 Recommendations")
                    
                    # Categorize recommendations
                    priority_recommendations = []
                    general_recommendations = []
                    
                    for rec in set(recommendations):  # Remove duplicates
                        if any(keyword in rec.lower() for keyword in ["critical", "immediate", "security", "primary key"]):
                            priority_recommendations.append(rec)
                        else:
                            general_recommendations.append(rec)
                    
                    if priority_recommendations:
                        st.write("**🔥 High Priority:**")
                        for i, rec in enumerate(priority_recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    if general_recommendations:
                        st.write("**📋 General Improvements:**")
                        for i, rec in enumerate(general_recommendations, 1):
                            st.write(f"{i}. {rec}")
                
                # Export Results
                st.subheader("📤 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON Export
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_data,
                        file_name=f"db_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # CSV Export (summary data)
                    if "schema" in results and results["schema"].get("tables"):
                        summary_data = []
                        for table in results["schema"]["tables"]:
                            summary_data.append({
                                "Table": table["name"],
                                "Columns": len(table["columns"]),
                                "Rows": table.get("row_count", 0),
                                "Size MB": table.get("size_mb", 0),
                                "Has Primary Key": "Yes" if table.get("primary_keys") else "No",
                                "Index Count": len(table.get("indexes", []))
                            })
                        
                        csv_data = pd.DataFrame(summary_data).to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV Summary",
                            data=csv_data,
                            file_name=f"db_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Add notifications
                try:
                    import notifications
                    
                    # Determine notification type based on overall health
                    if "comprehensive" in results:
                        overall_score = results["comprehensive"].get("overall_health", {}).get("overall_score", 0)
                        critical_issues = results["comprehensive"].get("executive_summary", {}).get("critical_issues", [])
                        
                        if overall_score < 60:
                            notifications.add_notification(
                                module_name="database_insights",
                                status="error",
                                message=f"Database health score is critically low: {overall_score:.1f}/100",
                                details=f"Critical issues found: {', '.join(critical_issues[:3])}",
                                action_steps=[
                                    "Address critical issues immediately",
                                    "Review detailed analysis results",
                                    "Implement high priority recommendations"
                                ]
                            )
                        elif overall_score < 80:
                            notifications.add_notification(
                                module_name="database_insights",
                                status="warning",
                                message=f"Database health needs attention: {overall_score:.1f}/100",
                                details=f"Issues found: {len(critical_issues)} critical issues detected",
                                action_steps=[
                                    "Review analysis results",
                                    "Plan improvements based on recommendations",
                                    "Monitor database health regularly"
                                ]
                            )
                        else:
                            notifications.add_notification(
                                module_name="database_insights",
                                status="success",
                                message=f"Database is in good health: {overall_score:.1f}/100",
                                details="Analysis completed successfully with minimal issues",
                                action_steps=[
                                    "Continue monitoring database health",
                                    "Implement recommended optimizations",
                                    "Schedule regular health checks"
                                ]
                            )
                    else:
                        notifications.add_notification(
                            module_name="database_insights",
                            status="success",
                            message="Database analysis completed successfully",
                            details=f"Analyzed {len(analysis_options)} aspects of the database",
                            action_steps=[
                                "Review analysis results",
                                "Export reports for stakeholders",
                                "Implement recommendations"
                            ]
                        )
                        
                except ImportError:
                    pass
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Database analysis failed: {e}")
        
        # Disconnect button
        if st.button("🔌 Disconnect"):
            if hasattr(st.session_state, 'db_analyzer'):
                st.session_state.db_analyzer.disconnect()
            st.session_state.connected = False
            st.success("Disconnected from database")
    
    # Help Section
    with st.expander("ℹ️ Help & Setup Guide"):
        st.markdown("""
        ### Database Connection Setup
        
        **Supported Databases:**
        - **PostgreSQL**: Requires `psycopg2` package
        - **MySQL**: Requires `pymysql` package  
        - **Oracle**: Requires `cx_Oracle` package
        - **SQL Server**: Requires `pyodbc` package
        - **SQLite**: Built-in support
        
        **Installation Commands:**
        ```bash
        # PostgreSQL
        pip install psycopg2-binary
        
        # MySQL
        pip install pymysql
        
        # Oracle (requires Oracle Client)
        pip install cx_Oracle
        
        # SQL Server
        pip install pyodbc
        ```
        
        **Analysis Types:**
        
        **Schema Analysis:**
        - Table and column inventory
        - Index analysis
        - Foreign key relationships
        - Primary key validation
        - Schema health scoring
        
        **Performance Analysis:**
        - Slow query identification
        - Index usage statistics
        - Database performance metrics
        - Connection and I/O analysis
        
        **Data Quality Analysis:**
        - Null value analysis
        - Duplicate detection
        - Data type validation
        - Completeness assessment
        - Quality scoring
        
        **Security Analysis:**
        - User and role audit
        - Permission analysis
        - Security configuration review
        - Compliance checking
        
        **Health Scores:**
        - **90-100**: Excellent (Grade A)
        - **80-89**: Good (Grade B)
        - **70-79**: Fair (Grade C)
        - **60-69**: Poor (Grade D)
        - **Below 60**: Critical (Grade F)
        
        **Best Practices:**
        - Run analysis during off-peak hours
        - Start with sample tables for large databases
        - Review recommendations regularly
        - Monitor trends over time
        """)

if __name__ == "__main__":
    show_ui()