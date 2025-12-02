"""
Agentic AI Workflow for Dynamic Test Generation

This module implements an intelligent agentic workflow that orchestrates
the entire test generation process with smart decision-making and optimization.
Enhanced with insights from Dynamic Test Case Generation module.
"""

import logging
import asyncio
import json
import time
import sys
import os
import re
import tempfile
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse
import requests
import base64

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings from dynamic test generation insights
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*torch.*_path.*")

# Enhanced RAG integration
RAG_AVAILABLE = False
try:
    from rag.rag_handler import RAGHandler
    from rag.rag_config import RAGConfig
    RAG_AVAILABLE = True
    logger.info("RAG module imported successfully for agentic workflow")
except ImportError:
    try:
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        rag_path = os.path.join(parent_dir, 'rag')
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)
        from rag_handler import RAGHandler
        from rag_config import RAGConfig
        RAG_AVAILABLE = True
        logger.info("RAG module imported successfully (fallback method)")
    except ImportError as e:
        logger.warning(f"RAG module not available: {e}. RAG features will be disabled.")
        RAG_AVAILABLE = False

# Azure OpenAI client integration
AZURE_CLIENT_AVAILABLE = False
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_CLIENT_AVAILABLE = True
    logger.info("Azure OpenAI client imported successfully for agentic workflow")
except ImportError:
    logger.warning("Azure OpenAI client not available. Using fallback configuration.")
    AZURE_CLIENT_AVAILABLE = False

# Notification system integration
NOTIFICATIONS_AVAILABLE = False
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    logger.warning("Notifications module not available. Notification features will be disabled.")
    NOTIFICATIONS_AVAILABLE = False

class TaskStatus(Enum):
    """Status of workflow tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class AgentType(Enum):
    """Types of AI agents in the workflow"""
    ANALYZER = "analyzer"
    PLANNER = "planner"
    GENERATOR = "generator"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"

@dataclass
class WorkflowTask:
    """Represents a task in the workflow"""
    id: str
    name: str
    agent_type: AgentType
    status: TaskStatus
    priority: int
    dependencies: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None

@dataclass
class WorkflowResult:
    """Result of the entire workflow"""
    success: bool
    test_plan: Optional[str] = None
    manual_test_cases: Optional[List[Dict[str, Any]]] = None
    automated_test_cases: Optional[List[Dict[str, Any]]] = None
    robot_scripts: Optional[str] = None
    recommendations: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class BaseAgent:
    """Base class for AI agents"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.config = config or {}
        self.name = f"{agent_type.value}_agent"
        
    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute the agent's task"""
        raise NotImplementedError
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate task inputs"""
        return True

class RequirementAnalyzerAgent(BaseAgent):
    """Agent for analyzing requirements with brand context"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.ANALYZER, config)
        
    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """Analyze requirements with enhanced context awareness"""
        try:
            logger.info(f"[{self.name}] Starting requirement analysis...")
            
            analysis = task.inputs.get("analysis")
            brand_context = task.inputs.get("brand_context", "Auto-detect")
            
            if not analysis:
                raise ValueError("No analysis data provided")
            
            # Enhanced analysis with brand awareness
            enhanced_analysis = await self._enhance_with_brand_context(analysis, brand_context)
            
            # Extract business context
            business_context = await self._extract_business_context(enhanced_analysis)
            
            # Identify test priorities
            test_priorities = await self._identify_test_priorities(enhanced_analysis)
            
            # Generate test strategy recommendations
            strategy_recommendations = await self._generate_strategy_recommendations(
                enhanced_analysis, brand_context
            )
            
            return {
                "enhanced_analysis": enhanced_analysis,
                "business_context": business_context,
                "test_priorities": test_priorities,
                "strategy_recommendations": strategy_recommendations,
                "confidence_score": self._calculate_confidence_score(enhanced_analysis)
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            raise
    
    async def _enhance_with_brand_context(self, analysis: Dict[str, Any], brand_context: str) -> Dict[str, Any]:
        """Enhance analysis with brand-specific context"""
        
        brand_mappings = {
            "bluehost": {
                "domain": "web_hosting",
                "key_features": ["wordpress", "shared_hosting", "vps", "cpanel", "ssl"],
                "test_focus": ["hosting_performance", "uptime", "wordpress_integration", "security"]
            },
            "domain.com": {
                "domain": "domain_registration",
                "key_features": ["domain_search", "registration", "transfer", "dns", "privacy"],
                "test_focus": ["domain_availability", "registration_flow", "dns_management", "whois"]
            },
            "hostgator": {
                "domain": "web_hosting",
                "key_features": ["shared_hosting", "wordpress", "website_builder", "reseller"],
                "test_focus": ["hosting_features", "website_builder", "migration", "support"]
            },
            "network_solutions": {
                "domain": "domain_and_hosting",
                "key_features": ["domains", "hosting", "email", "marketing", "ecommerce"],
                "test_focus": ["domain_services", "email_hosting", "marketing_tools", "ecommerce"]
            },
            "register.com": {
                "domain": "domain_services",
                "key_features": ["domain_registration", "hosting", "email", "marketing"],
                "test_focus": ["domain_registration", "email_services", "marketing_features"]
            },
            "web.com": {
                "domain": "website_building",
                "key_features": ["website_builder", "ecommerce", "marketing", "lead_generation"],
                "test_focus": ["website_creation", "ecommerce_features", "marketing_automation"]
            }
        }
        
        enhanced = analysis.copy()
        
        if brand_context.lower() in brand_mappings:
            brand_info = brand_mappings[brand_context.lower()]
            enhanced["brand_context"] = brand_info
            enhanced["domain_focus"] = brand_info["domain"]
            enhanced["key_features"] = brand_info["key_features"]
            enhanced["recommended_test_focus"] = brand_info["test_focus"]
        
        return enhanced
    
    async def _extract_business_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business context from requirements"""
        
        business_keywords = {
            "user_management": ["user", "account", "profile", "login", "authentication"],
            "payment_processing": ["payment", "billing", "subscription", "credit card", "checkout"],
            "content_management": ["content", "cms", "blog", "page", "article"],
            "ecommerce": ["product", "cart", "order", "inventory", "catalog"],
            "communication": ["email", "notification", "message", "alert", "contact"],
            "analytics": ["analytics", "reporting", "dashboard", "metrics", "statistics"],
            "security": ["security", "encryption", "ssl", "privacy", "protection"],
            "integration": ["api", "integration", "third-party", "webhook", "sync"]
        }
        
        # Handle both string and dict formats for requirements
        requirements = analysis.get("requirements", [])
        requirements_text_parts = []
        for req in requirements:
            if isinstance(req, str):
                requirements_text_parts.append(req)
            elif isinstance(req, dict):
                requirements_text_parts.append(req.get("text", ""))
            else:
                requirements_text_parts.append(str(req))
        
        requirements_text = " ".join(requirements_text_parts).lower()
        
        identified_contexts = {}
        for context, keywords in business_keywords.items():
            score = sum(1 for keyword in keywords if keyword in requirements_text)
            if score > 0:
                identified_contexts[context] = {
                    "score": score,
                    "keywords_found": [kw for kw in keywords if kw in requirements_text]
                }
        
        return {
            "primary_contexts": identified_contexts,
            "complexity_indicators": self._identify_complexity_indicators(requirements_text),
            "integration_points": self._identify_integration_points(requirements_text)
        }
    
    async def _identify_test_priorities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify test priorities based on analysis"""
        
        priorities = []
        requirements = analysis.get("requirements", [])
        
        # High priority: Security and authentication
        security_reqs = []
        for req in requirements:
            req_text = req if isinstance(req, str) else req.get("text", "")
            if any(keyword in req_text.lower() for keyword in ["security", "authentication", "login", "password"]):
                security_reqs.append(req)
        if security_reqs:
            priorities.append({
                "category": "Security & Authentication",
                "priority": "HIGH",
                "reason": "Critical for user protection and data security",
                "requirements": len(security_reqs),
                "test_types": ["functional", "security", "negative"]
            })
        
        # High priority: Payment and financial
        payment_reqs = []
        for req in requirements:
            req_text = req if isinstance(req, str) else req.get("text", "")
            if any(keyword in req_text.lower() for keyword in ["payment", "billing", "subscription", "credit card"]):
                payment_reqs.append(req)
        if payment_reqs:
            priorities.append({
                "category": "Payment Processing",
                "priority": "HIGH",
                "reason": "Financial transactions require thorough testing",
                "requirements": len(payment_reqs),
                "test_types": ["functional", "security", "integration", "negative"]
            })
        
        # Medium priority: Core business features
        core_features = analysis.get("key_features", [])
        if core_features:
            feature_reqs = []
            for req in requirements:
                req_text = req if isinstance(req, str) else req.get("text", "")
                if any(feature in req_text.lower() for feature in core_features):
                    feature_reqs.append(req)
                    
            priorities.append({
                "category": "Core Business Features",
                "priority": "MEDIUM",
                "reason": "Essential for business operations",
                "requirements": len(feature_reqs),
                "test_types": ["functional", "integration", "performance"]
            })
        
        return priorities
    
    async def _generate_strategy_recommendations(self, analysis: Dict[str, Any], 
                                               brand_context: str) -> List[str]:
        """Generate test strategy recommendations"""
        
        recommendations = []
        
        # Brand-specific recommendations
        if brand_context.lower() in ["bluehost", "hostgator"]:
            recommendations.extend([
                "Focus on hosting performance and uptime testing",
                "Include WordPress-specific integration tests",
                "Test cPanel functionality extensively",
                "Validate SSL certificate management"
            ])
        elif brand_context.lower() in ["domain.com", "register.com"]:
            recommendations.extend([
                "Prioritize domain search and availability testing",
                "Test domain registration and transfer flows",
                "Validate DNS management features",
                "Include WHOIS privacy testing"
            ])
        elif brand_context.lower() == "web.com":
            recommendations.extend([
                "Focus on website builder functionality",
                "Test drag-and-drop interface thoroughly",
                "Validate template customization features",
                "Include mobile responsiveness testing"
            ])
        
        # General recommendations based on complexity
        complexity = analysis.get("statistics", {}).get("avg_complexity", 0)
        if complexity > 0.7:
            recommendations.append("Consider phased testing approach due to high complexity")
            recommendations.append("Implement comprehensive integration testing")
        
        # Automation recommendations
        requirements_count = len(analysis.get("requirements", []))
        if requirements_count > 20:
            recommendations.append("Prioritize test automation for regression testing")
            recommendations.append("Implement data-driven testing for scalability")
        
        return recommendations
    
    def _calculate_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        
        base_score = 0.5
        
        # Boost for number of requirements
        req_count = len(analysis.get("requirements", []))
        req_score = min(req_count / 20, 0.3)  # Max 0.3 boost
        
        # Boost for brand context
        brand_score = 0.2 if analysis.get("brand_context") else 0
        
        # Boost for clear test scenarios
        scenarios = analysis.get("test_scenarios", [])
        scenario_score = min(len(scenarios) / 10, 0.2)  # Max 0.2 boost
        
        return min(base_score + req_score + brand_score + scenario_score, 1.0)
    
    def _identify_complexity_indicators(self, text: str) -> List[str]:
        """Identify complexity indicators in requirements"""
        
        complexity_indicators = []
        
        complex_keywords = {
            "integration": ["api", "integration", "third-party", "sync", "webhook"],
            "workflow": ["workflow", "process", "step", "approval", "routing"],
            "real-time": ["real-time", "live", "instant", "immediate", "streaming"],
            "scalability": ["scale", "performance", "load", "concurrent", "throughput"],
            "customization": ["custom", "configurable", "flexible", "personalize"]
        }
        
        for category, keywords in complex_keywords.items():
            if any(keyword in text for keyword in keywords):
                complexity_indicators.append(category)
        
        return complexity_indicators
    
    def _identify_integration_points(self, text: str) -> List[str]:
        """Identify integration points from requirements"""
        
        integration_points = []
        
        integration_keywords = [
            "api", "integration", "third-party", "external", "webhook",
            "sync", "import", "export", "connection", "service"
        ]
        
        # This is a simplified approach - in reality, you'd use NLP
        # to better extract integration points
        if any(keyword in text for keyword in integration_keywords):
            integration_points = ["External APIs", "Third-party Services", "Data Sync"]
        
        return integration_points

class TestPlannerAgent(BaseAgent):
    """Agent for creating comprehensive test plans"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.PLANNER, config)
        
    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """Create a comprehensive test plan"""
        try:
            logger.info(f"[{self.name}] Creating test plan...")
            
            enhanced_analysis = task.inputs.get("enhanced_analysis")
            test_priorities = task.inputs.get("test_priorities", [])
            strategy_recommendations = task.inputs.get("strategy_recommendations", [])
            test_config = task.inputs.get("test_config", {})
            
            # Create test plan structure
            test_plan = await self._create_test_plan_structure(
                enhanced_analysis, test_priorities, strategy_recommendations, test_config
            )
            
            # Generate test execution timeline
            timeline = await self._generate_execution_timeline(test_plan, test_config)
            
            # Estimate effort and resources
            effort_estimation = await self._estimate_effort(test_plan)
            
            return {
                "test_plan": test_plan,
                "execution_timeline": timeline,
                "effort_estimation": effort_estimation,
                "risk_assessment": await self._assess_risks(enhanced_analysis)
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            raise
    
    async def _create_test_plan_structure(self, analysis: Dict[str, Any], 
                                        priorities: List[Dict[str, Any]],
                                        recommendations: List[str],
                                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive test plan structure"""
        
        return {
            "project_overview": {
                "name": f"Test Plan - {analysis.get('brand_context', 'Application') if isinstance(analysis.get('brand_context'), str) else analysis.get('brand_context', {}).get('domain', 'Application')}",
                "brand_context": analysis.get("brand_context", {}),
                "scope": config.get("scope", "Both"),
                "test_type": config.get("test_type", "All"),
                "complexity_level": config.get("test_complexity", "Intermediate")
            },
            "test_objectives": await self._define_test_objectives(analysis, priorities),
            "test_strategy": await self._define_test_strategy(recommendations, config),
            "test_scope": await self._define_test_scope(analysis, config),
            "test_priorities": priorities,
            "test_environments": await self._define_test_environments(analysis),
            "entry_exit_criteria": await self._define_entry_exit_criteria(priorities),
            "deliverables": await self._define_deliverables(config),
            "assumptions_dependencies": await self._identify_assumptions_dependencies(analysis)
        }
    
    async def _define_test_objectives(self, analysis: Dict[str, Any], 
                                    priorities: List[Dict[str, Any]]) -> List[str]:
        """Define test objectives based on analysis"""
        
        objectives = [
            "Verify that all functional requirements are implemented correctly",
            "Ensure the application meets quality standards and user expectations",
            "Identify defects early in the development cycle",
            "Validate system performance under expected load conditions"
        ]
        
        # Add brand-specific objectives
        brand_context = analysis.get("brand_context", {})
        if brand_context:
            # Handle both string and dict types for brand_context
            if isinstance(brand_context, str):
                domain = brand_context.lower()
            else:
                domain = brand_context.get("domain", "").lower()
                
            if "hosting" in domain:
                objectives.append("Validate hosting service reliability and performance")
                objectives.append("Ensure website deployment and management features work correctly")
            elif "domain" in domain:
                objectives.append("Verify domain registration and management processes")
                objectives.append("Validate DNS configuration and propagation")
            elif "website" in domain:
                objectives.append("Ensure website building tools are intuitive and functional")
                objectives.append("Validate template customization and publishing features")
        
        # Add priority-based objectives
        for priority in priorities:
            if priority["category"] == "Security & Authentication":
                objectives.append("Validate security measures and authentication mechanisms")
            elif priority["category"] == "Payment Processing":
                objectives.append("Ensure payment processing is secure and reliable")
        
        return objectives
    
    async def _define_test_strategy(self, recommendations: List[str], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Define test strategy"""
        
        strategy = {
            "approach": config.get("test_type", "All"),
            "automation_preference": config.get("automation_preference", "Balanced"),
            "test_levels": [],
            "test_types": [],
            "tools_frameworks": [],
            "recommendations": recommendations
        }
        
        # Define test levels based on test type
        test_type = config.get("test_type", "All")
        if test_type in ["Component", "All"]:
            strategy["test_levels"].append("Unit Testing")
        if test_type in ["Integration", "All"]:
            strategy["test_levels"].append("Integration Testing")
        if test_type in ["Acceptance", "All"]:
            strategy["test_levels"].append("User Acceptance Testing")
        
        # Define test types based on scope
        scope = config.get("scope", "Both")
        if scope in ["Functional", "Both"]:
            strategy["test_types"].extend(["Functional Testing", "UI Testing", "API Testing"])
        if scope in ["Non-Functional", "Both"]:
            strategy["test_types"].extend(["Performance Testing", "Security Testing", "Usability Testing"])
        
        # Add accessibility if requested
        if config.get("include_accessibility"):
            strategy["test_types"].append("Accessibility Testing")
        
        # Define tools and frameworks
        strategy["tools_frameworks"] = [
            "Robot Framework for test automation",
            "Selenium WebDriver for UI testing",
            "Postman/REST Assured for API testing"
        ]
        
        if "Performance Testing" in strategy["test_types"]:
            strategy["tools_frameworks"].append("JMeter/Locust for performance testing")
        
        return strategy
    
    async def _define_test_scope(self, analysis: Dict[str, Any], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Define test scope"""
        
        components = config.get("components", "").split(",") if config.get("components") else []
        components = [c.strip() for c in components if c.strip()]
        
        return {
            "in_scope": {
                "functional_areas": components or analysis.get("key_features", []),
                "test_scenarios": analysis.get("test_scenarios", []),
                "business_flows": analysis.get("functional_flows", []),
                "integration_points": analysis.get("integration_points", [])
            },
            "out_of_scope": [
                "Third-party service internal testing",
                "Infrastructure testing (unless specified)",
                "Load testing beyond specified limits"
            ],
            "assumptions": [
                "Test environment will be available and stable",
                "Test data will be provided or created as needed",
                "Application features are implemented as per requirements"
            ]
        }
    
    async def _define_test_environments(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Define test environments"""
        
        environments = [
            {
                "name": "Development",
                "purpose": "Component and early integration testing",
                "url": "dev.example.com",
                "data": "Development test data"
            },
            {
                "name": "QA/Testing",
                "purpose": "Comprehensive functional and integration testing",
                "url": "qa.example.com",
                "data": "QA test data with realistic scenarios"
            },
            {
                "name": "Staging",
                "purpose": "Pre-production validation and user acceptance testing",
                "url": "staging.example.com",
                "data": "Production-like data (anonymized)"
            }
        ]
        
        # Add performance environment if needed
        if any("performance" in req.get("text", "").lower() 
               for req in analysis.get("requirements", [])):
            environments.append({
                "name": "Performance",
                "purpose": "Load and performance testing",
                "url": "perf.example.com",
                "data": "Large dataset for performance validation"
            })
        
        return environments
    
    async def _define_entry_exit_criteria(self, priorities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Define entry and exit criteria"""
        
        entry_criteria = [
            "Requirements have been reviewed and finalized",
            "Test environment is set up and accessible",
            "Application build is deployed and functional",
            "Test data is prepared and validated"
        ]
        
        exit_criteria = [
            "All planned test cases have been executed",
            "Critical and high priority defects are resolved",
            "Test coverage meets defined criteria (minimum 80%)",
            "Performance benchmarks are met"
        ]
        
        # Add priority-specific criteria
        for priority in priorities:
            if priority["priority"] == "HIGH":
                entry_criteria.append(f"{priority['category']} requirements are clearly defined")
                exit_criteria.append(f"{priority['category']} tests pass with 100% success rate")
        
        return {
            "entry_criteria": entry_criteria,
            "exit_criteria": exit_criteria
        }
    
    async def _define_deliverables(self, config: Dict[str, Any]) -> List[str]:
        """Define test deliverables"""
        
        deliverables = [
            "Test Plan Document",
            "Test Case Specifications",
            "Test Execution Reports",
            "Defect Reports and Status",
            "Test Coverage Reports"
        ]
        
        if config.get("automation_preference") in ["Prefer Automated", "Balanced"]:
            deliverables.extend([
                "Automated Test Scripts (Robot Framework)",
                "Test Automation Framework Documentation"
            ])
        
        if config.get("include_performance"):
            deliverables.append("Performance Test Results and Analysis")
        
        if config.get("include_accessibility"):
            deliverables.append("Accessibility Compliance Report")
        
        return deliverables
    
    async def _identify_assumptions_dependencies(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify assumptions and dependencies"""
        
        assumptions = [
            "Test environment mirrors production environment",
            "All required integrations are available for testing",
            "Test data can be created or provided as needed",
            "Development team will provide timely defect fixes"
        ]
        
        dependencies = [
            "Availability of test environment",
            "Completion of application features",
            "Access to required test tools and licenses",
            "Availability of subject matter experts for clarifications"
        ]
        
        # Add analysis-specific dependencies
        integration_points = analysis.get("integration_points", [])
        if integration_points:
            dependencies.append("Availability of external systems for integration testing")
        
        return {
            "assumptions": assumptions,
            "dependencies": dependencies
        }
    
    async def _generate_execution_timeline(self, test_plan: Dict[str, Any], 
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test execution timeline"""
        
        phases = []
        
        # Planning phase
        phases.append({
            "phase": "Test Planning",
            "duration": "3-5 days",
            "activities": [
                "Finalize test plan",
                "Create test cases",
                "Set up test environment",
                "Prepare test data"
            ]
        })
        
        # Execution phases based on test type
        test_type = config.get("test_type", "All")
        
        if test_type in ["Component", "All"]:
            phases.append({
                "phase": "Component Testing",
                "duration": "5-7 days",
                "activities": [
                    "Execute unit tests",
                    "Validate individual components",
                    "Report component-level defects"
                ]
            })
        
        if test_type in ["Integration", "All"]:
            phases.append({
                "phase": "Integration Testing",
                "duration": "7-10 days",
                "activities": [
                    "Execute integration test cases",
                    "Validate end-to-end workflows",
                    "Test API integrations",
                    "Verify data flow between components"
                ]
            })
        
        if test_type in ["Acceptance", "All"]:
            phases.append({
                "phase": "User Acceptance Testing",
                "duration": "5-7 days",
                "activities": [
                    "Execute user scenarios",
                    "Validate business workflows",
                    "Gather user feedback",
                    "Confirm acceptance criteria"
                ]
            })
        
        # Closure phase
        phases.append({
            "phase": "Test Closure",
            "duration": "2-3 days",
            "activities": [
                "Complete test execution",
                "Generate final reports",
                "Document lessons learned",
                "Archive test artifacts"
            ]
        })
        
        return {
            "phases": phases,
            "total_duration": "15-25 business days",
            "critical_path": "Environment setup → Component testing → Integration testing → UAT"
        }
    
    async def _estimate_effort(self, test_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate testing effort"""
        
        # This is a simplified estimation model
        # In practice, you'd use more sophisticated methods
        
        base_effort = 40  # hours
        
        # Adjust based on complexity
        complexity = test_plan.get("project_overview", {}).get("complexity_level", "Intermediate")
        complexity_multiplier = {
            "Basic": 0.7,
            "Intermediate": 1.0,
            "Advanced": 1.5,
            "Comprehensive": 2.0
        }.get(complexity, 1.0)
        
        estimated_effort = base_effort * complexity_multiplier
        
        return {
            "total_effort": f"{estimated_effort:.0f} hours",
            "team_size": "2-3 testers",
            "duration": f"{estimated_effort / 8:.0f} person-days",
            "breakdown": {
                "planning": f"{estimated_effort * 0.2:.0f} hours",
                "execution": f"{estimated_effort * 0.6:.0f} hours",
                "reporting": f"{estimated_effort * 0.2:.0f} hours"
            }
        }
    
    async def _assess_risks(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Assess testing risks"""
        
        risks = [
            {
                "risk": "Test environment instability",
                "impact": "Medium",
                "probability": "Medium",
                "mitigation": "Have backup environment ready, monitor environment health"
            },
            {
                "risk": "Incomplete requirements",
                "impact": "High",
                "probability": "Low",
                "mitigation": "Regular requirement reviews, maintain traceability matrix"
            },
            {
                "risk": "Late delivery of features",
                "impact": "High",
                "probability": "Medium",
                "mitigation": "Parallel test case development, incremental testing approach"
            }
        ]
        
        # Add analysis-specific risks
        complexity_indicators = analysis.get("complexity_indicators", [])
        if "integration" in complexity_indicators:
            risks.append({
                "risk": "Integration points failure",
                "impact": "High",
                "probability": "Medium",
                "mitigation": "Early integration testing, mock services for dependencies"
            })
        
        if "real-time" in complexity_indicators:
            risks.append({
                "risk": "Performance issues in real-time features",
                "impact": "High",
                "probability": "Medium",
                "mitigation": "Early performance testing, load testing automation"
            })
        
        return risks

class TestGeneratorAgent(BaseAgent):
    """Agent for generating comprehensive test cases"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.GENERATOR, config)
        
    async def execute(self, task: WorkflowTask) -> Dict[str, Any]:
        """Generate comprehensive test cases"""
        try:
            logger.info(f"[{self.name}] Generating test cases...")
            
            test_plan = task.inputs.get("test_plan")
            enhanced_analysis = task.inputs.get("enhanced_analysis")
            test_config = task.inputs.get("test_config", {})
            
            # Generate different types of test cases
            manual_tests = await self._generate_manual_test_cases(
                enhanced_analysis, test_plan, test_config
            )
            
            automated_tests = await self._generate_automated_test_cases(
                enhanced_analysis, test_plan, test_config
            )
            
            # Generate Robot Framework scripts
            robot_scripts = await self._generate_robot_scripts(automated_tests, test_config)
            
            return {
                "manual_test_cases": manual_tests,
                "automated_test_cases": automated_tests,
                "robot_scripts": robot_scripts,
                "test_data": await self._generate_test_data(enhanced_analysis),
                "coverage_analysis": await self._analyze_coverage(manual_tests, automated_tests, enhanced_analysis)
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            raise
    
    async def _generate_manual_test_cases(self, analysis: Dict[str, Any], 
                                        test_plan: Dict[str, Any],
                                        config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate manual test cases"""
        
        # Import the existing test case generator
        import sys
        import os
        
        # Add the parent directories to the path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if grandparent_dir not in sys.path:
            sys.path.insert(0, grandparent_dir)
        
        try:
            from generator.testcase_generator import TestCaseGenerator
        except ImportError:
            try:
                from gen_ai.generator.testcase_generator import TestCaseGenerator
            except ImportError:
                # Create a minimal test case generator if imports fail
                logger.warning("Could not import TestCaseGenerator, using fallback")
                return []
        
        generator = TestCaseGenerator(analysis, config)
        
        # Generate base test cases
        base_tests = generator.generate(
            scope=config.get("scope", "Both"),
            test_type=config.get("test_type", "All"),
            components=config.get("components", "").split(",") if config.get("components") else None
        )
        
        # If no tests were generated, create some basic ones from requirements
        if not base_tests:
            base_tests = self._create_fallback_test_cases(analysis)
        
        # Enhance with manual-specific details - Include ALL tests as manual candidates
        # Some tests may be better suited for manual execution even if automation is possible
        manual_tests = []
        for test in base_tests:
            # Create manual version of the test case
            manual_test = {
                **test,
                "execution_type": "manual",
                "estimated_time": self._estimate_manual_execution_time(test),
                "prerequisites": self._generate_prerequisites(test, analysis),
                "detailed_steps": self._enhance_steps_for_manual(test.get("steps", [])),
                "validation_points": self._generate_validation_points(test),
                "manual_priority": self._determine_manual_priority(test)
            }
            manual_tests.append(manual_test)
        
        # Apply intelligent filtering for manual tests
        # Prioritize tests that are better suited for manual execution
        filtered_manual_tests = []
        for test in manual_tests:
            automation_feasibility = test.get("automation_feasibility", {})
            complexity = automation_feasibility.get("complexity", "medium")
            effort = automation_feasibility.get("effort", "medium")
            
            # Include tests that are:
            # 1. Explicitly marked as not feasible for automation
            # 2. High complexity or effort to automate
            # 3. Usability/UX focused tests
            # 4. Exploratory testing scenarios
            # 5. Business logic validation requiring human judgment
            
            should_include_manual = (
                not automation_feasibility.get("feasible", True) or
                complexity in ["high", "very_high"] or
                effort in ["high", "very_high"] or
                "usability" in test.get("category", "").lower() or
                "ux" in test.get("category", "").lower() or
                "exploratory" in test.get("description", "").lower() or
                "visual" in test.get("description", "").lower() or
                test.get("manual_priority", "medium") == "high" or
                "user experience" in test.get("description", "").lower() or
                "look and feel" in test.get("description", "").lower()
            )
            
            if should_include_manual:
                filtered_manual_tests.append(test)
        
        # Ensure we have at least some manual test cases (minimum 20% of total)
        if len(filtered_manual_tests) < len(base_tests) * 0.2:
            # Add some automated tests as manual alternatives for better coverage
            remaining_tests = [t for t in manual_tests if t not in filtered_manual_tests]
            additional_count = int(len(base_tests) * 0.2) - len(filtered_manual_tests)
            filtered_manual_tests.extend(remaining_tests[:additional_count])
        
        return filtered_manual_tests
    
    async def _generate_automated_test_cases(self, analysis: Dict[str, Any], 
                                           test_plan: Dict[str, Any],
                                           config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automated test cases"""
        
        try:
            from generator.testcase_generator import TestCaseGenerator
        except ImportError:
            try:
                from gen_ai.generator.testcase_generator import TestCaseGenerator
            except ImportError:
                # Create a minimal test case generator if imports fail
                logger.warning("Could not import TestCaseGenerator for automated tests, using fallback")
                return []
        
        generator = TestCaseGenerator(analysis, config)
        base_tests = generator.generate(
            scope=config.get("scope", "Both"),
            test_type=config.get("test_type", "All"),
            components=config.get("components", "").split(",") if config.get("components") else None
        )
        
        # If no tests were generated, create some basic ones from requirements
        if not base_tests:
            base_tests = self._create_fallback_test_cases(analysis)
        
        # Filter and enhance for automation - be more selective
        automated_tests = []
        for test in base_tests:
            automation_feasibility = test.get("automation_feasibility", {})
            complexity = automation_feasibility.get("complexity", "medium")
            effort = automation_feasibility.get("effort", "medium")
            
            # Only include tests that are truly suitable for automation
            is_suitable_for_automation = (
                automation_feasibility.get("feasible", True) and
                complexity not in ["very_high"] and
                effort not in ["very_high"] and
                "visual" not in test.get("description", "").lower() and
                "usability" not in test.get("category", "").lower() and
                "ux" not in test.get("category", "").lower() and
                "exploratory" not in test.get("description", "").lower() and
                "user experience" not in test.get("description", "").lower()
            )
            
            if is_suitable_for_automation:
                automated_test = {
                    **test,
                    "execution_type": "automated",
                    "automation_framework": "Robot Framework",
                    "selectors": self._generate_selectors(test, analysis),
                    "data_driven": self._determine_data_driven_approach(test),
                    "error_handling": self._generate_error_handling(test),
                    "maintenance_notes": self._generate_maintenance_notes(test)
                }
                automated_tests.append(automated_test)
        
        return automated_tests
    
    async def _generate_robot_scripts(self, automated_tests: List[Dict[str, Any]], 
                                    config: Dict[str, Any]) -> str:
        """Generate Robot Framework scripts"""
        
        # Import the robot writer
        import os
        
        try:
            from robot_writer.robot_writer import RobotWriter
        except ImportError:
            try:
                from gen_ai.robot_writer.robot_writer import RobotWriter
            except ImportError:
                logger.warning("Could not import RobotWriter, returning simple Robot script")
                return self._generate_simple_robot_script(automated_tests)
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), "generated_tests")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            writer = RobotWriter(output_dir)
            
            # Convert test cases to Robot Framework format
            robot_file_path = writer.write(automated_tests, "agentic_automated_tests.robot")
            
            # Read the generated content for return
            robot_content = ""
            if robot_file_path and os.path.exists(robot_file_path):
                try:
                    with open(robot_file_path, 'r', encoding='utf-8') as f:
                        robot_content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read generated Robot file: {e}")
                    robot_content = f"# Robot Framework script generated successfully\n# File location: {robot_file_path}"
            
            return robot_content
        except Exception as e:
            logger.warning(f"RobotWriter failed: {e}, generating simple script")
            return self._generate_simple_robot_script(automated_tests)
    
    def _generate_simple_robot_script(self, automated_tests: List[Dict[str, Any]]) -> str:
        """Generate a simple Robot Framework script when RobotWriter is not available"""
        
        script_lines = [
            "*** Settings ***",
            "Library    SeleniumLibrary",
            "Library    Collections",
            "",
            "*** Variables ***",
            "${BROWSER}    Chrome",
            "${BASE_URL}    http://localhost:8080",
            "",
            "*** Test Cases ***"
        ]
        
        for i, test in enumerate(automated_tests[:10]):  # Limit to 10 tests for simplicity
            test_name = test.get('title', f'Test Case {i+1}').replace(' ', '_').replace('-', '_')
            script_lines.extend([
                f"{test_name}",
                f"    [Documentation]    {test.get('description', 'Automated test case')}",
                f"    [Tags]    {test.get('category', 'automated')}",
                f"    Log    Executing test: {test.get('title', test_name)}",
            ])
            
            # Add test steps
            steps = test.get('steps', [])
            for step in steps[:5]:  # Limit steps
                script_lines.append(f"    Log    {step}")
            
            script_lines.extend([
                f"    Log    Expected Result: {test.get('expected_result', 'Test completed successfully')}",
                ""
            ])
        
        script_lines.extend([
            "*** Keywords ***",
            "Setup Test Environment",
            "    Open Browser    ${BASE_URL}    ${BROWSER}",
            "    Maximize Browser Window",
            "",
            "Teardown Test Environment", 
            "    Close All Browsers"
        ])
        
        return "\n".join(script_lines)

    def _create_fallback_test_cases(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback test cases when generator doesn't produce any"""
        
        fallback_tests = []
        requirements = analysis.get("requirements", [])
        
        for i, req in enumerate(requirements):
            req_text = req.get("text", "")
            req_id = req.get("id", f"REQ{i+1:03d}")
            category = req.get("category", "functional")
            
            # Determine if test should be manual or automated
            is_visual_or_ux = any(keyword in req_text.lower() for keyword in 
                                ["visual", "ui", "ux", "responsive", "look", "feel", "design", "brand"])
            
            # Create basic test case structure
            test_case = {
                "id": f"TC{i+1:03d}",
                "title": f"Verify {req_text}",
                "description": f"Test to verify requirement: {req_text}",
                "category": category,
                "priority": "Medium",
                "complexity": "Medium",
                "requirement_ids": [req_id],
                "steps": [
                    f"Navigate to the relevant section",
                    f"Perform action related to: {req_text}",
                    f"Verify expected behavior"
                ],
                "expected_result": f"Requirement '{req_text}' is satisfied",
                "automation_feasibility": {
                    "feasible": not is_visual_or_ux,
                    "complexity": "high" if is_visual_or_ux else "medium",
                    "effort": "high" if is_visual_or_ux else "medium",
                    "reason": "Visual/UX testing requires manual validation" if is_visual_or_ux else "Can be automated"
                }
            }
            
            fallback_tests.append(test_case)
        
        # Ensure we have at least one test case
        if not fallback_tests:
            fallback_tests.append({
                "id": "TC001",
                "title": "Basic Application Functionality Test",
                "description": "Verify that the application loads and basic functionality works",
                "category": "functional",
                "priority": "High",
                "complexity": "Low",
                "requirement_ids": ["BASIC001"],
                "steps": [
                    "Launch the application",
                    "Verify the application loads successfully",
                    "Test basic navigation and functionality"
                ],
                "expected_result": "Application loads and basic functionality works as expected",
                "automation_feasibility": {
                    "feasible": True,
                    "complexity": "low",
                    "effort": "low",
                    "reason": "Basic functionality can be easily automated"
                }
            })
        
        logger.info(f"Created {len(fallback_tests)} fallback test cases")
        return fallback_tests

    def _estimate_manual_execution_time(self, test: Dict[str, Any]) -> str:
        """Estimate manual execution time"""
        
        steps = test.get("steps", [])
        base_time = len(steps) * 2  # 2 minutes per step
        
        complexity = test.get("complexity", "Medium")
        complexity_multiplier = {
            "Low": 0.8,
            "Medium": 1.0,
            "High": 1.5,
            "Very High": 2.0
        }.get(complexity, 1.0)
        
        estimated_minutes = base_time * complexity_multiplier
        
        if estimated_minutes < 60:
            return f"{estimated_minutes:.0f} minutes"
        else:
            return f"{estimated_minutes/60:.1f} hours"
    
    def _generate_prerequisites(self, test: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Generate prerequisites for test execution"""
        
        prerequisites = [
            "Test environment is accessible and stable",
            "Required test data is available"
        ]
        
        # Add test-specific prerequisites
        if "login" in test.get("title", "").lower():
            prerequisites.append("Valid user credentials are available")
        
        if "payment" in test.get("title", "").lower():
            prerequisites.append("Test payment gateway is configured")
            prerequisites.append("Test credit card details are available")
        
        return prerequisites
    
    def _enhance_steps_for_manual(self, steps: List[str]) -> List[Dict[str, str]]:
        """Enhance steps with additional details for manual execution"""
        
        enhanced_steps = []
        for i, step in enumerate(steps, 1):
            enhanced_step = {
                "step_number": i,
                "action": step,
                "expected_result": f"Step {i} completes successfully",
                "notes": "Verify that the action produces the expected result before proceeding"
            }
            
            # Add specific guidance based on step content
            if "click" in step.lower():
                enhanced_step["notes"] = "Ensure the element is clickable and responds appropriately"
            elif "enter" in step.lower() or "input" in step.lower():
                enhanced_step["notes"] = "Verify that the input is accepted and formatted correctly"
            elif "verify" in step.lower() or "check" in step.lower():
                enhanced_step["notes"] = "Take screenshot or note if verification fails"
            
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps
    
    def _generate_validation_points(self, test: Dict[str, Any]) -> List[str]:
        """Generate validation points for manual testing"""
        
        validation_points = [
            "All steps execute without errors",
            "Expected results match actual results",
            "No unexpected behavior or error messages",
            "UI elements display correctly"
        ]
        
        # Add specific validation based on test category
        category = test.get("category", "").lower()
        if "api" in category:
            validation_points.extend([
                "Response status codes are correct",
                "Response data format is valid",
                "Response time is acceptable"
            ])
        elif "ui" in category:
            validation_points.extend([
                "Visual elements are properly aligned",
                "Text is readable and correctly formatted",
                "Interactive elements respond appropriately"
            ])
        elif "security" in category:
            validation_points.extend([
                "Unauthorized access is prevented",
                "Sensitive data is protected",
                "Error messages don't reveal sensitive information"
            ])
        
        return validation_points
    
    def _determine_manual_priority(self, test: Dict[str, Any]) -> str:
        """Determine priority for manual testing"""
        
        # Check if test involves visual elements or user experience
        description = test.get("description", "").lower()
        category = test.get("category", "").lower()
        
        high_priority_keywords = [
            "visual", "ui", "ux", "user experience", "usability", 
            "accessibility", "responsive", "layout", "design",
            "workflow", "user journey", "exploratory"
        ]
        
        if any(keyword in description or keyword in category for keyword in high_priority_keywords):
            return "high"
        
        # Check automation feasibility
        automation_feasibility = test.get("automation_feasibility", {})
        if not automation_feasibility.get("feasible", True):
            return "high"
        
        if automation_feasibility.get("complexity", "medium") in ["high", "very_high"]:
            return "medium"
        
        return "low"
    
    def _generate_selectors(self, test: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate selectors for automated tests"""
        
        selectors = {}
        
        # Extract potential selectors from test steps
        steps = test.get("steps", [])
        for step in steps:
            if "click" in step.lower():
                if "button" in step.lower():
                    selectors["button"] = "xpath=//button[contains(text(), 'Submit')]"
                elif "link" in step.lower():
                    selectors["link"] = "xpath=//a[contains(text(), 'Continue')]"
            elif "enter" in step.lower() or "input" in step.lower():
                if "username" in step.lower():
                    selectors["username_field"] = "id=username"
                elif "password" in step.lower():
                    selectors["password_field"] = "id=password"
                elif "email" in step.lower():
                    selectors["email_field"] = "id=email"
        
        return selectors
    
    def _determine_data_driven_approach(self, test: Dict[str, Any]) -> bool:
        """Determine if test should use data-driven approach"""
        
        # Check if test involves multiple data scenarios
        description = test.get("description", "").lower()
        steps = test.get("steps", [])
        
        data_driven_indicators = [
            "multiple", "various", "different", "range of", 
            "boundary", "valid", "invalid", "edge case"
        ]
        
        return any(indicator in description for indicator in data_driven_indicators) or \
               any(indicator in " ".join(steps).lower() for indicator in data_driven_indicators)
    
    def _generate_error_handling(self, test: Dict[str, Any]) -> List[str]:
        """Generate error handling strategies for automated tests"""
        
        error_handling = [
            "Take screenshot on failure",
            "Log detailed error information",
            "Retry failed actions once",
            "Clean up test data on failure"
        ]
        
        # Add specific error handling based on test type
        if "api" in test.get("category", "").lower():
            error_handling.extend([
                "Validate response status codes",
                "Handle timeout exceptions",
                "Verify error response format"
            ])
        elif "ui" in test.get("category", "").lower():
            error_handling.extend([
                "Wait for elements to be visible",
                "Handle stale element references",
                "Verify page load completion"
            ])
        
        return error_handling
    
    def _generate_maintenance_notes(self, test: Dict[str, Any]) -> List[str]:
        """Generate maintenance notes for automated tests"""
        
        notes = [
            "Update selectors if UI changes",
            "Review test data periodically",
            "Monitor test execution times"
        ]
        
        # Add specific notes based on automation feasibility
        automation_feasibility = test.get("automation_feasibility", {})
        complexity = automation_feasibility.get("complexity", "medium")
        
        if complexity in ["high", "very_high"]:
            notes.extend([
                "Complex test - review regularly for optimization",
                "Consider breaking into smaller test cases",
                "Monitor for flaky behavior"
            ])
        
        return notes

    async def _generate_test_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data based on analysis"""
        
        test_data = {
            "valid_data": {},
            "invalid_data": {},
            "boundary_data": {},
            "special_cases": {}
        }
        
        # Generate data based on brand context
        brand_context = analysis.get("brand_context", {})
        if isinstance(brand_context, str):
            domain = brand_context.lower()
        else:
            domain = brand_context.get("domain", "").lower()
        
        if "hosting" in domain:
            test_data["valid_data"] = {
                "domain_names": ["example.com", "test-site.org", "my-website.net"],
                "email_addresses": ["user@example.com", "admin@test.com"],
                "passwords": ["SecurePass123!", "TestPassword456#"]
            }
        elif "domain" in domain:
            test_data["valid_data"] = {
                "domain_names": ["newdomain.com", "searchtest.org", "available.net"],
                "whois_data": {"registrant": "Test User", "email": "user@test.com"}
            }
        
        # Add invalid data
        test_data["invalid_data"] = {
            "domain_names": ["invalid..domain", "toolongdomainnamethatshouldnotbeaccepted.com"],
            "email_addresses": ["invalid-email", "user@", "@domain.com"],
            "passwords": ["weak", "123", ""]
        }
        
        return test_data

    async def _analyze_coverage(self, manual_tests: List[Dict[str, Any]], 
                              automated_tests: List[Dict[str, Any]],
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test coverage"""
        
        total_requirements = len(analysis.get("requirements", []))
        total_tests = len(manual_tests) + len(automated_tests)
        
        covered_requirements = set()
        for test in manual_tests + automated_tests:
            # Extract requirement IDs from test case
            req_ids = test.get("requirement_ids", [])
            covered_requirements.update(req_ids)
        
        coverage_percentage = (len(covered_requirements) / total_requirements * 100) if total_requirements > 0 else 0
        
        return {
            "total_requirements": total_requirements,
            "covered_requirements": len(covered_requirements),
            "coverage_percentage": round(coverage_percentage, 2),
            "total_test_cases": total_tests,
            "manual_test_cases": len(manual_tests),
            "automated_test_cases": len(automated_tests),
            "automation_percentage": round((len(automated_tests) / total_tests * 100) if total_tests > 0 else 0, 2)
        }

class WorkflowOrchestrator:
    """Orchestrates the entire agentic workflow"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tasks = []
        self.agents = {
            AgentType.ANALYZER: RequirementAnalyzerAgent(config),
            AgentType.PLANNER: TestPlannerAgent(config),
            AgentType.GENERATOR: TestGeneratorAgent(config)
        }
        self.results = {}
        
    async def execute_workflow(self, analysis: Dict[str, Any], 
                             test_config: Dict[str, Any]) -> WorkflowResult:
        """Execute the complete agentic workflow"""
        
        try:
            logger.info("Starting agentic workflow for test generation...")
            
            # Create workflow tasks
            self._create_workflow_tasks(analysis, test_config)
            
            # Execute tasks in dependency order
            await self._execute_tasks()
            
            # Compile final results
            result = await self._compile_results()
            
            logger.info("Agentic workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                success=False,
                errors=[str(e)]
            )
    
    def _create_workflow_tasks(self, analysis: Dict[str, Any], test_config: Dict[str, Any]):
        """Create workflow tasks with dependencies"""
        
        # Task 1: Enhanced Requirement Analysis
        analysis_task = WorkflowTask(
            id="task_analysis",
            name="Enhanced Requirement Analysis",
            agent_type=AgentType.ANALYZER,
            status=TaskStatus.PENDING,
            priority=1,
            dependencies=[],
            inputs={
                "analysis": analysis,
                "brand_context": test_config.get("brand_context", "Auto-detect")
            },
            outputs={}
        )
        
        # Task 2: Test Planning
        planning_task = WorkflowTask(
            id="task_planning",
            name="Comprehensive Test Planning",
            agent_type=AgentType.PLANNER,
            status=TaskStatus.PENDING,
            priority=2,
            dependencies=["task_analysis"],
            inputs={"test_config": test_config},
            outputs={}
        )
        
        # Task 3: Test Generation
        generation_task = WorkflowTask(
            id="task_generation",
            name="Intelligent Test Generation",
            agent_type=AgentType.GENERATOR,
            status=TaskStatus.PENDING,
            priority=3,
            dependencies=["task_analysis", "task_planning"],
            inputs={"test_config": test_config},
            outputs={}
        )
        
        self.tasks = [analysis_task, planning_task, generation_task]
    
    async def _execute_tasks(self):
        """Execute tasks in dependency order"""
        
        while any(task.status == TaskStatus.PENDING for task in self.tasks):
            # Find tasks ready to execute (all dependencies completed)
            ready_tasks = [
                task for task in self.tasks 
                if task.status == TaskStatus.PENDING and self._dependencies_met(task)
            ]
            
            if not ready_tasks:
                # Check for circular dependencies or other issues
                pending_tasks = [task for task in self.tasks if task.status == TaskStatus.PENDING]
                if pending_tasks:
                    logger.error("Workflow deadlock detected - circular dependencies or missing agents")
                    for task in pending_tasks:
                        task.status = TaskStatus.FAILED
                        task.error_message = "Workflow deadlock"
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                await self._execute_task(task)
    
    def _dependencies_met(self, task: WorkflowTask) -> bool:
        """Check if task dependencies are met"""
        
        for dep_id in task.dependencies:
            dep_task = next((t for t in self.tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _execute_task(self, task: WorkflowTask):
        """Execute a single task"""
        
        try:
            logger.info(f"Executing task: {task.name}")
            
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            start_time = time.time()
            
            # Get agent
            agent = self.agents.get(task.agent_type)
            if not agent:
                raise ValueError(f"No agent available for type: {task.agent_type}")
            
            # Prepare inputs with results from dependencies
            task_inputs = task.inputs.copy()
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.tasks if t.id == dep_id), None)
                if dep_task and dep_task.outputs:
                    task_inputs.update(dep_task.outputs)
            
            task.inputs = task_inputs
            
            # Execute task
            outputs = await agent.execute(task)
            task.outputs = outputs
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            task.duration = time.time() - start_time
            
            logger.info(f"Task completed: {task.name} (Duration: {task.duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Task failed: {task.name} - {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            task.duration = time.time() - start_time if 'start_time' in locals() else 0
    
    def _enforce_manual_automated_balance(self, manual_tests: List[Dict[str, Any]], 
                                        automated_tests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Enforce business rule: Manual test cases must always be >= automated test cases"""
        if not manual_tests:
            manual_tests = []
        if not automated_tests:
            automated_tests = []
            
        manual_count = len(manual_tests)
        automated_count = len(automated_tests)
        
        logger.info(f"Original test distribution: {manual_count} manual, {automated_count} automated")
        
        # If manual >= automated, we're good
        if manual_count >= automated_count:
            logger.info("Manual test count already satisfies business rule")
            return manual_tests, automated_tests
        
        # Business rule violation: automated > manual
        deficit = automated_count - manual_count
        logger.warning(f"Business rule violation detected: {automated_count} automated > {manual_count} manual. Deficit: {deficit}")
        
        # Convert some automated tests to manual tests
        candidates_for_conversion = []
        remaining_automated = []
        
        for test in automated_tests:
            should_convert = (
                deficit > 0 and (
                    "visual" in test.get("description", "").lower() or
                    "validation" in test.get("description", "").lower() or
                    test.get("automation_feasibility", {}).get("complexity", "medium") in ["high", "very_high"]
                )
            )
            
            if should_convert:
                manual_version = {
                    **test,
                    "execution_type": "manual",
                    "converted_from_automated": True,
                    "conversion_reason": "Business rule enforcement: maintain manual >= automated balance"
                }
                candidates_for_conversion.append(manual_version)
                deficit -= 1
            else:
                remaining_automated.append(test)
        
        new_manual_tests = manual_tests + candidates_for_conversion
        new_automated_tests = remaining_automated
        
        # If still have deficit, remove some automated tests
        if deficit > 0:
            new_automated_tests = new_automated_tests[:-deficit]
        
        final_manual_count = len(new_manual_tests)
        final_automated_count = len(new_automated_tests)
        
        logger.info(f"Balanced test distribution: {final_manual_count} manual, {final_automated_count} automated")
        
        return new_manual_tests, new_automated_tests
    
    async def _compile_results(self) -> WorkflowResult:
        """Compile final workflow results"""
        
        # Check if workflow succeeded
        failed_tasks = [task for task in self.tasks if task.status == TaskStatus.FAILED]
        success = len(failed_tasks) == 0
        
        if not success:
            return WorkflowResult(
                success=False,
                errors=[f"Task '{task.name}' failed: {task.error_message}" for task in failed_tasks]
            )
        
        # Extract results from completed tasks
        test_plan = None
        manual_test_cases = None
        automated_test_cases = None
        robot_scripts = None
        recommendations = []
        metrics = {}
        
        for task in self.tasks:
            if task.agent_type == AgentType.PLANNER and task.outputs:
                test_plan = task.outputs.get("test_plan")
                recommendations.extend(task.outputs.get("strategy_recommendations", []))
                
            elif task.agent_type == AgentType.GENERATOR and task.outputs:
                manual_test_cases = task.outputs.get("manual_test_cases")
                automated_test_cases = task.outputs.get("automated_test_cases")
                robot_scripts = task.outputs.get("robot_scripts")
                metrics.update(task.outputs.get("coverage_analysis", {}))
        
        # CRITICAL: Enforce business rule - manual test cases must be >= automated test cases
        if manual_test_cases is None:
            manual_test_cases = []
        if automated_test_cases is None:
            automated_test_cases = []
            
        # Apply business logic validation and rebalancing
        manual_test_cases, automated_test_cases = self._enforce_manual_automated_balance(
            manual_test_cases, automated_test_cases
        )
        
        # Add enhanced workflow metrics with agent information
        total_duration = sum(task.duration or 0 for task in self.tasks)
        completed_tasks = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.tasks if t.status == TaskStatus.FAILED]
        
        # Agent performance tracking
        agent_performance = {}
        for task in self.tasks:
            agent_name = task.agent_type.value
            if agent_name not in agent_performance:
                agent_performance[agent_name] = {
                    "status": task.status.value,
                    "duration": task.duration or 0,
                    "task_name": task.name
                }
        
        # Calculate comprehensive test coverage
        total_manual = len(manual_test_cases) if manual_test_cases else 0
        total_automated = len(automated_test_cases) if automated_test_cases else 0
        total_tests = total_manual + total_automated
        
        automation_percentage = (total_automated / total_tests * 100) if total_tests > 0 else 0
        manual_percentage = (total_manual / total_tests * 100) if total_tests > 0 else 0
        
        # Enhanced metrics
        metrics.update({
            "workflow_duration": f"{total_duration:.2f} seconds",
            "total_tasks": len(self.tasks),
            "tasks_executed": len(self.tasks),
            "tasks_succeeded": len(completed_tasks),
            "tasks_failed": len(failed_tasks),
            "success_rate": f"{(len(completed_tasks) / len(self.tasks) * 100):.1f}%" if self.tasks else "0%",
            
            # Agent information
            "agents_used": len(agent_performance),
            "agents_available": len(self.agents),
            "agent_performance": agent_performance,
            "agents_missing": [agent_type.value for agent_type in [AgentType.ANALYZER, AgentType.PLANNER, AgentType.GENERATOR, AgentType.REVIEWER, AgentType.OPTIMIZER] 
                             if agent_type not in self.agents],
            
            # Test distribution
            "test_distribution": {
                "manual_tests": total_manual,
                "automated_tests": total_automated,
                "total_tests": total_tests,
                "manual_percentage": f"{manual_percentage:.1f}%",
                "automation_percentage": f"{automation_percentage:.1f}%"
            },
            
            # Coverage validation
            "coverage_validation": {
                "balanced_approach": 20 <= automation_percentage <= 80,
                "manual_coverage_adequate": total_manual >= max(1, total_tests * 0.2),
                "automation_coverage_reasonable": total_automated >= max(1, total_tests * 0.2) if total_tests > 2 else True,
                "business_rule_satisfied": total_manual >= total_automated,  # CRITICAL business rule
                "manual_automated_ratio": f"{total_manual}:{total_automated}" if total_automated > 0 else f"{total_manual}:0"
            }
        })
        
        return WorkflowResult(
            success=True,
            test_plan=test_plan,
            manual_test_cases=manual_test_cases,
            automated_test_cases=automated_test_cases,
            robot_scripts=robot_scripts,
            recommendations=recommendations,
            metrics=metrics
        )

# Convenience function for running the workflow
async def run_agentic_workflow(analysis: Dict[str, Any], 
                              test_config: Dict[str, Any]) -> WorkflowResult:
    """Run the complete agentic workflow"""
    
    orchestrator = WorkflowOrchestrator(test_config)
    return await orchestrator.execute_workflow(analysis, test_config)
