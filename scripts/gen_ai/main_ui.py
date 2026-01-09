import streamlit as st
import importlib
import sys
import os
import json
from datetime import datetime
import pandas as pd
import logging

# Enhanced logging setup (before page config)
try:
    # Add the use_cases directory to path for enhanced_logging import
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    use_cases_dir = os.path.join(script_dir, "gen_ai", "use_cases")
    if use_cases_dir not in sys.path:
        sys.path.insert(0, use_cases_dir)

    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("MainUI", level=logging.INFO, log_file="main_ui.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Configure the page to use wide mode and set a nice title
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="The Vortex - Gen AI Testing Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safely import optional dependencies
try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False
    st.warning("""
    📦 Some visualisation features require additional packages.
    Run: `pip3 install -r requirements.txt` to install all dependencies.
    """)

# Add the scripts directory to the path for imports
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# ============================================================================
# AUTHENTICATION & AUTHORIZATION SYSTEM
# ============================================================================
try:
    from gen_ai.auth.user_manager import UserManager
    from gen_ai.auth.audit_logger import AuditLogger, AuditAction, AuditSeverity
    from gen_ai.auth.auth_manager import AuthManager, StreamlitAuthManager
    from gen_ai.auth.rbac_config import Permission, SYSTEM_ROLES
    from gen_ai.auth.login_ui import render_login_page, render_password_change_form, render_user_profile
    from gen_ai.auth.admin_panel import render_admin_panel

    # Initialize auth components (singleton pattern)
    if 'auth_initialized' not in st.session_state:
        auth_dir = os.path.join(script_dir, "gen_ai", "auth")
        user_manager = UserManager(storage_path=os.path.join(auth_dir, "users.json"))
        audit_logger = AuditLogger(log_path=os.path.join(auth_dir, "audit_logs.jsonl"))
        auth_manager = AuthManager(user_manager, audit_logger)
        st_auth = StreamlitAuthManager(auth_manager)

        # Store in session state
        st.session_state.user_manager = user_manager
        st.session_state.audit_logger = audit_logger
        st.session_state.auth_manager = auth_manager
        st.session_state.st_auth = st_auth
        st.session_state.auth_initialized = True
    else:
        # Retrieve from session state
        st_auth = st.session_state.st_auth

    AUTH_ENABLED = True
except ImportError as e:
    st.warning(f"""
    ⚠️ Authentication system not fully initialized. 
    Some features may be limited. Error: {str(e)}
    """)
    AUTH_ENABLED = False
    st_auth = None

# Apply custom CSS for modern Newfold Digital branding
st.markdown("""
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* CSS Variables for theme consistency */
:root {
    --primary-color: #EC5328;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --secondary-color: #EC5328;
    --accent-color: #f59e0b;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --background-primary: #ffffff;
    --background-secondary: #f8fafc;
    --background-dark: #0f172a;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-light: #94a3b8;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    --gradient-primary: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    --gradient-accent: linear-gradient(135deg, var(--accent-color), var(--warning-color));
    --border-radius-sm: 0.375rem;
    --border-radius-md: 0.5rem;
    --border-radius-lg: 0.75rem;
    --border-radius-xl: 1rem;
}

/* Global styles */
* {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main app container */
.main .block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1600px;
    margin: 0 auto;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: var(--border-radius-xl);
    box-shadow: var(--shadow-sm);
}

/* Improve overall spacing */
.element-container {
    margin-bottom: 0.75rem;
}

/* Better section dividers */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 1.5rem 0;
}

/* Remove excessive top spacing from Streamlit */
.main > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.main {
    padding-top: 0 !important;
}

/* Compact header spacing */
div[data-testid="stVerticalBlock"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove top padding from all Streamlit containers */
section.main > div {
    padding-top: 0 !important;
}

.stApp {
    padding-top: 0 !important;
}

/* Force content to top */
.block-container {
    padding-top: 0 !important;
}

div[data-testid="stAppViewContainer"] > section {
    padding-top: 0 !important;
}

/* Enhanced header with glassmorphism effect */
.main-header {
    font-family: 'Inter', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin: 0 0 3rem 0;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    letter-spacing: -0.025em;
    line-height: 1.1;
}

.main-header::after {
    content: '';
    position: absolute;
    bottom: -1rem;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 4px;
    background: var(--gradient-primary);
    border-radius: var(--border-radius-xl);
    opacity: 0.8;
}

/* Header controls with glassmorphism */
.header-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--border-radius-lg);
    box-shadow: var(--shadow-md);
}

/* Modern button styling */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: var(--border-radius-md);
    padding: 0.75rem 1.5rem;
    background: var(--gradient-primary);
    color: white;
    box-shadow: var(--shadow-md);
    transform: translateY(0);
    position: relative;
    overflow: hidden;
    min-height: 3rem;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    background: #EC5328;
    color: white !important;
    font-weight: 700 !important;
    border: 2px solid white !important;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(0);
}

/* Ensure Submit Feedback button font color stays white on click/active */
div.stButton > button {
    color: #fff !important;
}
div.stButton > button:active {
    color: #fff !important;
}
div.stButton > button:focus {
    color: #fff !important;
}

/* Fix rating dropdown to be non-editable */
.stSelectbox > div > div > div > div > input {
    pointer-events: none !important;
    cursor: default !important;
    caret-color: transparent !important;
}

/* Ensure rating dropdown has proper styling */
div[data-testid="stSelectbox"] > div > div > div > div {
    border-radius: var(--border-radius-md) !important;
}

div[data-testid="stSelectbox"] > div > div > div > div:hover {
    border-color: var(--primary-dark) !important;
    # box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1) !important;
}

/* Additional selectbox styling for feedback section */
.stSelectbox label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Remove spacing between expander sections */
div[data-testid="stExpander"] {
    margin-bottom: 0 !important;
}

div.streamlit-expanderHeader {
    margin-bottom: 0 !important;
}

section.main > div > div > div > div {
    gap: 0 !important;
}

/* Reduce spacing after expanders */
.element-container:has(div[data-testid="stExpander"]) {
    margin-bottom: 0 !important;
}

/* Navigation container with improved grid */
.nav-container {
    padding: 0.5rem;
    background: linear-gradient(135deg, rgba(248, 250, 252, 0.9), rgba(241, 245, 249, 0.9));
    border-radius: var(--border-radius-xl);
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}

/* Modern module card styling */
.module-card {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}

.module-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 12px 24px rgba(236, 83, 40, 0.2);
    border-color: #EC5328;
}

.module-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(236, 83, 40, 0.1), transparent);
    transition: left 0.5s;
}

.module-card:hover::before {
    left: 100%;
}

/* Active module card */
.module-card-active {
    background: linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%);
    border: 3px solid #EC5328;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 16px rgba(236, 83, 40, 0.3);
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
    animation: pulseGlow 2s ease-in-out infinite;
}

@keyframes pulseGlow {
    0%, 100% {
        box-shadow: 0 8px 16px rgba(236, 83, 40, 0.3);
    }
    50% {
        box-shadow: 0 12px 24px rgba(236, 83, 40, 0.5);
    }
}

.module-card-active:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 16px 32px rgba(236, 83, 40, 0.4);
}

/* Category header styling */
.nav-container h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--text-primary);
    margin-bottom: 1.5rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-container h3:first-child {
    margin-top: 0;
}

/* Add hover effect for use case modules */
.st-emotion-cache-10kvrwj:hover {
    color: var(--primary-color) !important;
    border-color: var(--primary-color) !important;
    transition: color 0.3s ease, border-color 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
    background: rgba(248, 250, 252, 0.9);
    border-radius: var(--border-radius-md);
    padding: 0.5rem 1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
}

/* Enhanced search bar */
.search-bar {
    margin-bottom: 2rem;
    position: relative;
}

.search-bar input {
    width: 100%;
    padding: 1rem 1rem 1rem 3rem;
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    background: var(--background-primary);
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.search-bar input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    outline: none;
}

/* Dashboard cards with modern styling */
.dashboard-card {
    background: var(--background-primary);
    border-radius: var(--border-radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
}

.dashboard-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
}

.dashboard-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
}

/* Enhanced metric cards */
.metric-card {
    background: var(--background-primary);
    border-radius: var(--border-radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--gradient-primary);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: var(--shadow-xl);
}

.metric-card h3 {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-card p {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin: 0;
}

.metric-icon {
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    font-size: 2rem;
    opacity: 0.3;
    color: var(--primary-color);
}

/* AI insights with modern styling */
.ai-insights {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(6, 182, 212, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-left: 4px solid var(--primary-color);
    padding: 2rem;
    margin: 2rem 0;
    border-radius: var(--border-radius-lg);
    backdrop-filter: blur(10px);
    position: relative;
}

.ai-insights::before {
    content: '💡';
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 1.5rem;
    opacity: 0.6;
}

.ai-insights h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 1rem;
    font-size: 1.25rem;
}

/* Enhanced notification styling */
.notification-card {
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    border-radius: var(--border-radius-lg);
    box-shadow: var(--shadow-md);
    position: relative;
    border: 1px solid var(--border-color);
    background: var(--background-primary);
    backdrop-filter: blur(10px);
    overflow: hidden;
}

.notification-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--primary-color);
}

.notification-success::before {
    background: var(--success-color);
}

.notification-warning::before {
    background: var(--warning-color);
}

.notification-error::before {
    background: var(--error-color);
}

.notification-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(16, 185, 129, 0.02));
    border-color: rgba(16, 185, 129, 0.2);
}

.notification-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(245, 158, 11, 0.02));
    border-color: rgba(245, 158, 11, 0.2);
}

.notification-error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.05), rgba(239, 68, 68, 0.02));
    border-color: rgba(239, 68, 68, 0.2);
}

.notification-read {
    opacity: 0.7;
    transform: scale(0.98);
}

.notification-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    color: var(--text-light);
    margin-bottom: 0.75rem;
    font-weight: 500;
}

.notification-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    margin-bottom: 1rem;
    font-size: 1.125rem;
    line-height: 1.4;
    color: var(--text-primary);
}

.notification-module {
    background: rgba(99, 102, 241, 0.1);
    display: inline-block;
    padding: 0.375rem 0.75rem;
    border-radius: var(--border-radius-xl);
    font-size: 0.875rem;
    margin-bottom: 1rem;
    font-weight: 600;
    color: var(--primary-color);
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.notification-details {
    margin: 1rem 0;
    padding: 1rem;
    background: rgba(248, 250, 252, 0.8);
    border-radius: var(--border-radius-md);
    border: 1px solid var(--border-color);
    font-size: 0.9375rem;
    color: var(--text-secondary);
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}

.notification-actions {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
}

.action-step {
    margin: 0.75rem 0;
    display: flex;
    align-items: flex-start;
    line-height: 1.5;
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
}

.action-step:before {
    content: "→";
    margin-right: 0.75rem;
    color: var(--primary-color);
    font-weight: bold;
    font-size: 1rem;
    flex-shrink: 0;
}

/* Enhanced accessibility and form styling */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: var(--primary-color);
    font-size: 1.125rem;
}

.stDataFrame {
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.stDataFrame [data-testid="stTable"] {
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.stSelectbox label, .stTextInput label {
    color: var(--text-primary);
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    font-size: 0.9375rem;
}

/* Focus states for accessibility */
button:focus, select:focus, input:focus, textarea:focus {
    outline: 2px solid var(--primary-color) !important;
    outline-offset: 2px;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--background-secondary);
    border-radius: var(--border-radius-md);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #EC5328, #ff6b6b);
    border-radius: var(--border-radius-md);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #d64520, #ff5555);
}

/* Smooth scroll behavior */
html {
    scroll-behavior: smooth;
}

/* Loading indicator */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

/* Scroll to top button */
.scroll-to-top {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: linear-gradient(135deg, #EC5328, #ff6b6b);
    color: white;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(236, 83, 40, 0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 1000;
}

.scroll-to-top:hover {
    transform: translateY(-5px) scale(1.1);
    box-shadow: 0 8px 20px rgba(236, 83, 40, 0.6);
}

/* Enhanced tooltips */
[data-tooltip] {
    position: relative;
    cursor: help;
}

[data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(-8px);
    background: #1e293b;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.875rem;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 1000;
}

[data-tooltip]:hover::after {
    opacity: 1;
}

/* Animation keyframes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}

@keyframes slideIn {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Cosmic Loading Animation Keyframes */
@keyframes cosmicEntry {
    0% {
        opacity: 0;
        transform: scale(0.3) rotateY(90deg);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.1) rotateY(0deg);
    }
    100% {
        opacity: 1;
        transform: scale(1) rotateY(0deg);
    }
}

@keyframes starTwinkle {
    0%, 100% {
        opacity: 0.3;
        transform: scale(0.8);
    }
    50% {
        opacity: 1;
        transform: scale(1.2);
    }
}

@keyframes floatParticle {
    0% {
        transform: translateY(100vh) translateX(-50px);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% {
        transform: translateY(-100px) translateX(50px);
        opacity: 0;
    }
}

@keyframes portalPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(236, 83, 40, 0.4);
        transform: scale(1);
    }
    50% {
        box-shadow: 0 0 0 30px rgba(236, 83, 40, 0);
        transform: scale(1.05);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(236, 83, 40, 0);
        transform: scale(1);
    }
}

@keyframes textGlow {
    0%, 100% {
        text-shadow: 0 0 5px rgba(236, 83, 40, 0.5);
    }
    50% {
        text-shadow: 0 0 20px rgba(236, 83, 40, 0.8), 0 0 30px rgba(236, 83, 40, 0.6);
    }
}

@keyframes galaxyRotate {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(0deg);
    }
}

/* Cosmic Loading Screen */
.cosmic-loader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: linear-gradient(135deg, #0a0a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #e94560 100%);
    background-size: 400% 400%;
    animation: galaxyRotate 20s ease-in-out infinite;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    overflow: hidden;
}

.cosmic-loader.fade-out {
    animation: fadeOut 1s ease-out forwards;
}

@keyframes fadeOut {
    to {
        opacity: 0;
        visibility: hidden;
    }
}

/* Animated Stars Background */
.stars {
    position: absolute;
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.star {
    position: absolute;
    background: white;
    border-radius: 50%;
    animation: starTwinkle 3s ease-in-out infinite;
}

.star:nth-child(1) { width: 2px; height: 2px; top: 20%; left: 20%; animation-delay: 0s; }
.star:nth-child(2) { width: 3px; height: 3px; top: 60%; left: 80%; animation-delay: 0.5s; }
.star:nth-child(3) { width: 1px; height: 1px; top: 80%; left: 30%; animation-delay: 1s; }
.star:nth-child(4) { width: 2px; height: 2px; top: 30%; left: 70%; animation-delay: 1.5s; }
.star:nth-child(5) { width: 3px; height: 3px; top: 70%; left: 10%; animation-delay: 2s; }
.star:nth-child(6) { width: 1px; height: 1px; top: 10%; left: 50%; animation-delay: 2.5s; }
.star:nth-child(7) { width: 2px; height: 2px; top: 50%; left: 90%; animation-delay: 3s; }
.star:nth-child(8) { width: 3px; height: 3px; top: 90%; left: 60%; animation-delay: 3.5s; }

/* Enhanced Stars with more variety */
.star:nth-child(9) { width: 1px; height: 1px; top: 15%; left: 85%; animation-delay: 4s; }
.star:nth-child(10) { width: 2px; height: 2px; top: 75%; left: 65%; animation-delay: 4.5s; }
.star:nth-child(11) { width: 3px; height: 3px; top: 40%; left: 25%; animation-delay: 5s; }
.star:nth-child(12) { width: 1px; height: 1px; top: 85%; left: 75%; animation-delay: 5.5s; }
.star:nth-child(13) { width: 2px; height: 2px; top: 25%; left: 45%; animation-delay: 6s; }
.star:nth-child(14) { width: 3px; height: 3px; top: 95%; left: 15%; animation-delay: 6.5s; }
.star:nth-child(15) { width: 1px; height: 1px; top: 5%; left: 95%; animation-delay: 7s; }
.star:nth-child(16) { width: 2px; height: 2px; top: 65%; left: 35%; animation-delay: 7.5s; }

/* Floating Particles */
.particles {
    position: absolute;
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.particle {
    position: absolute;
    background: linear-gradient(45deg, #EC5328, #ff6b6b, #4ecdc4);
    border-radius: 50%;
    animation: floatParticle 8s linear infinite;
}

/* Enhanced Particles with unique animations */
.particle-1 { animation: floatParticle 6s linear infinite; }
.particle-2 { animation: floatParticle 7s linear infinite; }
.particle-3 { animation: floatParticle 8s linear infinite; }
.particle-4 { animation: floatParticle 9s linear infinite; }
.particle-5 { animation: floatParticle 10s linear infinite; }
.particle-6 { animation: floatParticle 11s linear infinite; }
.particle-7 { animation: floatParticle 12s linear infinite; }
.particle-8 { animation: floatParticle 13s linear infinite; }
.particle-9 { animation: floatParticle 14s linear infinite; }
.particle-10 { animation: floatParticle 15s linear infinite; }
.particle-11 { animation: floatParticle 16s linear infinite; }
.particle-12 { animation: floatParticle 17s linear infinite; }

.particle-1 { left: 5%; }
.particle-2 { left: 15%; }
.particle-3 { left: 25%; }
.particle-4 { left: 35%; }
.particle-5 { left: 45%; }
.particle-6 { left: 55%; }
.particle-7 { left: 65%; }
.particle-8 { left: 75%; }
.particle-9 { left: 85%; }
.particle-10 { left: 95%; }
.particle-11 { left: 12%; }
.particle-12 { left: 88%; }

/* Energy Rings Animation */
.energy-rings {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 5;
}

.ring {
    position: absolute;
    border: 2px solid rgba(236, 83, 40, 0.4);
    border-radius: 50%;
    animation: energyPulse 3s ease-in-out infinite;
}

.ring-1 {
    width: 300px;
    height: 300px;
    margin: -150px 0 0 -150px;
    animation-delay: 0s;
}

.ring-2 {
    width: 400px;
    height: 400px;
    margin: -200px 0 0 -200px;
    animation-delay: 1s;
    border-color: rgba(255, 107, 107, 0.3);
}

.ring-3 {
    width: 500px;
    height: 500px;
    margin: -250px 0 0 -250px;
    animation-delay: 2s;
    border-color: rgba(78, 205, 196, 0.3);
}

@keyframes energyPulse {
    0%, 100% {
        opacity: 0.2;
        transform: scale(0.8);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.2);
    }
}

/* Enhanced Portal with inner elements */
.portal-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60%;
    height: 60%;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 50%;
    animation: none;
}

.portal-core {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 30%;
    height: 30%;
    background: radial-gradient(circle, rgba(236, 83, 40, 0.6) 0%, rgba(236, 83, 40, 0.2) 40%, transparent 70%);
    border-radius: 50%;
    animation: corePulse 2s ease-in-out infinite;
}

@keyframes corePulse {
    0%, 100% {
        opacity: 0.6;
        transform: translate(-50%, -50%) scale(1);
    }
    50% {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1.3);
    }
}

/* Enhanced Cosmic Spinner */
.cosmic-spinner-container {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 3rem;
}

.spinner-glow {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(236, 83, 40, 0.3) 0%, transparent 70%);
    border-radius: 50%;
    animation: glowPulse 2s ease-in-out infinite;
}

@keyframes glowPulse {
    0%, 100% {
        opacity: 0.3;
        transform: translate(-50%, -50%) scale(1);
    }
    50% {
        opacity: 0.8;
        transform: translate(-50%, -50%) scale(1.2);
    }
}

/* Enhanced welcome text styling for loading screen */
.welcome-text {
    text-align: center;
    z-index: 10;
    position: relative;
    margin-bottom: 2rem;
}

.welcome-title {
    font-family: 'Inter', sans-serif;
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff, #EC5328, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    animation: textGlow 3s ease-in-out infinite;
    text-shadow: 0 0 30px rgba(236, 83, 40, 0.5);
    letter-spacing: -0.02em;
}

.welcome-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin-bottom: 1rem;
    animation: fadeInUp 2s ease-out;
}

.welcome-description {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.8);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
    animation: fadeInUp 2.5s ease-out;
}

.loading-text {
    margin-top: 3rem;
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    opacity: 0.9;
    animation: textFade 1.5s ease-in-out infinite alternate;
    text-align: center;
    background: linear-gradient(135deg, #ffffff, #EC5328);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 20px rgba(236, 83, 40, 0.6);
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Add cosmic loading screen HTML immediately after CSS
st.markdown("""
<div class="cosmic-loader" id="cosmicLoader">
    <div class="stars">
        <div class="star"></div><div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div><div class="star"></div>
    </div>
    <div class="particles">
        <div class="particle particle-1"></div><div class="particle particle-2"></div>
        <div class="particle particle-3"></div><div class="particle particle-4"></div>
        <div class="particle particle-5"></div><div class="particle particle-6"></div>
        <div class="particle particle-7"></div><div class="particle particle-8"></div>
        <div class="particle particle-9"></div><div class="particle particle-10"></div>
        <div class="particle particle-11"></div><div class="particle particle-12"></div>
    </div>
    <div class="energy-rings">
        <div class="ring ring-1"></div>
        <div class="ring ring-2"></div>
        <div class="ring ring-3"></div>
    </div>
    <div class="portal">
        <div class="portal-inner"></div>
        <div class="portal-core"></div>
    </div>
    <div class="welcome-text">
        <h1 class="welcome-title">Welcome to The Vortex</h1>
        <h4 class="welcome-subtitle">Entering the Cosmic World of AI</h4>
        <p class="welcome-description">Your gateway to intelligent testing automation powered by advanced AI. Prepare to explore a universe of possibilities where <br>Technology 🤝 Innovation</p>
    </div>
    <div class="cosmic-spinner-container">
        <div class="cosmic-spinner"></div>
        <div class="spinner-glow"></div>
        <div class="loading-text" id="loadingText">Initialising AI Systems...</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add the JavaScript in a separate script tag with better Streamlit compatibility
st.components.v1.html("""
<script>
// Global flag to prevent multiple animations
if (!window.vortexAnimationStarted) {
    window.vortexAnimationStarted = false;
}

// Single animation function
function startVortexAnimation() {
    // Prevent multiple executions
    if (window.vortexAnimationStarted) {
        console.log('Vortex animation already started, skipping...');
        return;
    }
    
    window.vortexAnimationStarted = true;
    console.log('Starting Vortex loading animation');
    
    const texts = [
        'Initialising AI Systems...',
        'Loading Neural Networks...',
        'Configuring Test Modules...',
        'Establishing Connections...',
        'Calibrating ML Models...',
        'Syncing Data Sources...',
        'Preparing Test Environment...',
        'Indexing Use Case Modules...',
        'Optimising Performance...',
        'Finalising Setup...',
        'Welcome to The Vortex!'
    ];
    
    let currentStep = 0;
    
    function updateLoadingText() {
        const loadingElement = document.getElementById('loadingText') || 
                              document.querySelector('.loading-text') ||
                              parent.document.getElementById('loadingText') ||
                              parent.document.querySelector('.loading-text');
        
        if (loadingElement && currentStep < texts.length) {
            loadingElement.textContent = texts[currentStep];
            console.log('Step ' + (currentStep + 1) + '/' + texts.length + ': ' + texts[currentStep]);
            currentStep++;
            return true;
        }
        return false;
    }
    
    function hideLoader() {
        const cosmicLoader = document.getElementById('cosmicLoader') || 
                           parent.document.getElementById('cosmicLoader');
        
        if (cosmicLoader) {
            cosmicLoader.style.transition = 'opacity 1.5s ease-out';
            cosmicLoader.style.opacity = '0';
            
            setTimeout(() => {
                cosmicLoader.style.display = 'none';
                console.log('Loading screen hidden');
                // Reset for potential future use
                window.vortexAnimationStarted = false;
            }, 1500);
        }
    }
    
    // Start the animation
    if (updateLoadingText()) {
        const interval = setInterval(() => {
            if (currentStep < texts.length) {
                if (!updateLoadingText()) {
                    clearInterval(interval);
                }
            } else {
                clearInterval(interval);
                console.log('Animation completed, hiding loader in 2 seconds');
                setTimeout(hideLoader, 2000);
            }
        }, 500); // 0.5 second intervals
    } else {
        console.warn('Could not find loading text element');
        // Hide loader after 8 seconds as fallback
        setTimeout(hideLoader, 8000);
    }
}

// Try to start animation when window loads
window.addEventListener('load', function() {
    console.log('Window loaded, attempting to start Vortex animation');
    startVortexAnimation();
});

// Fallback: Try to start animation after a delay
setTimeout(function() {
    console.log('Fallback timer triggered, attempting to start Vortex animation');
    startVortexAnimation();
}, 1000);
</script>
""", height=0)

# Function to save usage history
def save_usage_history(module_name):
    """Save module usage to history with timestamp."""
    # Always add history for module changes, but avoid duplicates if rapidly clicking the same module
    if not st.session_state.history or module_name != st.session_state.history[-1].get('module'):
        # Add to history with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({'module': module_name, 'timestamp': current_time})

        # Update metrics
        st.session_state.execution_metrics['modules_usage'][module_name] = st.session_state.execution_metrics[
                                                                               'modules_usage'].get(module_name, 0) + 1

        # Save to file for persistent storage immediately, if possible
        try:
            history_file = os.path.join(script_dir, "gen_ai", "usage_history.json")
            with open(history_file, 'w') as f:
                json.dump(st.session_state.history, f)
        except Exception as e:
            st.warning(f"Could not save usage history: {e}")

    # Always update tracking variables
    st.session_state.last_saved_module = module_name

    # Clear pending flag if it exists
    if 'pending_history_module' in st.session_state:
        del st.session_state.pending_history_module


# Function to add a notification
def add_notification(module_name, status, message, details=None, action_steps=None):
    """
    Add a new notification to the system.

    Parameters:
    - module_name: The name of the module generating the notification
    - status: Status type ("success", "warning", "error")
    - message: Short notification message
    - details: Optional detailed information
    - action_steps: Optional list of recommended actions
    """
    if not details:
        details = ""
    if not action_steps:
        action_steps = []

    # Get friendly module name
    module_display_name = get_module_friendly_name(module_name)

    # Create notification object
    notification = {
        "id": len(st.session_state.notifications) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module_name,
        "module_display_name": module_display_name,
        "status": status,
        "message": message,
        "details": details,
        "action_steps": action_steps,
        "read": False
    }

    # Add to notifications list
    st.session_state.notifications.append(notification)

    # Update counters
    st.session_state.notification_count += 1
    st.session_state.unread_notifications += 1

    # Play notification sound if enabled
    if st.session_state.get('notification_sound', False):
        try:
            # Play sound using HTML5 audio
            sound_html = """
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTCO1/LGdiQFJnvK79uSQwoXZLbp7qNUFApJoOLyxG4gBTKQ2fLJeCQEJ3zL79+TPwsYZ7nq76ZYFgxLouPztmcgBTCS2PLLeyUGKH7N8N+UP
QsZabzr8KpbFwxNo+T0u2sgBTOU2vLNfSYGK4DQ8eGWPwsbbr3s8K5eFw5OpuX1wmwiBTSX2/LPgCcHLIHT8uGYQAwccr/t8bBgGA9PqOb2xG8jBTeX3PLS" type="audio/wav">
            </audio>
            """
            # Use a more compatible notification sound
            st.markdown(sound_html, unsafe_allow_html=True)
        except Exception as e:
            # Silently fail if sound cannot be played
            pass

    # Auto-show notifications if enabled
    if st.session_state.get('auto_notifications', True):
        st.session_state.show_notifications = True

    # Store notifications to file for persistence
    try:
        notifications_file = os.path.join(script_dir, "gen_ai", "notifications.json")
        with open(notifications_file, 'w') as f:
            json.dump(st.session_state.notifications, f)
    except Exception as e:
        st.warning(f"Could not save notifications: {e}")


# Function to handle test execution results and generate appropriate notifications
def handle_execution_result(module_name, success, execution_details=None):
    """Generate notifications based on test execution results."""

    if success:
        add_notification(
            module_name=module_name,
            status="success",
            message=f"Tests executed successfully in {module_name}",
            details=execution_details or "All tests passed without errors.",
            action_steps=["Review results to ensure expected behavior", "Consider adding more test coverage"]
        )
        # Update execution metrics
        st.session_state.execution_metrics['tests_executed'] += 1
        st.session_state.execution_metrics['successful_tests'] += 1
    else:
        # Get error information from execution details
        error_info = "Unknown error occurred"
        if execution_details:
            error_info = execution_details

        # Generate actionable steps based on error type
        action_steps = ["Review error details", "Check input parameters"]

        # Add more specific action steps based on error patterns
        if "timeout" in str(error_info).lower():
            action_steps.append("Increase timeout settings for network operations")
        elif "connection" in str(error_info).lower():
            action_steps.append("Check network connectivity")
            action_steps.append("Verify endpoint URLs are correct")
        elif "permission" in str(error_info).lower() or "access" in str(error_info).lower():
            action_steps.append("Check credentials and access permissions")
        elif "syntax" in str(error_info).lower():
            action_steps.append("Fix syntax errors in test code")
        elif "not found" in str(error_info).lower():
            action_steps.append("Verify file paths or resource identifiers")

        add_notification(
            module_name=module_name,
            status="error",
            message=f"Test execution failed in {module_name}",
            details=error_info,
            action_steps=action_steps
        )

        # Update execution metrics
        st.session_state.execution_metrics['tests_executed'] += 1
        st.session_state.execution_metrics['failed_tests'] += 1


# Function to mark a notification as read
def mark_notification_read(notification_id):
    """Mark a specific notification as read."""
    for notification in st.session_state.notifications:
        if notification["id"] == notification_id and not notification["read"]:
            notification["read"] = True
            st.session_state.unread_notifications -= 1
            return True
    return False


# Function to mark all notifications as read
def mark_all_notifications_read():
    """Mark all notifications as read."""
    for notification in st.session_state.notifications:
        if not notification["read"]:
            notification["read"] = True

    st.session_state.unread_notifications = 0

    # Update stored notifications
    try:
        notifications_file = os.path.join(script_dir, "gen_ai", "notifications.json")
        with open(notifications_file, 'w') as f:
            json.dump(st.session_state.notifications, f)
    except Exception:
        pass  # Silently fail if we can't save


# Function to clear notifications
def clear_notifications():
    """Clear all notifications."""
    st.session_state.notifications = []
    st.session_state.notification_count = 0
    st.session_state.unread_notifications = 0

    # Update stored notifications
    try:
        notifications_file = os.path.join(script_dir, "gen_ai", "notifications.json")
        with open(notifications_file, 'w') as f:
            json.dump(st.session_state.notifications, f)
    except Exception:
        pass  # Silently fail if we can't save


# Initialise session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'execution_metrics' not in st.session_state:
    # Initialise execution metrics with default values
    st.session_state.execution_metrics = {
        'tests_executed': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'execution_time': 0,
        'modules_usage': {'dynamic_tc_generation': 0, 'intelligent_test_data_generation': 0}
    }
if 'last_module' not in st.session_state:
    st.session_state.last_module = None
if 'last_saved_module' not in st.session_state:  # Track the last module saved to history
    st.session_state.last_saved_module = None

# Enhanced notification system
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'notification_count' not in st.session_state:
    st.session_state.notification_count = 0
if 'unread_notifications' not in st.session_state:
    st.session_state.unread_notifications = 0

if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# ============================================================================
# AUTHENTICATION & AUTHORIZATION CHECK
# ============================================================================
if AUTH_ENABLED and st_auth:
    import logging

    # Check if user is authenticated
    is_auth = st_auth.is_authenticated()
    logging.info(f"🔐 Authentication check: is_authenticated={is_auth}, session_id={st.session_state.get('auth_session_id')}")

    if not is_auth:
        logging.info("❌ User not authenticated - showing login page")
        # Show login page
        render_login_page(st_auth)
        st.stop()

    logging.info("✅ User authenticated successfully")

    # Check if password change is required
    if st.session_state.get('force_password_change', False):
        logging.info("🔑 Password change required - showing password change form")
        render_password_change_form(st_auth)
        st.stop()

    # Get current user for use throughout the app
    current_user = st_auth.get_current_user()
    logging.info(f"👤 Current user: {current_user.username if current_user else 'None'}")

    # Session cleanup (remove expired sessions)
    if 'last_session_cleanup' not in st.session_state:
        st.session_state.last_session_cleanup = datetime.now()

    # Cleanup every 5 minutes
    if (datetime.now() - st.session_state.last_session_cleanup).seconds > 300:
        st.session_state.auth_manager.cleanup_sessions()
        st.session_state.last_session_cleanup = datetime.now()
else:
    # Auth disabled - use demo mode
    current_user = None
    import logging
    logging.info("🔓 Authentication disabled - demo mode")

# Check if we have a pending module selection from navigation that needs to be saved to history
if 'pending_history_module' in st.session_state and st.session_state.pending_history_module:
    # This means a navigation happened but history wasn't saved yet
    module_to_save = st.session_state.pending_history_module
    if module_to_save != st.session_state.last_saved_module:
        save_usage_history(module_to_save)

# Note: RobotMCP initialization moved to AFTER admin panel check
# It will only initialize when user reaches the main homepage

# ============================================================================
# 🤖 ROBOTMCP STATUS DISPLAY (Only after successful login)
# ============================================================================
# Show RobotMCP status in sidebar only when user is authenticated on main page
# This prevents initialization during login/password change flows
if AUTH_ENABLED and current_user and not st.session_state.get('show_admin_panel', False) and not st.session_state.get('show_user_profile', False):
    try:
        from gen_ai.use_cases.test_pilot import (
            ROBOTMCP_AVAILABLE,
            _robotmcp_connection_pool,
            start_robotmcp_background_connection,
            get_robotmcp_helper
        )

        if ROBOTMCP_AVAILABLE:
            with st.sidebar:
                st.markdown("---")

                # Header with manual refresh button
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown("### 🤖 RobotMCP Status")
                with col2:
                    if st.button("🔄", key="refresh_robotmcp_global", help="Refresh status and reconnect if needed", use_container_width=True):
                        # Check connection status and reinitialize if disconnected
                        import logging
                        import time
                        current_status = _robotmcp_connection_pool.get('connection_status', 'disconnected')
                        current_helper = _robotmcp_connection_pool.get('helper')
                        bg_task = _robotmcp_connection_pool.get('background_task')

                        # Determine if reconnection is needed
                        need_reconnect = False

                        # Handle 'connecting' state - wait for connection to complete
                        if current_status == 'connecting':
                            logging.info("🔄 Refresh: Connection in progress, waiting for completion...")

                            # Wait up to 10 seconds for connection to complete
                            max_wait = 10
                            waited = 0
                            while waited < max_wait and current_status == 'connecting':
                                time.sleep(0.5)
                                waited += 0.5
                                current_status = _robotmcp_connection_pool.get('connection_status', 'disconnected')
                                current_helper = _robotmcp_connection_pool.get('helper')

                            # Check final status after waiting
                            if current_status == 'connected':
                                logging.info("✅ Refresh: Connection completed successfully")
                            elif current_status == 'connecting':
                                # Still connecting after timeout - check if thread is alive
                                thread_alive = bg_task.is_alive() if bg_task else False
                                if thread_alive:
                                    logging.info("⏳ Refresh: Connection still in progress (thread alive)")
                                else:
                                    logging.warning("⚠️ Refresh: Connection timeout - thread dead, will re-attempt")
                                    need_reconnect = True
                            else:
                                # Connection failed
                                logging.warning(f"⚠️ Refresh: Connection failed (status: {current_status}), will re-attempt")
                                need_reconnect = True

                        elif current_status in ['disconnected', 'error']:
                            # Status indicates problem - check helper state
                            if current_helper and hasattr(current_helper, 'is_connected'):
                                try:
                                    if not current_helper.is_connected:
                                        need_reconnect = True
                                        logging.warning("🔄 Refresh: Helper exists but not connected - will reconnect")
                                    else:
                                        logging.info("✅ Refresh: Helper is connected, auto-correcting status")
                                        _robotmcp_connection_pool['connection_status'] = 'connected'
                                except:
                                    need_reconnect = True
                                    logging.warning("🔄 Refresh: Error checking helper - will reconnect")
                            else:
                                need_reconnect = True
                                logging.warning("🔄 Refresh: No helper or status disconnected - will reconnect")

                        elif current_status == 'connected':
                            # Already connected - just verify
                            if current_helper and hasattr(current_helper, 'is_connected'):
                                try:
                                    if not current_helper.is_connected:
                                        logging.warning("⚠️ Refresh: Status 'connected' but helper not connected - will reconnect")
                                        need_reconnect = True
                                    else:
                                        logging.info("✅ Refresh: Connection verified as healthy")
                                except:
                                    logging.warning("⚠️ Refresh: Error verifying connection - will reconnect")
                                    need_reconnect = True
                            else:
                                logging.warning("⚠️ Refresh: Status 'connected' but no helper - will reconnect")
                                need_reconnect = True

                        # Trigger reconnection if needed
                        if need_reconnect:
                            logging.info("🔄 Refresh button: Reinitializing MCP server connection")
                            try:
                                # Reset state
                                st.session_state.robotmcp_prewarming_started = False
                                # Start new connection
                                start_robotmcp_background_connection()
                                st.session_state.robotmcp_prewarming_started = True
                                _robotmcp_connection_pool['connection_status'] = 'connecting'
                                logging.info("✅ Refresh: Reconnection initiated successfully")
                            except Exception as e:
                                logging.error(f"❌ Refresh: Reconnection failed: {e}")
                                _robotmcp_connection_pool['connection_status'] = 'error'

                        # Rerun to show updated status
                        st.rerun()

                status = _robotmcp_connection_pool.get('connection_status', 'disconnected')
                helper = _robotmcp_connection_pool.get('helper')  # Get directly from pool

                # ============================================================
                # AUTO-RECONNECTION LOGIC - Keep connection alive
                # ============================================================
                actual_connected = False
                needs_reconnection = False

                if helper is not None:
                    try:
                        actual_connected = helper.is_connected
                        # Auto-correct status if needed
                        if actual_connected and status in ['error', 'disconnected']:
                            _robotmcp_connection_pool['connection_status'] = 'connected'
                            status = 'connected'
                            import logging
                            logging.info("✅ Auto-corrected status to 'connected' (helper is connected)")
                        # If helper says disconnected but status says connected, mark for reconnection
                        elif not actual_connected and status == 'connected':
                            needs_reconnection = True
                            _robotmcp_connection_pool['connection_status'] = 'error'
                            status = 'error'
                            import logging
                            logging.warning("⚠️ Helper disconnected but status was 'connected' - will reconnect")
                    except:
                        # Error checking connection - assume disconnected
                        if status == 'connected':
                            needs_reconnection = True
                            _robotmcp_connection_pool['connection_status'] = 'error'
                            status = 'error'
                            import logging
                            logging.warning("⚠️ Error checking helper connection - will reconnect")
                else:
                    # Helper is None - check if status was 'connected' before marking as disconnected
                    # This prevents false disconnections when helper is still being created
                    if status == 'connected':
                        import logging
                        logging.warning("⚠️ Helper is None but status was 'connected' - may be loading, will check again")
                        # Don't immediately mark as disconnected - give it a chance
                        # Only mark for reconnection if it stays None
                        pass  # Status stays 'connected', will be checked next refresh

                # Check if background task died but connection isn't established
                bg_task = _robotmcp_connection_pool.get('background_task')
                if bg_task is not None:
                    try:
                        if not bg_task.is_alive() and status in ['connecting', 'disconnected'] and not actual_connected:
                            needs_reconnection = True
                    except:
                        pass

                # AUTO-RECONNECT if needed
                if needs_reconnection:
                    import logging
                    logging.info("🔄 Auto-reconnecting RobotMCP (connection lost)")
                    try:
                        # Reset state
                        st.session_state.robotmcp_prewarming_started = False
                        # Start new connection
                        start_robotmcp_background_connection()
                        st.session_state.robotmcp_prewarming_started = True
                        _robotmcp_connection_pool['connection_status'] = 'connecting'
                        status = 'connecting'
                    except Exception as e:
                        logging.error(f"Auto-reconnection failed: {e}")
                        _robotmcp_connection_pool['connection_status'] = 'error'
                        status = 'error'

                # Note: Aggressive health checks disabled for performance
                # Health checks now only happen on manual refresh button click

                # Display status
                if status == 'connecting':
                    st.info("Connecting...", icon="🔄")
                    if needs_reconnection:
                        st.caption("⚡ Auto-reconnecting...")
                    else:
                        st.caption("Establishing connection...")
                elif status == 'connected':
                    st.success("Connected & Ready", icon="✅")
                    st.caption("✓ Available to all modules • Auto-monitoring active")
                elif status == 'error':
                    st.warning("Connection Issue", icon="⚠️")
                    st.caption("Auto-reconnecting in background...")
                    # Manual reconnect button
                    if st.button("🔄 Reconnect Now", key="manual_reconnect_robotmcp", use_container_width=True):
                        st.session_state.robotmcp_prewarming_started = False
                        _robotmcp_connection_pool['connection_status'] = 'disconnected'
                        st.rerun()
                else:
                    st.info("Initializing...", icon="⏳")
                    st.caption("Starting in background...")

                # Comprehensive Debug Info
                with st.expander("🔍 Debug Info", expanded=False):
                    # Connection Pool Status
                    st.markdown("**📊 Connection Pool**")
                    st.caption(f"• Status: `{status}`")
                    st.caption(f"• Helper Instance: `{helper is not None}`")
                    st.caption(f"• Helper Connected: `{actual_connected}`")

                    # Helper Details
                    if helper is not None:
                        st.caption(f"• Helper Type: `{type(helper).__name__}`")
                        try:
                            if hasattr(helper, 'session'):
                                st.caption(f"• MCP Session: `{helper.session is not None}`")
                            if hasattr(helper, 'current_session_id'):
                                st.caption(f"• Session ID: `{helper.current_session_id}`")
                        except:
                            pass

                    st.markdown("---")

                    # Background Task Status
                    st.markdown("**🔄 Background Task**")
                    bg_task = _robotmcp_connection_pool.get('background_task')
                    if bg_task is not None:
                        try:
                            is_alive = bg_task.is_alive()
                            st.caption(f"• Thread Exists: `True`")
                            st.caption(f"• Thread Alive: `{is_alive}`")
                            if hasattr(bg_task, 'name'):
                                st.caption(f"• Thread Name: `{bg_task.name}`")
                        except Exception as e:
                            st.caption(f"• Thread Status: `Error - {str(e)[:50]}`")
                    else:
                        st.caption(f"• Thread Exists: `False`")

                    st.markdown("---")

                    # Health Check & Timestamps
                    st.markdown("**⏱️ Timestamps & Health**")
                    last_check = _robotmcp_connection_pool.get('last_health_check')
                    if last_check:
                        try:
                            import datetime as dt
                            if isinstance(last_check, dt.datetime):
                                time_ago = (dt.datetime.now() - last_check).total_seconds()
                                st.caption(f"• Last Health Check: `{int(time_ago)}s ago`")
                                st.caption(f"• Timestamp: `{last_check.strftime('%H:%M:%S')}`")
                            else:
                                st.caption(f"• Last Health Check: `{last_check}`")
                        except:
                            st.caption(f"• Last Health Check: `Unknown`")
                    else:
                        st.caption(f"• Last Health Check: `Never`")

                    # Next health check countdown
                    if 'robotmcp_last_health_check' in st.session_state:
                        import time
                        time_since_check = time.time() - st.session_state.robotmcp_last_health_check
                        next_check_in = max(0, 60 - int(time_since_check))
                        st.caption(f"• Next Health Check: `{next_check_in}s`")

                    # Auto-reconnection status
                    st.caption(f"• Auto-Reconnect: `{'Active' if needs_reconnection else 'Standby'}`")
                    st.caption(f"• Health Check: `Disabled for performance`")
                    st.caption(f"• Keep-Alive: `Active (prevents session timeout)`")

                    st.markdown("---")

                    # Session State
                    st.markdown("**💾 Session State**")
                    st.caption(f"• Global Init: `{st.session_state.get('robotmcp_global_init', False)}`")
                    st.caption(f"• Pre-warming Started: `{st.session_state.get('robotmcp_prewarming_started', False)}`")

                    st.markdown("---")

                    # Connection Lock
                    st.markdown("**🔒 Connection Lock**")
                    conn_lock = _robotmcp_connection_pool.get('connection_lock')
                    st.caption(f"• Lock Exists: `{conn_lock is not None}`")
                    if conn_lock is not None:
                        st.caption(f"• Lock Type: `{type(conn_lock).__name__}`")

                    st.markdown("---")

                    # Raw Pool Data
                    st.markdown("**📦 Raw Connection Pool**")
                    pool_keys = list(_robotmcp_connection_pool.keys())
                    st.caption(f"• Keys: `{', '.join(pool_keys)}`")
                    st.caption(f"• Total Entries: `{len(pool_keys)}`")
    except ImportError:
        # RobotMCP not available - skip status display
        pass
    except Exception as e:
        # Any error - skip status display
        pass


# Function to handle feedback submission
def submit_feedback(rating, comments):
    """Save user feedback to session state and optionally to a file."""
    if rating == "Select rating":
        return False, "Please select a rating"

    # Create feedback entry
    feedback_entry = {
        'rating': rating,
        'comments': comments,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'module': st.session_state.last_module  # Track which module was active
    }

    # Add to session state
    st.session_state.feedback_data.append(feedback_entry)

    # Try to save to a file for persistence
    try:
        feedback_file = os.path.join(script_dir, "gen_ai", "user_feedback.json")

        # Load existing feedback if file exists
        existing_feedback = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                try:
                    existing_feedback = json.load(f)
                except json.JSONDecodeError:
                    existing_feedback = []

        # Append new feedback and save
        existing_feedback.append(feedback_entry)
        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f)

        return True, "Thank you for your feedback!"
    except Exception as e:
        # Still consider it successful even if file save fails
        return True, f"Thanks for your feedback! (Note: {str(e)})"


# Create a module for accessing notification functions from other modules
def create_notifications_module():
    """Create a module that exposes the notification functions to other modules."""
    import sys
    import types

    # Create a new module
    notifications_module = types.ModuleType('notifications')

    # Add the notification functions to the module
    notifications_module.add_notification = add_notification
    notifications_module.handle_execution_result = handle_execution_result
    notifications_module.mark_notification_read = mark_notification_read
    notifications_module.mark_all_notifications_read = mark_all_notifications_read
    notifications_module.clear_notifications = clear_notifications

    # Add the module to sys.modules so it can be imported by other modules
    sys.modules['notifications'] = notifications_module


# Initialize the notifications module so it can be imported by other modules
create_notifications_module()


# Function to get module-friendly names
def get_module_friendly_name(module_id):
    """Convert internal module IDs to user-friendly display names."""
    module_names = {
        "dynamic_tc_generation": "Dynamic Test Cases",
        "intelligent_test_data_generation": "Intelligent Test Data",
        "self_healing_tests": "Self-Healing Tests",
        "visual_ai_testing": "Visual AI Testing",
        "api_generation": "API Generation",
        "auto_documentation": "Auto Documentation",
        "performance_testing": "Performance Testing",
        "robocop_lint_checker": "RoboCop Lint Checker",
        "smart_cx_navigator": "Smart CX Navigator",
        "security_penetration_testing": "Security Penetration Testing",
        "pull_requests_reviewer": "Pull Requests Reviewer",
        "database_insights": "Database Insights",
        "jenkins_dashboard": "Jenkins Dashboard",
        "rf_dashboard_analytics": "RF Dashboard Analytics",
        "fos_checks": "FOS Quality Checks",
        "intelligent_bug_predictor": "AI Bug Predictor",
        "smart_test_optimizer": "Smart Test Optimizer",
        "ai_cross_platform_orchestrator": "Cross-Platform Orchestrator",
        "ai_test_environment_manager": "AI Environment Manager",
        "manual_test_analysis": "Manual Test Analyzer",
        "ai_test_execution_orchestrator": "AI Execution Orchestrator",
        "ai_quality_assurance_guardian": "AI Quality Guardian",
        "browser_agent": "Browser Agent",
        "test_pilot": "TestPilot",
        "edb_query_manager": "EDB Query Manager",
        "newfold_migration_toolkit": "Newfold Migration Toolkit"
    }
    return module_names.get(module_id, module_id)  # Return the ID itself if no mapping exists


# Function to get execution suggestions based on past usage
def get_suggestions():
    if not st.session_state.history:
        return []

    # Simple algorithm: suggest most used modules
    module_counts = {}
    for entry in st.session_state.history:
        module_counts[entry['module']] = module_counts.get(entry['module'], 0) + 1

    # Sort by usage count
    suggestions = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
    return [module for module, _ in suggestions[:3]]


# Helper function to generate dynamic insights
def generate_insights():
    insights = []

    if st.session_state.execution_metrics['tests_executed'] > 0:
        success_rate = (st.session_state.execution_metrics['successful_tests'] /
                        st.session_state.execution_metrics['tests_executed']) * 100

        if success_rate < 70:
            insights.append(
                "Success rate is below 70%. Consider reviewing your test cases for common failure patterns.")
        else:
            insights.append(f"Great job! Your test success rate is {success_rate:.1f}%.")

    if st.session_state.history:
        # Get the most used module identifier
        most_used = max(st.session_state.execution_metrics['modules_usage'].items(), key=lambda x: x[1])

        # Get the user-friendly name, or fallback to the identifier if not found
        friendly_name = get_module_friendly_name(most_used[0])
        insights.append(f"You use '{friendly_name}' most frequently. Consider exploring other modules too.")

    return insights


# Main header with enhanced design and larger buttons
st.markdown("""
<style>
/* Aggressively reduce top padding/margin to move hero section to the very top */
.main .block-container {
    padding-top: 0.25rem !important;
    padding-bottom: 1rem !important;
    margin-top: 0 !important;
}

/* Remove all spacing from the main content area */
section.main > div {
    padding-top: 0 !important;
}

/* Reduce spacing in the main content block */
div[data-testid="stVerticalBlock"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove gap between stacked elements */
div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
    margin-top: 0 !important;
}

/* Enhanced header button styling */
div[data-testid="column"] button[kind="secondary"] {
    font-size: 2rem !important;
    padding: 1rem 1.5rem !important;
    min-height: 60px !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #EC5328, #ff6b6b) !important;
    color: white !important;
    border: 2px solid rgba(255,255,255,0.2) !important;
    box-shadow: 0 4px 12px rgba(236, 83, 40, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="column"] button[kind="secondary"]:hover {
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 0 8px 20px rgba(236, 83, 40, 0.5) !important;
    border: 2px solid white !important;
}

div[data-testid="column"] button[kind="secondary"]:active {
    transform: translateY(-2px) scale(1.02) !important;
}

/* Better alignment for header columns */
div[data-testid="column"] > div {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# USER MENU & NAVIGATION (Top Right)
# ============================================================================
# Only show navigation when on main page (not in admin panel or profile)
if AUTH_ENABLED and current_user and not st.session_state.get('show_admin_panel', False) and not st.session_state.get('show_user_profile', False):
    # Create a top navigation bar with user info
    nav_col1, nav_col2, nav_col3 = st.columns([3, 1, 1])

    with nav_col1:
        st.markdown(f"""
        <div style="padding: 0.5rem; color: #475569; font-size: 0.95rem;">
            👤 <strong>{current_user.full_name or current_user.username}</strong> 
            <span style="color: #94a3b8;">({', '.join(current_user.roles)})</span>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        # Check if user has admin role (super_admin or admin only)
        has_admin_access = st_auth.is_admin()
        if has_admin_access:
            if st.button("🔐 Admin Panel", key="admin_panel_btn", use_container_width=True):
                import logging
                logging.info("🔐 Admin Panel button clicked - setting show_admin_panel=True")
                # Clear conflicting navigation states
                st.session_state.show_user_profile = False
                st.session_state.show_admin_panel = True
                st.session_state.user_menu_previous = "🧭 Menu"
                logging.info(f"🔐 State after setting: show_admin_panel={st.session_state.show_admin_panel}")
                st.rerun()

    with nav_col3:
        # Initialize user menu state to prevent auto-redirect
        if 'user_menu_previous' not in st.session_state:
            st.session_state.user_menu_previous = "🧭 Menu"

        user_menu = st.selectbox(
            "User Menu",
            ["🧭 Menu", "👤 Profile", "🚪 Logout"],
            key="user_menu_select",
            label_visibility="collapsed",
            index=0  # Always default to Menu
        )

        # Only trigger action if user menu selection has changed and is not the default
        if user_menu != "🧭 Menu" and user_menu != st.session_state.user_menu_previous:
            st.session_state.user_menu_previous = user_menu

            if user_menu == "🚪 Logout":
                st_auth.logout()
                # Clear all session state on logout
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            elif user_menu == "👤 Profile":
                st.session_state.show_user_profile = True
                st.rerun()
        elif user_menu == "🧭 Menu":
            # Reset previous if user goes back to menu
            st.session_state.user_menu_previous = "🧭 Menu"

# Check if admin panel should be shown
if AUTH_ENABLED and st.session_state.get('show_admin_panel', False):
    import logging
    logging.info("🔐 Rendering Admin Panel (show_admin_panel=True)")

    # SECURITY: Double-check user has admin role before rendering
    if not st_auth.is_admin():
        logging.warning(f"🚨 Unauthorized admin panel access attempt by user: {st_auth.get_current_user().username if st_auth.get_current_user() else 'unknown'}")
        st.error("⛔ Access Denied: Admin privileges required.")
        st.session_state.show_admin_panel = False
        st.rerun()

    if st.button("⬅️ Back to Main", key="back_from_admin"):
        logging.info("⬅️ Back button clicked - clearing show_admin_panel")
        st.session_state.show_admin_panel = False
        # Reset user menu to default
        st.session_state.user_menu_previous = "🧭 Menu"
        st.rerun()
    render_admin_panel(st_auth)
    logging.info("🔐 Admin panel rendered - calling st.stop()")
    st.stop()

# Check if user profile should be shown
if AUTH_ENABLED and st.session_state.get('show_user_profile', False):
    import logging
    logging.info("👤 Rendering User Profile (show_user_profile=True)")
    if st.button("⬅️ Back to Main", key="back_from_profile"):
        logging.info("⬅️ Back button clicked - clearing show_user_profile")
        st.session_state.show_user_profile = False
        # Reset user menu to default
        st.session_state.user_menu_previous = "🧭 Menu"
        st.rerun()
    render_user_profile(st_auth)
    logging.info("👤 User profile rendered - calling st.stop()")
    st.stop()

# If we reach here, we're on the main homepage
logger.info("🌀 Rendering main homepage (not admin panel or profile)")

# ============================================================================
# 🚀 ROBOTMCP INITIALIZATION - Only on main homepage
# ============================================================================
# Initialize RobotMCP ONLY when user is on main homepage (not admin panel/profile)
# This ensures it doesn't initialize during intermediate navigation
if AUTH_ENABLED and current_user:
    if 'robotmcp_global_init' not in st.session_state:
        st.session_state.robotmcp_global_init = False

    if not st.session_state.robotmcp_global_init:
        try:
            # Import RobotMCP initialization from test_pilot
            from gen_ai.use_cases.test_pilot import (
                ROBOTMCP_AVAILABLE,
                start_robotmcp_background_connection,
                _robotmcp_connection_pool
            )

            if ROBOTMCP_AVAILABLE:
                # Start connection in background (non-blocking)
                start_robotmcp_background_connection()
                st.session_state.robotmcp_global_init = True
                import logging
                logging.info("🚀 RobotMCP initialized on main homepage")
        except ImportError:
            # RobotMCP not available - silently continue
            st.session_state.robotmcp_global_init = True
        except Exception as e:
            # Any other error - log and continue
            st.session_state.robotmcp_global_init = True
            import logging
            logging.debug(f"RobotMCP initialization skipped: {e}")

# Hero Section - The Vortex
st.markdown("""
<div style="
    background: linear-gradient(135deg, #EC5328 0%, #ff6b6b 50%, #EC5328 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(236, 83, 40, 0.35);
    border: 2px solid rgba(255, 255, 255, 0.2);
    margin-top: 0;
    margin-bottom: 0.5rem;
">
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 2.5rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">🌀</span>
        <h1 style="
            font-family: 'Inter', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            margin: 0;
            color: white;
            letter-spacing: -0.02em;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        ">The Vortex</h1>
    </div>
    <p style="
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        color: rgba(255,255,255,0.95);
        margin: 0 0 0.75rem 0;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    ">
        <u>V</u>irtual <u>O</u>rchestrator for <u>R</u>eal-world <u>T</u>echnology <u>EX</u>cellence
    </p>
    <div style="
        height: 2px;
        background: rgba(255,255,255,0.3);
        margin: 0.75rem auto;
        width: 50%;
        border-radius: 2px;
    "></div>
    <p style="
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: rgba(255,255,255,0.98);
        margin: 0.75rem 0 0 0;
        line-height: 1.6;
        font-weight: 400;
    ">
        Your AI-powered portal for intelligent test automation, quality assurance,<br>
        and continuous testing excellence. Select a module below to begin your journey.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Buttons - Centered below the hero section
st.markdown("""
<style>
.nav-button-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    margin: 0 auto 0.75rem auto;
    max-width: 600px;
}

/* Remove extra spacing around navigation columns */
div[data-testid="column"]:has(button[key="home_button"]),
div[data-testid="column"]:has(button[key="history_button"]),
div[data-testid="column"]:has(button[key="notification_button"]),
div[data-testid="column"]:has(button[key="settings_button"]) {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# First row - Main navigation
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6, nav_col7 = st.columns([2, 1.2, 1.2, 1.2, 1.2, 2, 0.1])
with nav_col2:
    if st.button("🏠 Home", help="Go to Homepage", key="home_button", use_container_width=True):
        st.session_state.last_module = None
        if "module" in st.query_params:
            del st.query_params["module"]
        st.rerun()
with nav_col3:
    if st.button("📋 Usage History", help="View Usage History", key="history_button", use_container_width=True):
        st.session_state.show_history = not st.session_state.get('show_history', False)
with nav_col4:
    notification_count = st.session_state.notification_count
    notification_label = f"🔔 Notifications ({notification_count})" if notification_count > 0 else "🔔 Notifications"
    if st.button(notification_label, help="View Notifications", key="notification_button", use_container_width=True):
        st.session_state.show_notifications = not st.session_state.get('show_notifications', False)
with nav_col5:
    if st.button("⚙️ Settings", help="Configuration Settings", key="settings_button", use_container_width=True):
        st.session_state.show_settings = not st.session_state.get('show_settings', False)

# Second row - Info & Guide buttons
info_col1, info_col2, info_col3, info_col4, info_col5, info_col6, info_col7 = st.columns([0.5, 1.3, 1.3, 1.3, 1.3, 1.3, 0.5])
with info_col2:
    if st.button("🚀 Quick Start", help="Get Started in 3 Easy Steps", key="quickstart_button", use_container_width=True):
        st.session_state.show_quickstart = not st.session_state.get('show_quickstart', False)
with info_col3:
    if st.button("📚 Modules", help="Complete Module List", key="overview_button", use_container_width=True):
        st.session_state.show_overview = not st.session_state.get('show_overview', False)
with info_col4:
    if st.button("✨ Features", help="Feature Highlights", key="features_button", use_container_width=True):
        st.session_state.show_features = not st.session_state.get('show_features', False)
with info_col5:
    if st.button("💡 Tips", help="Tips & Best Practices", key="tips_button", use_container_width=True):
        st.session_state.show_tips = not st.session_state.get('show_tips', False)
with info_col6:
    insights_available = len(st.session_state.history) > 2
    insights_label = "🔮 AI Insights" if insights_available else "🔮 AI Insights (N/A)"
    insights_help = "AI-Powered Insights & Recommendations" if insights_available else "Requires 3+ activities to generate insights"
    if st.button(insights_label, help=insights_help, key="insights_button", use_container_width=True, disabled=not insights_available):
        st.session_state.show_insights = not st.session_state.get('show_insights', False)

# Quick Start Guide - Show on button click
if st.session_state.get('show_quickstart', False):
    with st.expander("🚀 Quick Start Guide - Get Started in 3 Easy Steps", expanded=True):
        quick_col1, quick_col2, quick_col3 = st.columns(3)

        with quick_col1:
            st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-left: 6px solid #EC5328;
            border-radius: 16px;
            padding: 2rem;
            height: 280px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1rem;
                text-align: center;
            ">Step 1: Choose Your Module</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                color: #64748b;
                line-height: 1.6;
                text-align: center;
            ">
                Browse through our organized categories below and select the module that fits your testing needs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with quick_col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-left: 6px solid #6366f1;
            border-radius: 16px;
            padding: 2rem;
            height: 280px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚙️</div>
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1rem;
                text-align: center;
            ">Step 2: Configure & Execute</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                color: #64748b;
                line-height: 1.6;
                text-align: center;
            ">
                Use the intuitive interface to configure your tests and let AI handle the complexity.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with quick_col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-left: 6px solid #10b981;
            border-radius: 16px;
            padding: 2rem;
            height: 280px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1rem;
                text-align: center;
            ">Step 3: Analyze Results</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                color: #64748b;
                line-height: 1.6;
                text-align: center;
            ">
                Review AI-powered insights, metrics, and recommendations for continuous improvement.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Detailed Module Overview - Complete List - Show on button click
if st.session_state.get('show_overview', False):
    with st.expander("📚 Detailed Module Overview - Complete List", expanded=True):
        st.markdown("""
    ### 🧪 Test Generation & Intelligence
    - **TestPilot**: AI-powered assistant that fetches test cases from Jira/Zephyr and converts them to Robot Framework scripts
    - **Dynamic Test Cases**: Generate comprehensive test cases from requirements and specifications
    - **Intelligent Test Data**: Create realistic and edge-case test data automatically
    - **API Generation**: Generate API tests from OpenAPI/Swagger specifications

    ### 🔧 Test Maintenance & Quality
    - **Self-Healing Tests**: Automatically repair broken tests when UI elements change
    - **Visual AI Testing**: Perform AI-powered visual regression testing and UI analysis
    - **RoboCop Lint Checker**: Static code analysis for Robot Framework to enforce coding standards
    - **Smart Test Optimizer**: Reduce test execution time by up to 70% with AI optimization
    - **FOS Quality Checks**: Comprehensive front-of-site quality assurance

    ### 🚀 Automation & Integration
    - **Cross-Platform Orchestrator**: Intelligent orchestration across browsers, devices, and platforms
    - **Performance Testing**: Load, stress, and performance testing capabilities
    - **Security Penetration Testing**: Automated security vulnerability scanning and OWASP checks
    - **Browser Agent**: AI-powered browser automation for complex workflows

    ### 📊 DevOps & Monitoring
    - **RF Dashboard Analytics**: AI-powered insights from Robot Framework test results with trend analysis
    - **Jenkins Dashboard**: Integration with Jenkins for CI/CD pipeline monitoring
    - **EDB Query Manager**: Comprehensive EDB account management with AI-powered insights
    - **Database Insights**: Database optimization and performance recommendations
    - **AI Environment Manager**: Smart test environment provisioning and data management
    - **Newfold Migration Toolkit**: CSRT and RAS operations for product lifecycle management

    ### 📝 Analysis & Documentation
    - **Auto Documentation**: Generate comprehensive test documentation automatically
    - **Smart CX Navigator**: AI-driven customer experience optimization and insights
    - **Pull Requests Reviewer**: AI-powered code review for quality and compliance
    - **Manual Test Analyzer**: Transform manual testing with AI recommendations

    ### 🔮 Predictive Intelligence
    - **AI Bug Predictor**: Predict potential bugs before they occur in production
    - **AI Execution Orchestrator**: Intelligent test execution with priority-based scheduling
    - **AI Quality Guardian**: Comprehensive quality assurance with ML-powered insights
    """)

    # Add a dashboard or recent activity section
    st.markdown("### 📊 Recent Activity")
    if st.session_state.history:
        st.markdown("Here are your last 5 activities:")
        # Display the last 5 activities in a table
        recent_activities = st.session_state.history[-5:]
        recent_df = pd.DataFrame(recent_activities)
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
        recent_df = recent_df.sort_values(by='timestamp', ascending=False)
        recent_df = recent_df[['module', 'timestamp']]
        recent_df.columns = ['Module', 'Timestamp']
        recent_df['Timestamp'] = recent_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        recent_df.reset_index(drop=True, inplace=True)
        st.markdown("#### Last 5 Activities")
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.markdown('<div style="background-color: var(--primary-color); padding: 10px; border-radius: var(--border-radius-md); color: white;">No recent activity. Start by clicking on one of the use case options above.</div>', unsafe_allow_html=True)

# Feature Highlights - Show on button click
if st.session_state.get('show_features', False):
    with st.expander("✨ Feature Highlights - Discover What The Vortex Can Do", expanded=True):
        feat_col1, feat_col2, feat_col3 = st.columns(3)

        with feat_col1:
            st.markdown("""
            <div style="
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                height: 180px;
                display: flex;
                flex-direction: column;
            ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #EC5328;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">🤖 AI-Driven Automation</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                Leverage advanced AI models to automatically generate, maintain, and optimize your test suites.
            </p>
        </div>
        
        <div style="
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 180px;
            display: flex;
            flex-direction: column;
        ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #6366f1;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">🔧 Self-Healing Capabilities</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                Tests automatically adapt to UI changes, reducing maintenance overhead by up to 80%.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
        <div style="
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 180px;
            display: flex;
            flex-direction: column;
        ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #10b981;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">📊 Real-Time Analytics</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                Get instant insights with AI-powered dashboards that predict failures and suggest optimizations.
            </p>
        </div>
        
        <div style="
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 180px;
            display: flex;
            flex-direction: column;
        ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #f59e0b;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">🚀 Cross-Platform Testing</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                Seamlessly test across web, mobile, desktop, and API platforms with intelligent orchestration.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col3:
        st.markdown("""
        <div style="
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 180px;
            display: flex;
            flex-direction: column;
        ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #8b5cf6;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">🔮 Predictive Intelligence</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                AI predicts potential bugs and quality issues before they reach production.
            </p>
        </div>
        
        <div style="
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 180px;
            display: flex;
            flex-direction: column;
        ">
            <h4 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                color: #ec4899;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">🏢 Enterprise Ready</h4>
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                color: #64748b;
                line-height: 1.6;
                margin: 0;
            ">
                Built with scalability, security, and observability in mind for enterprise deployments.
            </p>
        </div>
        """, unsafe_allow_html=True)


# Display history if toggled
if st.session_state.get('show_history', False):
    with st.expander("📊 Usage History", expanded=True):
        if st.session_state.history:
            # Apply max_history limit
            max_history = st.session_state.get('max_history', 50)
            limited_history = st.session_state.history[-max_history:] if len(st.session_state.history) > max_history else st.session_state.history

            history_df = pd.DataFrame(limited_history)
            st.dataframe(history_df, use_container_width=True)

            if len(st.session_state.history) > max_history:
                st.caption(f"Showing latest {max_history} of {len(st.session_state.history)} total entries. Adjust in Settings to show more.")

            # Display usage chart
            if len(st.session_state.execution_metrics['modules_usage']) > 0 and PLOTLY_AVAILABLE:
                module_usage = pd.DataFrame({
                    'Modules': st.session_state.execution_metrics['modules_usage'].keys(),
                    'Usage Counts': st.session_state.execution_metrics['modules_usage'].values()
                })
                fig = px.bar(
                    module_usage,
                    x='Modules',
                    y='Usage Counts',
                    title="Module Usage Statistics",
                    color_discrete_sequence=["#EC5328"]  # Use --primary-color
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div style="background-color: var(--primary-color); padding: 10px; border-radius: var(--border-radius-md); color: white;">No usage history yet</div>', unsafe_allow_html=True)

# Display notifications if toggled with enhanced styling
if st.session_state.get('show_notifications', False):
    with st.expander("🔔 Notifications Center", expanded=True):
        if len(st.session_state.notifications) > 0:
            # Add management buttons at the top
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Mark All as Read"):
                    mark_all_notifications_read()
                    st.rerun()
            with col2:
                if st.button("Clear All Notifications"):
                    clear_notifications()
                    st.rerun()

            # Display notifications sorted by timestamp (newest first)
            sorted_notifications = sorted(st.session_state.notifications,
                                          key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"),
                                          reverse=True)

            # Group notifications by module
            module_notifications = {}
            for notification in sorted_notifications:
                module = notification['module']
                if module not in module_notifications:
                    module_notifications[module] = []
                module_notifications[module].append(notification)

            # Loop through modules
            for module, notifications in module_notifications.items():
                module_name = get_module_friendly_name(module)
                st.markdown(f"### {module_name}")

                # Display notifications for this module
                for notification in notifications:
                    # Choose card style based on notification status
                    card_class = f"notification-card notification-{notification['status']}"
                    if notification['read']:
                        card_class += " notification-read"

                    # Status icon
                    status_icon = "✅" if notification['status'] == "success" else "⚠️" if notification[
                                                                                              'status'] == "warning" else "❌"

                    # Create notification card
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="notification-time">{notification['timestamp']}</div>
                        <div class="notification-title">{status_icon} {notification['message']}</div>
                        <span class="notification-module">{notification['module_display_name']}</span>
                    """, unsafe_allow_html=True)

                    # Add details if present
                    if notification['details']:
                        st.markdown(f"""
                        <div class="notification-details">
                            {notification['details']}
                        </div>
                        """, unsafe_allow_html=True)

                    # Add action steps if present
                    if notification['action_steps']:
                        st.markdown("<div class=\"notification-actions\">", unsafe_allow_html=True)
                        for step in notification['action_steps']:
                            st.markdown(f"<div class=\"action-step\">{step}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Close the notification card
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Add a mark as read button if notification is unread
                    if not notification['read']:
                        if st.button(f"Mark as Read", key=f"read_{notification['id']}"):
                            mark_notification_read(notification['id'])
                            st.rerun()

                st.markdown("---")
        else:
            st.markdown('<div style="background-color: var(--primary-color); padding: 10px; border-radius: var(--border-radius-md); color: white;">No notifications yet. Notifications will appear here when tests are executed.</div>', unsafe_allow_html=True)

# AI Insights section - Show on button click
if st.session_state.get('show_insights', False) and len(st.session_state.history) > 2:
    insights = generate_insights()
    if insights:
        with st.expander("💡 AI-Powered Insights & Recommendations", expanded=True):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(236, 83, 40, 0.05) 0%, rgba(255, 107, 107, 0.05) 100%);
                border: 2px solid rgba(236, 83, 40, 0.2);
                border-left: 6px solid #EC5328;
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 4px 12px rgba(236, 83, 40, 0.1);
                position: relative;
            ">
                <div style="
                    position: absolute;
                    top: 1rem;
                    right: 1rem;
                    font-size: 2.5rem;
                    opacity: 0.2;
                ">💡</div>
                <p style="
                    font-family: 'Inter', sans-serif;
                    font-size: 0.9rem;
                    color: #64748b;
                    margin-bottom: 1rem;
                ">Based on your usage patterns and test execution history</p>
            </div>
            """, unsafe_allow_html=True)

            for idx, insight in enumerate(insights):
                st.markdown(f"""
                <div style="
                    font-family: 'Inter', sans-serif;
                    font-size: 1rem;
                    color: #1e293b;
                    padding: 1rem 1.25rem;
                    background: white;
                    border-radius: 10px;
                    margin-bottom: 0.75rem;
                    border-left: 4px solid #EC5328;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    transition: all 0.3s ease;
                ">
                    <div style="display: flex; align-items: flex-start; gap: 1rem;">
                        <div style="
                            background: linear-gradient(135deg, #EC5328, #ff6b6b);
                            color: white;
                            width: 28px;
                            height: 28px;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: 700;
                            font-size: 0.85rem;
                            flex-shrink: 0;
                        ">{idx + 1}</div>
                        <div style="flex: 1; padding-top: 2px;">{insight}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# Display Settings if toggled
if st.session_state.get('show_settings', False):
    with st.expander("⚙️ Configuration Settings", expanded=True):

        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            st.subheader("🎨 Display Preferences")

            # Theme settings with proper index handling
            theme_options = ["Auto (System)", "Light", "Dark"]
            current_theme = st.session_state.get('theme_mode', 'Auto (System)')
            theme_index = theme_options.index(current_theme) if current_theme in theme_options else 0

            theme_option = st.selectbox(
                "Theme Mode",
                theme_options,
                index=theme_index,
                help="Choose your preferred theme",
                key="theme_selector"
            )

            # Update theme in session state
            if theme_option != st.session_state.get('theme_mode', 'Auto (System)'):
                st.session_state.theme_mode = theme_option
                st.info(f"Theme changed to: {theme_option}. Click 'Save Settings' to apply.")

            # Compact mode
            compact_mode = st.checkbox(
                "Compact Mode",
                value=st.session_state.get('compact_mode', False),
                help="Reduce spacing for more content on screen",
                key="compact_mode_checkbox"
            )
            st.session_state.compact_mode = compact_mode

            # Show tooltips
            show_tooltips = st.checkbox(
                "Show Tooltips",
                value=st.session_state.get('show_tooltips', True),
                help="Display helpful tooltips on hover",
                key="tooltips_checkbox"
            )
            st.session_state.show_tooltips = show_tooltips

            st.subheader("🔔 Notification Settings")

            # Auto-show notifications
            auto_notifications = st.checkbox(
                "Auto-show Notifications",
                value=st.session_state.get('auto_notifications', True),
                help="Automatically display notifications panel after test execution",
                key="auto_notif_checkbox"
            )
            st.session_state.auto_notifications = auto_notifications

            # Notification sound
            notification_sound = st.checkbox(
                "Notification Sound",
                value=st.session_state.get('notification_sound', False),
                help="Play sound when new notifications arrive",
                key="notif_sound_checkbox"
            )
            st.session_state.notification_sound = notification_sound

            if notification_sound:
                st.caption("🔊 Sound will play when tests complete")

        with settings_col2:
            st.subheader("⚡ Performance Settings")

            # Cache duration
            cache_duration = st.slider(
                "Cache Duration (hours)",
                min_value=1,
                max_value=24,
                value=st.session_state.get('cache_duration', 12),
                help="How long to cache test data",
                key="cache_slider"
            )
            st.session_state.cache_duration = cache_duration

            # Max history items
            max_history = st.slider(
                "Max History Items",
                min_value=10,
                max_value=100,
                value=st.session_state.get('max_history', 50),
                step=10,
                help="Maximum number of history items to keep",
                key="history_slider"
            )
            st.session_state.max_history = max_history

            st.subheader("🔐 Security & Privacy")

            # Auto-logout
            auto_logout = st.checkbox(
                "Auto-logout on Inactivity",
                value=st.session_state.get('auto_logout', False),
                help="Automatically logout after period of inactivity",
                key="auto_logout_checkbox"
            )
            st.session_state.auto_logout = auto_logout

            if auto_logout:
                logout_duration = st.slider(
                    "Inactivity Timeout (minutes)",
                    min_value=5,
                    max_value=60,
                    value=st.session_state.get('logout_duration', 30),
                    step=5,
                    key="logout_slider"
                )
                st.session_state.logout_duration = logout_duration

        # Action buttons
        action_col1, action_col2, action_col3, action_col4 = st.columns([1, 1, 1, 3])

        with action_col1:
            if st.button("💾 Save Settings", use_container_width=True, key="save_settings_btn"):
                st.success("✅ Settings saved successfully!")
                st.rerun()

        with action_col2:
            if st.button("🔄 Reset to Default", use_container_width=True, key="reset_settings_btn"):
                # Reset to defaults
                st.session_state.theme_mode = 'Auto (System)'
                st.session_state.compact_mode = False
                st.session_state.show_tooltips = True
                st.session_state.auto_notifications = True
                st.session_state.notification_sound = False
                st.session_state.cache_duration = 12
                st.session_state.max_history = 50
                st.session_state.auto_logout = False
                st.session_state.logout_duration = 30
                st.success("✅ Settings reset to defaults!")
                st.rerun()

        with action_col3:
            if st.button("🗑️ Clear Cache", use_container_width=True, key="clear_cache_btn"):
                st.cache_data.clear()
                st.success("✅ Cache cleared successfully!")

# Apply theme mode settings dynamically
if st.session_state.get('theme_mode') == 'Light':
    st.markdown("""
    <style>
    /* Force Light Mode with Enhanced Contrast */
    :root, [data-theme="light"] {
        --background-color: #ffffff;
        --secondary-background-color: #f8fafc;
        --text-color: #1e293b;
        color-scheme: light;
    }
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Ensure all text has proper contrast in light mode */
    body, p, span, div, label, li, td, th, a, button {
        color: #1e293b !important;
    }
    
    /* Streamlit specific elements */
    .stMarkdown, .stText {
        color: #1e293b !important;
    }
    
    /* Headings with proper contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    
    /* Links with proper contrast */
    a {
        color: #DC2626 !important;
    }
    a:hover {
        color: #B91C1C !important;
    }
    
    /* Input fields and form elements */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-color: #cbd5e1 !important;
    }
    
    /* Buttons with proper contrast */
    button[kind="secondary"] {
        color: #1e293b !important;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] summary {
        color: #1e293b !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    
    /* Secondary text - using darker slate for better contrast */
    .caption, [data-testid="stCaption"] {
        color: #475569 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
    }
    
    /* Tables */
    table {
        color: #1e293b !important;
    }
    th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Dataframes */
    .dataframe {
        color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply compact mode settings
if st.session_state.get('compact_mode', False):
    st.markdown("""
    <style>
    /* Compact Mode - Reduce all spacing */
    .main .block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 0.5rem !important;
    }
    div[data-testid="stExpander"] {
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    p {
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply tooltips settings
if not st.session_state.get('show_tooltips', True):
    st.markdown("""
    <style>
    /* Hide all tooltips when disabled */
    div[data-testid="stTooltipIcon"] {
        display: none !important;
    }
    [title]:hover::after {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Quick Tips Section - Show on button click
if st.session_state.get('show_tips', False):
    with st.expander("💡 Quick Tips & Best Practices", expanded=True):
        tips_col1, tips_col2 = st.columns(2)

        with tips_col1:
            st.markdown("""
            ### 🎯 Navigation Tips
            - **Use Search**: Press `Ctrl+F` to quickly find modules
            - **Categories**: Browse by category for organized discovery
            - **Suggested Modules**: Look for 🌟 badges on frequently used modules
            - **Recent Activity**: Check your usage history to revisit modules
            
            ### ⚡ Productivity Hacks
            - **Favorites**: Star your favorite modules for quick access
            - **Keyboard Shortcuts**: Use `Tab` to navigate between elements
            - **Quick Start**: Expand the guide above for step-by-step help
            """)

        with tips_col2:
            st.markdown("""
            ### 🔔 Notifications
            - Click the 🔔 icon to view test execution results
            - Notifications show success/failure status with actionable steps
            - Mark notifications as read to keep your inbox clean
            
            ### 📊 Metrics
            - Monitor your test execution metrics in real-time
            - Success rate indicator shows test quality at a glance
            - Track total execution time to optimize performance
            """)

# Define use cases organized by category with icons and descriptions (needed for search)
use_case_categories = {
    "🧪 Test Generation & Intelligence": {
        "icon": "🧪",
        "modules": {
            "TestPilot": {
                "id": "test_pilot",
                "icon": "🚀",
                "description": "AI-powered test automation with Jira/Zephyr integration"
            },
            "Dynamic Test Cases": {
                "id": "dynamic_tc_generation",
                "icon": "📝",
                "description": "Generate test cases from requirements"
            },
            "Intelligent Test Data": {
                "id": "intelligent_test_data_generation",
                "icon": "🎲",
                "description": "Smart test data generation with realistic values"
            },
            "API Generation": {
                "id": "api_generation",
                "icon": "🔌",
                "description": "Generate API tests from specifications"
            }
        }
    },
    "🔧 Test Maintenance & Quality": {
        "icon": "🔧",
        "modules": {
            "Self-Healing Tests": {
                "id": "self_healing_tests",
                "icon": "🔄",
                "description": "Auto-repair broken tests automatically"
            },
            "Visual AI Testing": {
                "id": "visual_ai_testing",
                "icon": "👁️",
                "description": "Visual regression and UI analysis"
            },
            "RoboCop Lint Checker": {
                "id": "robocop_lint_checker",
                "icon": "🚨",
                "description": "Static code analysis for quality"
            },
            "Smart Test Optimizer": {
                "id": "smart_test_optimizer",
                "icon": "⚡",
                "description": "Optimize test suites for faster execution"
            },
            "FOS Quality Checks": {
                "id": "fos_checks",
                "icon": "✅",
                "description": "Front-of-site quality assurance"
            }
        }
    },
    "🚀 Automation & Integration": {
        "icon": "🚀",
        "modules": {
            "Cross-Platform Orchestrator": {
                "id": "ai_cross_platform_orchestrator",
                "icon": "🌐",
                "description": "Multi-platform test orchestration"
            },
            "Performance Testing": {
                "id": "performance_testing",
                "icon": "⚡",
                "description": "Load and performance testing tools"
            },
            "Security Penetration Testing": {
                "id": "security_penetration_testing",
                "icon": "🛡️",
                "description": "Automated security vulnerability scanning"
            },
            "Browser Agent": {
                "id": "browser_agent",
                "icon": "🌍",
                "description": "AI-powered browser automation"
            }
        }
    },
    "📊 DevOps & Monitoring": {
        "icon": "📊",
        "modules": {
            "RF Dashboard Analytics": {
                "id": "rf_dashboard_analytics",
                "icon": "📈",
                "description": "Robot Framework insights with AI"
            },
            "Jenkins Dashboard": {
                "id": "jenkins_dashboard",
                "icon": "🔨",
                "description": "CI/CD pipeline monitoring"
            },
            "EDB Query Manager": {
                "id": "edb_query_manager",
                "icon": "🗄️",
                "description": "EDB account query and management"
            },
            "Database Insights": {
                "id": "database_insights",
                "icon": "💾",
                "description": "Database optimization insights"
            },
            "AI Environment Manager": {
                "id": "ai_test_environment_manager",
                "icon": "🎛️",
                "description": "Smart environment provisioning"
            },
            "Newfold Migration Toolkit": {
                "id": "newfold_migration_toolkit",
                "icon": "🔄",
                "description": "CSRT and RAS operations platform"
            }
        }
    },
    "📝 Analysis & Documentation": {
        "icon": "📝",
        "modules": {
            "Auto Documentation": {
                "id": "auto_documentation",
                "icon": "📄",
                "description": "Generate documentation automatically"
            },
            "Smart CX Navigator": {
                "id": "smart_cx_navigator",
                "icon": "🧭",
                "description": "Customer experience optimization"
            },
            "Pull Requests Reviewer": {
                "id": "pull_requests_reviewer",
                "icon": "🔍",
                "description": "AI-powered code review"
            },
            "Manual Test Analyzer": {
                "id": "manual_test_analysis",
                "icon": "📋",
                "description": "Analyze and improve manual tests"
            }
        }
    },
    "🔮 Predictive Intelligence": {
        "icon": "🔮",
        "modules": {
            "AI Bug Predictor": {
                "id": "intelligent_bug_predictor",
                "icon": "🐛",
                "description": "Predict bugs before they occur"
            },
            "AI Execution Orchestrator": {
                "id": "ai_test_execution_orchestrator",
                "icon": "🎯",
                "description": "Intelligent test execution"
            },
            "AI Quality Guardian": {
                "id": "ai_quality_assurance_guardian",
                "icon": "🛡️",
                "description": "Comprehensive quality assurance"
            }
        }
    }
}

# Create backward compatibility mapping (flat structure for existing code)
use_cases = {}
for category_name, category_data in use_case_categories.items():
    for module_name, module_info in category_data["modules"].items():
        use_cases[module_name] = module_info["id"]

# Enhanced Search Bar with functionality
st.markdown("""
<style>
/* Search bar container styling */
.search-container {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Search input styling */
div[data-testid="stTextInput"] input {
    font-size: 1.05rem !important;
    padding: 0.75rem 1rem !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #EC5328 !important;
    box-shadow: 0 0 0 3px rgba(236, 83, 40, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# Search functionality
search_query = st.text_input(
    "🔍 Search",
    placeholder="Search modules, features, or get help... (e.g., 'API testing', 'self-healing', 'bug prediction')",
    key="module_search",
    label_visibility="collapsed"
)

# Search results
if search_query:
    st.markdown("### 🔎 Search Results")

    # Search through all modules
    search_results = []
    search_lower = search_query.lower()

    for category_name, category_data in use_case_categories.items():
        for module_name, module_info in category_data["modules"].items():
            # Search in module name, description, and category
            if (search_lower in module_name.lower() or
                search_lower in module_info["description"].lower() or
                search_lower in category_name.lower() or
                search_lower in module_info["id"].lower()):
                search_results.append({
                    "category": category_name,
                    "name": module_name,
                    "info": module_info
                })

    if search_results:
        st.success(f"Found {len(search_results)} matching module(s)")

        # Display search results in cards
        for i in range(0, len(search_results), 4):
            cols = st.columns(4)
            for j, result in enumerate(search_results[i:i+4]):
                with cols[j]:
                    module_id = result["info"]["id"]
                    is_active = st.session_state.last_module == module_id

                    st.markdown(f"""
                    <div style="
                        background: {'linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%)' if is_active else 'white'};
                        border: {'3px solid #EC5328' if is_active else '2px solid #e2e8f0'};
                        border-radius: 12px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{result["info"]["icon"]}</div>
                        <div style="
                            font-weight: 700;
                            font-size: 1rem;
                            color: {'white' if is_active else '#1e293b'};
                            margin-bottom: 0.5rem;
                        ">{result["name"]}</div>
                        <div style="
                            font-size: 0.85rem;
                            color: {'rgba(255,255,255,0.9)' if is_active else '#64748b'};
                            line-height: 1.4;
                        ">{result["info"]["description"]}</div>
                        <div style="
                            font-size: 0.75rem;
                            color: {'rgba(255,255,255,0.8)' if is_active else '#94a3b8'};
                            margin-top: 0.5rem;
                        ">{result["category"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Open {result['name']}", key=f"search_{module_id}", use_container_width=True):
                        st.session_state.last_module = module_id
                        st.session_state.pending_history_module = module_id
                        st.query_params["module"] = module_id
                        st.rerun()
    else:
        st.warning(f"No modules found matching '{search_query}'. Try searching for:\n- Module names (e.g., 'TestPilot', 'API Generation')\n- Features (e.g., 'self-healing', 'visual testing')\n- Categories (e.g., 'automation', 'analytics')")

# Add helpful info message above search bar with orange theme
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(236, 83, 40, 0.15) 0%, rgba(255, 107, 107, 0.15) 100%);
    border-left: 4px solid #EC5328;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 2px 8px rgba(236, 83, 40, 0.15);
">
    <span style="font-size: 1.5rem;">👇</span>
    <div style="
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #0f172a !important;
        line-height: 1.6;
    ">
        <strong style="color: #DC2626 !important; font-weight: 700;">Select a module from the categories below</strong> 
        <span style="color: #0f172a !important;">to get started, or expand the Quick Start Guide and Feature Highlights sections above for more information.</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Add suggested modules based on history
suggestions = get_suggestions()

# Create modern card-based navigation
for category_name, category_data in use_case_categories.items():
    st.markdown(f"### {category_name}")

    modules = category_data["modules"]
    num_modules = len(modules)
    cards_per_row = 4

    # Create rows of module cards
    module_items = list(modules.items())
    for i in range(0, num_modules, cards_per_row):
        cols = st.columns(cards_per_row)
        row_modules = module_items[i:i + cards_per_row]

        for col_idx, (module_name, module_info) in enumerate(row_modules):
            with cols[col_idx]:
                module_id = module_info["id"]
                is_active = st.session_state.last_module == module_id
                is_suggested = module_id in suggestions

                # Create card with enhanced styling
                card_class = "module-card-active" if is_active else "module-card"
                suggest_badge = "🌟 Suggested" if is_suggested else ""

                st.markdown(f"""
                <div class="{card_class}" style="
                    background: {'linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%)' if is_active else 'white'};
                    border: {'3px solid #EC5328' if is_active else '2px solid #e2e8f0'};
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 16px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: {'0 8px 16px rgba(236, 83, 40, 0.3)' if is_active else '0 2px 8px rgba(0,0,0,0.1)'};
                    height: 180px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="font-size: 48px; margin-bottom: 8px; text-align: center;">
                        {module_info['icon']}
                    </div>
                    <div style="
                        font-family: 'Inter', sans-serif;
                        font-weight: 700;
                        font-size: 16px;
                        color: {'white' if is_active else '#1e293b'};
                        text-align: center;
                        margin-bottom: 8px;
                    ">
                        {module_name}
                    </div>
                    <div style="
                        font-family: 'Inter', sans-serif;
                        font-size: 13px;
                        color: {'rgba(255,255,255,0.9)' if is_active else '#64748b'};
                        text-align: center;
                        line-height: 1.4;
                    ">
                        {module_info['description']}
                    </div>
                    {f'<div style="position: absolute; top: 10px; right: 10px; background: rgba(255,215,0,0.9); padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">{suggest_badge}</div>' if is_suggested else ''}
                </div>
                """, unsafe_allow_html=True)

                # Create invisible button for navigation
                if st.button(f"Open {module_name}", key=f"nav_{module_id}", use_container_width=True):
                    # Check module access if auth is enabled
                    if AUTH_ENABLED and st_auth and not st_auth.check_module_access(module_id, Permission.VIEW_MODULE):
                        st.error(f"⛔ Access Denied: You do not have permission to access '{module_name}'")
                        st.info("Contact your administrator to request access to this module.")
                    else:
                        st.session_state.last_module = module_id
                        st.session_state.pending_history_module = module_id
                        st.query_params["module"] = module_id
                        # Set flag to indicate this is a navigation button click (for auto-scroll)
                        st.session_state._nav_button_clicked = True
                        st.rerun()

# Check for URL anchor or query param to determine which tab to show
query_params = st.query_params
default_tab = next(iter(use_cases.values()))
selected_module = query_params.get("module") if query_params else None

# Handle string value properly - no need to extract from list
if isinstance(selected_module, list) and selected_module:
    selected_module = selected_module[0]

# Initialize session state from URL parameters if needed
if selected_module and st.session_state.last_module != selected_module:
    st.session_state.last_module = selected_module
    save_usage_history(selected_module)

# If this is from a component callback (JS button click)
if st.session_state.get('_component_callback', False):
    callback_data = st.session_state.get('_component_callback_data', {})
    if callback_data.get('action') == 'select' and callback_data.get('module'):
        selected_module = callback_data.get('module')
        st.session_state.last_module = selected_module
        # Update URL for consistency
        st.query_params.update(module=selected_module)

with st.container():
    # If a specific tab was selected in the URL or callback
    if selected_module in use_cases.values():
        # Check if this rerun was triggered by a navigation button click
        # Only scroll when the "Open {module_name}" button was clicked
        nav_button_clicked = st.session_state.get('_nav_button_clicked', False)

        # Clear the flag immediately so it doesn't trigger on subsequent reruns
        if nav_button_clicked:
            st.session_state._nav_button_clicked = False

        # Auto-scroll to module content section ONLY when navigation button was clicked
        if nav_button_clicked:
            # Use unique key based on selected module to force re-execution on every module change
            import streamlit.components.v1 as components
            import hashlib
            import time

            # Create unique identifier to force component reload on every module change
            scroll_key = hashlib.md5(f"{selected_module}_{time.time()}".encode()).hexdigest()

            components.html(f"""
            <!-- Auto-scroll component for module: {selected_module} | Key: {scroll_key} -->
            <script>
            // Robust auto-scroll with multiple attempts - scrolls to breadcrumb
            (function() {{
                let attempts = 0;
                const maxAttempts = 10;
                const scrollDelays = [100, 300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000];
                const moduleId = '{selected_module}';
                const scrollKey = '{scroll_key}';
                
                console.log('Auto-scroll initialized for module:', moduleId, 'Key:', scrollKey);
                
                function tryScroll() {{
                    if (attempts >= maxAttempts) {{
                        console.log('Max scroll attempts reached for module:', moduleId);
                        return;
                    }}
                    
                    try {{
                        const element = window.parent.document.getElementById('module-breadcrumb-{selected_module}');
                        if (element && element.offsetParent !== null) {{
                            // Element exists and is visible - scroll to breadcrumb
                            element.scrollIntoView({{ 
                                behavior: 'smooth', 
                                block: 'start',
                                inline: 'nearest'
                            }});
                            console.log('Successfully scrolled to breadcrumb for module:', moduleId);
                            return; // Success, stop trying
                        }}
                    }} catch (e) {{
                        console.log('Scroll attempt', attempts + 1, 'failed:', e);
                    }}
                    
                    // Try again with next delay
                    attempts++;
                    if (attempts < maxAttempts) {{
                        setTimeout(tryScroll, scrollDelays[attempts]);
                    }}
                }}
                
                // Start trying immediately and continue with retries
                tryScroll();
            }})();
            </script>
            """, height=0)

        # Add breadcrumb navigation
        module_display_name = get_module_friendly_name(selected_module)

        # Find which category this module belongs to
        module_category = None
        for cat_name, cat_data in use_case_categories.items():
            for mod_name, mod_info in cat_data["modules"].items():
                if mod_info["id"] == selected_module:
                    module_category = cat_name
                    break
            if module_category:
                break

        # Breadcrumb navigation with working anchor link and scroll target ID
        st.markdown(f"""
        <div id="module-breadcrumb-{selected_module}" style="
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        ">
            <a href="?module=" target="_self" style="
                color: #475569;
                text-decoration: none;
                font-weight: 500;
                transition: color 0.3s ease;
            " onmouseover="this.style.color='#EC5328'" onmouseout="this.style.color='#475569'">
                🏠 Home
            </a>
            <span style="color: #64748b;">→</span>
            <span style="color: #475569; font-weight: 500;">{module_category}</span>
            <span style="color: #64748b;">→</span>
            <span style="
                color: #DC2626;
                font-weight: 700;
                background: rgba(236, 83, 40, 0.1);
                padding: 0.25rem 0.75rem;
                border-radius: 6px;
            ">{module_display_name}</span>
        </div>
        """, unsafe_allow_html=True)
        # Update the last module in session state for UI consistency
        if st.session_state.last_module != selected_module:
            st.session_state.last_module = selected_module
            save_usage_history(selected_module)

        # Get the module name for display
        module_display_name = get_module_friendly_name(selected_module)

        # ============================================================================
        # ACCESS CONTROL CHECK
        # ============================================================================
        if AUTH_ENABLED and st_auth:
            # Check if user has access to view this module
            if not st_auth.check_module_access(selected_module, Permission.VIEW_MODULE):
                st.error(f"⛔ Access Denied: {module_display_name}")
                st.warning(f"""
                You do not have permission to access this module.
                
                **Your current roles:** {', '.join(current_user.roles)}
                
                Please contact your administrator to request access to this module.
                """)

                # Log access denial
                st.session_state.audit_logger.log(
                    action=AuditAction.ACCESS_DENIED,
                    username=current_user.username,
                    user_id=current_user.user_id,
                    success=False,
                    severity=AuditSeverity.WARNING,
                    module_id=selected_module,
                    session_id=st.session_state.auth_session_id
                )
                st.stop()

            # Log module view
            st.session_state.audit_logger.log(
                action=AuditAction.MODULE_VIEW,
                username=current_user.username,
                user_id=current_user.user_id,
                success=True,
                severity=AuditSeverity.INFO,
                module_id=selected_module,
                session_id=st.session_state.auth_session_id
            )

        # Show loading spinner while importing and rendering the module
        with st.spinner(f'🔄 Loading {module_display_name}...'):
            # Import the module but don't run any actions automatically
            try:
                # Update the import path to use correct relative import
                module_path = f"scripts.gen_ai.use_cases.{selected_module}"
                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError:
                    # Try an alternative import path if the first attempt fails
                    module_path = f"gen_ai.use_cases.{selected_module}"
                    module = importlib.import_module(module_path)

                # Show UI components for the module
                if hasattr(module, "show_ui"):
                    module.show_ui()
                else:
                    # If the module doesn't have a show_ui function, show a placeholder interface
                    st.info(f"This module ({module_display_name}) is not fully implemented yet.")
                    st.write("Coming soon! This section will provide:")

                    descriptions = {
                        "dynamic_tc_generation": "Tools for dynamically generating test cases based on application behavior and requirements.",
                        "intelligent_test_data_generation": "Smart test data generation with realistic and edge-case values.",
                        "self_healing_tests": "Automated repair of broken tests when UI elements change.",
                        "visual_ai_testing": "AI-powered visual regression testing and UI analysis.",
                        "api_generation": "Automatic generation of API tests based on specifications or existing endpoints.",
                        "auto_documentation": "Automated generation of test documentation from code and test results.",
                        "performance_testing": "Tools for performance testing and optimisation.",
                        "robocop_lint_checker": "Static code analysis for Python to enforce coding standards and detect errors.",
                        "smart_cx_navigator": "AI-driven navigation and insights for customer experience optimization.",
                        "security_penetration_testing": "Tools for automated security testing and vulnerability scanning.",
                        "pull_requests_reviewer": "AI-powered review of pull requests for code quality and compliance.",
                        "database_insights": "Intelligent insights and optimization suggestions for database performance.",
                        "jenkins_dashboard": "Integration with Jenkins for CI/CD pipeline monitoring and management.",
                        "rf_dashboard_analytics": "Robot Framework Dashboard Analytics - AI-powered insights from Jenkins test results with trend analysis, failure prediction, and optimization recommendations.",
                        "fos_checks": "Quality checks for FOS projects",
                        "intelligent_bug_predictor": "AI-powered bug prediction and analysis",
                        "smart_test_optimizer": "AI-driven test optimization and flakiness detection",
                        "ai_cross_platform_orchestrator": "Orchestrate tests across multiple platforms and devices",
                        "ai_test_environment_manager": "Automate setup and management of test environments",
                        "manual_test_analysis": "Analyze and improve manual test cases with AI assistance",
                        "ai_test_execution_orchestrator": "AI Execution Orchestrator",
                        "ai_quality_assurance_guardian": "AI Quality Guardian",
                        "browser_agent": "Browser Agent - AI-powered browser automation and testing",
                        "test_pilot": "TestPilot - AI-powered test automation assistant that fetches test cases from Jira/Zephyr, interprets steps, and generates Robot Framework scripts with intelligent keyword reuse",
                        "edb_query_manager": "EDB Query Manager - Comprehensive EDB account query and management with AI-powered insights for account lookup, domain queries, and product information",
                        "newfold_migration_toolkit": "Newfold Migration Toolkit - CSRT and RAS operations for product lifecycle management and migration testing"
                    }

                    st.write(descriptions.get(selected_module, "Features related to this use case."))

                    # Show placeholder UI for demonstration purposes
                    st.warning("This is a placeholder UI. The actual functionality is under development.")

                    # Show placeholder controls based on the module type
                    if selected_module == "dynamic_tc_generation":
                        # Skip showing placeholder UI for dynamic test case generation
                        # The module already has its own UI implementation
                        pass
                    elif selected_module == "intelligent_test_data_generation":
                        pass
                        # st.file_uploader("Upload Form Screenshot", type=["png", "jpg", "jpeg"], key=f"{selected_module}_uploader")
                        # st.text_input("Field Name", key=f"{selected_module}_field")
                        # st.selectbox("Field Type", ["Email", "Password", "Phone", "Date", "Name"], key=f"{selected_module}_type")
                        # st.button("Generate Test Data", key=f"{selected_module}_button")

                    elif selected_module == "self_healing_tests":
                        st.file_uploader("Upload Failed Test Report", type=["xml", "json", "html"],
                                         key=f"{selected_module}_uploader")
                        st.selectbox("Healing Strategy", ["Fuzzy Matching", "Computer Vision", "DOM Analysis"],
                                     key=f"{selected_module}_strategy")
                        st.button("Repair Tests", key=f"{selected_module}_button")

                    elif selected_module == "visual_ai_testing":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.file_uploader("Upload Baseline Image", type=["png", "jpg", "jpeg"],
                                             key=f"{selected_module}_baseline")
                        with col2:
                            st.file_uploader("Upload Comparison Image", type=["png", "jpg", "jpeg"],
                                             key=f"{selected_module}_comparison")
                        st.slider("Sensitivity", 0.0, 1.0, 0.8, key=f"{selected_module}_sensitivity")
                        st.button("Compare Images", key=f"{selected_module}_button")

                    elif selected_module == "api_generation":
                        st.file_uploader("Upload API Specification", type=["yaml", "json"],
                                         key=f"{selected_module}_uploader")
                        st.text_input("API Endpoint", key=f"{selected_module}_endpoint")
                        st.selectbox("Test Framework", ["Postman", "RestAssured", "Requests"],
                                     key=f"{selected_module}_framework")
                        st.button("Generate API Tests", key=f"{selected_module}_button")

                    elif selected_module == "auto_documentation":
                        st.file_uploader("Upload Test Results", accept_multiple_files=True,
                                         key=f"{selected_module}_uploader")
                        st.text_input("Project Name", key=f"{selected_module}_project")
                        st.selectbox("Documentation Format", ["HTML", "PDF", "Markdown", "Word"],
                                     key=f"{selected_module}_format")
                        st.button("Generate Documentation", key=f"{selected_module}_button")

            except ImportError as e:
                st.error(f"Module {selected_module} not found. Please make sure it exists in the use_cases directory.")
                st.code(f"Error: {e}")
                # If the module doesn't exist yet, show a placeholder
                st.info(f"This module ({selected_module}) hasn't been implemented yet.")
                st.write("When implemented, this section will contain:")

                descriptions = {
                    "dynamic_tc_generation": "Tools for dynamically generating test cases based on application behavior and requirements.",
                    "intelligent_test_data_generation": "Smart test data generation with realistic and edge-case values.",
                    "self_healing_tests": "Automated repair of broken tests when UI elements change.",
                    "visual_ai_testing": "AI-powered visual regression testing and UI analysis.",
                    "api_generation": "Automatic generation of API tests based on specifications or existing endpoints.",
                    "auto_documentation": "Automated generation of test documentation from code and test results.",
                    "performance_testing": "Tools for performance testing and optimisation.",
                    "robocop_lint_checker": "Static code analysis for Python to enforce coding standards and detect errors.",
                    "smart_cx_navigator": "AI-driven navigation and insights for customer experience optimization.",
                    "security_penetration_testing": "Tools for automated security testing and vulnerability scanning.",
                    "pull_requests_reviewer": "AI-powered review of pull requests for code quality and compliance.",
                    "database_insights": "Intelligent insights and optimization suggestions for database performance.",
                    "jenkins_dashboard": "Integration with Jenkins for CI/CD pipeline monitoring and management.",
                    "rf_dashboard_analytics": "Robot Framework Dashboard Analytics - AI-powered insights from Jenkins test results with trend analysis, failure prediction, and optimization recommendations.",
                    "fos_checks": "Quality checks for FOS projects",
                    "intelligent_bug_predictor": "AI-powered bug prediction and analysis",
                    "smart_test_optimizer": "AI-driven test optimization and flakiness detection",
                    "ai_cross_platform_orchestrator": "Orchestrate tests across multiple platforms and devices",
                    "ai_test_environment_manager": "Automate setup and management of test environments",
                    "manual_test_analysis": "Analyze and improve manual test cases with AI assistance",
                    "ai_test_execution_orchestrator": "AI Execution Orchestrator",
                    "ai_quality_assurance_guardian": "AI Quality Guardian",
                    "browser_agent": "Browser Agent - AI-powered browser automation and testing",
                    "test_pilot": "TestPilot - AI-powered test automation assistant that fetches test cases from Jira/Zephyr, interprets steps, and generates Robot Framework scripts with intelligent keyword reuse",
                    "edb_query_manager": "EDB Query Manager - Comprehensive EDB account query and management with AI-powered insights for account lookup, domain queries, and product information",
                    "newfold_migration_toolkit": "Newfold Migration Toolkit - CSRT and RAS operations for product lifecycle management and migration testing"
                }

                st.write(descriptions.get(selected_module, "Features related to this use case."))

            # Add favorite button
            col1, col2 = st.columns([9, 1])
            with col2:
                is_favorite = selected_module in st.session_state.favorites
                if st.button("★" if is_favorite else "☆", key=f"fav_{selected_module}"):
                    if is_favorite:
                        st.session_state.favorites.remove(selected_module)
                    else:
                        st.session_state.favorites.append(selected_module)
                    st.rerun()

# Add keyboard shortcuts info - moved right after module overview
with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
    st.markdown("""
    ### Quick Navigation Shortcuts
    
    | Shortcut | Action |
    |----------|--------|
    | `Ctrl + F` or `Cmd + F` | Focus search bar |
    | `Ctrl + Home` or `Cmd + ↑` | Scroll to top |
    | `Ctrl + End` or `Cmd + ↓` | Scroll to bottom |
    | `Tab` | Navigate between elements |
    | `Enter` | Activate focused button |
    
    **Pro Tip:** Use the search bar to quickly find modules by name, feature, or category!
    """)

# Feedback section
with st.expander("Provide Feedback", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        rating = st.selectbox("How would you rate this tool?", ["Select rating", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
    with col2:
        comments = st.text_area("Comments or suggestions:")
    submit_button = st.button("Submit Feedback")

    # Handle feedback submission
    if submit_button:
        success, message = submit_feedback(rating, comments)
        if success:
            st.success(message)
            st.session_state.feedback_submitted = True
            # Don't clear feedback_data here so it can be displayed below
        else:
            st.error(message)

# Display feedback history if available and we're not just after submission
if len(st.session_state.feedback_data) > 0:
    with st.expander("Your Feedback History", expanded=st.session_state.feedback_submitted):
        feedback_df = pd.DataFrame(st.session_state.feedback_data)
        if not feedback_df.empty:
            feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
            feedback_df = feedback_df.sort_values(by='timestamp', ascending=False)
            feedback_df['timestamp'] = feedback_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(feedback_df, use_container_width=True)
        else:
            st.info("No feedback history available.")

st.markdown("""
<style>
/* Ensure footer stays at bottom with minimal gap */
.main .block-container {
    padding-bottom: 0.5rem !important;
}
</style>
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 0;
">
    <div style="
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #EC5328, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    ">🌀The Vortex</div>
    <div style="
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 0.75rem;
    ">Virtual Orchestrator for Real-world Technology EXcellence</div>
    <div style="
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748b;
    ">
        🌀The Vortex - Built with ❤️ for Quality & AI Engineering Excellence © 2025-26
        <br>
        <span style="font-size: 0.75rem; margin-top: 0.5rem; display: inline-block;">
            Contact: <a href="mailto:siddhant.wadhwani@newfold.com" style="color: #EC5328; text-decoration: none;">Siddhant Wadhwani</a>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

