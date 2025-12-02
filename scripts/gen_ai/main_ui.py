import streamlit as st
import importlib
import sys
import os
import json
from datetime import datetime
import pandas as pd

# Configure the page to use wide mode and set a nice title
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="The Vortex - Gen AI Testing Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    padding: 2rem 3rem;
    max-width: 1400px;
    margin: 0 auto;
    background: var(--background-primary);
    border-radius: var(--border-radius-xl);
    box-shadow: var(--shadow-sm);
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

/* Navigation container with improved grid */
.nav-container {
    padding: 0.1rem;
    background: rgba(248, 250, 252, 0.8);
    border-radius: var(--border-radius-xl);
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
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
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--background-secondary);
    border-radius: var(--border-radius-md);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: var(--border-radius-md);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-dark);
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

# Check if we have a pending module selection from navigation that needs to be saved to history
if 'pending_history_module' in st.session_state and st.session_state.pending_history_module:
    # This means a navigation happened but history wasn't saved yet
    module_to_save = st.session_state.pending_history_module
    if module_to_save != st.session_state.last_saved_module:
        save_usage_history(module_to_save)


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
        "fos_checks": "FOS Quality Checks",
        "intelligent_bug_predictor": "AI Bug Predictor",
        "smart_test_optimizer": "Smart Test Optimizer",
        "ai_cross_platform_orchestrator": "Cross-Platform Orchestrator",
        "ai_test_environment_manager": "AI Environment Manager",
        "manual_test_analysis": "Manual Test Analyzer",
        "ai_test_execution_orchestrator": "AI Execution Orchestrator",
        "ai_quality_assurance_guardian": "AI Quality Guardian",
        "browser_agent": "Browser Agent",
        "test_pilot": "TestPilot"
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


# Main header with notifications
col1, col2, col3 = st.columns([1, 10, 1])
with col1:
    if st.button("📋"):
        st.session_state.show_history = not st.session_state.get('show_history', False)
with col2:
    st.markdown('<h1 class="main-header">"The Vortex" - Gen AI Testing Portal</h1>', unsafe_allow_html=True)
with col3:
    if st.button(f"🔔 {st.session_state.notification_count}" if st.session_state.notification_count > 0 else "🔔"):
        st.session_state.show_notifications = not st.session_state.get('show_notifications', False)

# Display history if toggled
if st.session_state.get('show_history', False):
    with st.expander("Usage History", expanded=True):
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)

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

# Display notifications if toggled
if st.session_state.get('show_notifications', False):
    with st.expander("Notifications", expanded=True):
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

# Search bar
with st.container():
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    search_query = st.text_input("🔍 Search use cases, features or get help...",
                                 placeholder="e.g., 'generate API tests' or 'fix failing tests'")
    st.markdown('</div>', unsafe_allow_html=True)

# AI Insights section (if we have enough data)
if len(st.session_state.history) > 2:
    insights = generate_insights()
    if insights:
        with st.container():
            st.markdown('<div class="ai-insights">', unsafe_allow_html=True)
            st.markdown("#### 💡 AI Insights")
            for insight in insights:
                st.markdown(f"- {insight}")
            st.markdown('</div>', unsafe_allow_html=True)

# Navigation menu with active state
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

# Define all the use cases
use_cases = {
    "Dynamic Test Cases": "dynamic_tc_generation",
    "Intelligent Test Data": "intelligent_test_data_generation",
    "Self-Healing Tests": "self_healing_tests",
    "Visual AI Testing": "visual_ai_testing",
    "API Generation": "api_generation",
    "Auto Documentation": "auto_documentation",
    "Performance Testing": "performance_testing",
    "RoboCop Lint Checker": "robocop_lint_checker",
    "Smart CX Navigator": "smart_cx_navigator",
    "Security Penetration Testing": "security_penetration_testing",
    "Pull Requests Reviewer": "pull_requests_reviewer",
    "Database Insights": "database_insights",
    "Jenkins Dashboard": "jenkins_dashboard",
    "FOS Quality Checks": "fos_checks",
    # Advanced AI-Powered Use Cases
    "AI Bug Predictor": "intelligent_bug_predictor",
    "Smart Test Optimizer": "smart_test_optimizer",
    "Cross-Platform Orchestrator": "ai_cross_platform_orchestrator",
    "AI Environment Manager": "ai_test_environment_manager",
    "Manual Test Analyzer": "manual_test_analysis",
    "AI Execution Orchestrator": "ai_test_execution_orchestrator",
    "AI Quality Guardian": "ai_quality_assurance_guardian",
    "Browser Agent": "browser_agent",
    "TestPilot": "test_pilot"
}

# Add suggested modules based on history
suggestions = get_suggestions()

# Create a multi-line grid layout for buttons - 4 buttons per row
num_buttons = len(use_cases)
buttons_per_row = 4
num_rows = (num_buttons + buttons_per_row - 1) // buttons_per_row  # Ceiling division

# Create buttons row by row
use_cases_items = list(use_cases.items())
for row in range(num_rows):
    # Create columns for this row
    start_idx = row * buttons_per_row
    end_idx = min(start_idx + buttons_per_row, num_buttons)
    row_items = use_cases_items[start_idx:end_idx]
    cols = st.columns(len(row_items))

    # Create buttons in this row
    for i, (case_name, module_name) in enumerate(row_items):
        with cols[i]:
            suggest_marker = " 🌟" if module_name in suggestions else ""
            button_label = f"{case_name}{suggest_marker}"

            # Highlight active button with a different style
            active_button = st.session_state.last_module == module_name
            button_style = f"""
            <style>
            div[data-testid="stHorizontalBlock"] div[data-testid="column"] button[key="nav_{module_name}"] {{
                background-color: {'#0D47A1' if active_button else '#1E88E5'};
                color: white;
                border: {'2px solid #ffffff' if active_button else 'none'};
                font-weight: {'bold' if active_button else 'normal'};
            }}
            div[data-testid="stHorizontalBlock"] div[data-testid="column"] button[key="nav_{module_name}"]:hover {{
                background-color: #0D47A1;
                border: 2px solid #ffffff;
            }}
            </style>
            """
            st.markdown(button_style, unsafe_allow_html=True)

            # Create clickable button for each module
            if st.button(button_label, key=f"nav_{module_name}",
                         use_container_width=True,
                         help=f"Go to {case_name} module"):
                # Update session state
                st.session_state.last_module = module_name

                # Set a pending flag to track this module selection across reruns
                st.session_state.pending_history_module = module_name

                # Update query parameters for URL persistence
                st.query_params["module"] = module_name  # Use direct assignment instead of update method

                # Force a rerun to refresh the UI with the new module
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Metrics Dashboard
with st.container():
    st.markdown("## 📊 Test Execution Metrics", unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<h3>{st.session_state.execution_metrics["tests_executed"]}</h3>'
            f'<p>Tests Executed</p>'
            f'<div class="metric-icon">🧪</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with metric_col2:
        success_rate = 0
        if st.session_state.execution_metrics["tests_executed"] > 0:
            success_rate = (st.session_state.execution_metrics["successful_tests"] /
                            st.session_state.execution_metrics["tests_executed"]) * 100

        st.markdown(
            f'<div class="metric-card">'
            f'<h3>{success_rate:.1f}%</h3>'
            f'<p>Success Rate</p>'
            f'<div class="metric-icon">✓</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with metric_col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<h3>{st.session_state.execution_metrics["execution_time"]:.1f}s</h3>'
            f'<p>Total Execution Time</p>'
            f'<div class="metric-icon">⏱️</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

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

# Content section with line separator
st.markdown("---")

with st.container():
    st.markdown('<div class="content-section">', unsafe_allow_html=True)

    # If a specific tab was selected in the URL or callback
    if selected_module in use_cases.values():
        # Update the last module in session state for UI consistency
        if st.session_state.last_module != selected_module:
            st.session_state.last_module = selected_module
            save_usage_history(selected_module)

        # Get the module name for display
        module_display_name = get_module_friendly_name(selected_module)

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
                    "fos_checks": "Quality checks for FOS projects",
                    "intelligent_bug_predictor": "AI-powered bug prediction and analysis",
                    "smart_test_optimizer": "AI-driven test optimization and flakiness detection",
                    "ai_cross_platform_orchestrator": "Orchestrate tests across multiple platforms and devices",
                    "ai_test_environment_manager": "Automate setup and management of test environments",
                    "manual_test_analysis": "Analyze and improve manual test cases with AI assistance",
                    "ai_test_execution_orchestrator": "AI Execution Orchestrator",
                    "ai_quality_assurance_guardian": "AI Quality Guardian",
                    "browser_agent": "Browser Agent - AI-powered browser automation and testing"
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
                "fos_checks": "Quality checks for FOS projects",
                "intelligent_bug_predictor": "AI-powered bug prediction and analysis",
                "smart_test_optimizer": "AI-driven test optimization and flakiness detection",
                "ai_cross_platform_orchestrator": "Orchestrate tests across multiple platforms and devices",
                "ai_test_environment_manager": "Automate setup and management of test environments",
                "manual_test_analysis": "Analyze and improve manual test cases with AI assistance",
                "ai_test_execution_orchestrator": "AI Execution Orchestrator",
                "ai_quality_assurance_guardian": "AI Quality Guardian",
                "browser_agent": "Browser Agent - AI-powered browser automation and testing",
                "test_pilot": "TestPilot - AI-powered test automation assistant that fetches test cases from Jira/Zephyr, interprets steps, and generates Robot Framework scripts with intelligent keyword reuse"
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
    else:
        # Show welcome screen when no module is selected
        st.markdown("## 👋 Welcome to \"The Vortex\"")
        st.markdown("""
        Click on one of the options above to get started with our comprehensive suite of AI-powered testing tools:

        ### 🧪 Test Generation & Data
        - **Dynamic Test Cases**: Generate test cases based on application behavior and requirements
        - **Intelligent Test Data**: Create realistic test data for your test cases
        - **TestPilot**: AI-powered assistant that fetches test cases from Jira/Zephyr and converts them to Robot Framework scripts with intelligent keyword reuse
        - **FOS Quality Checks**: Comprehensive front-of-site quality assurance including console error detection, media analysis, link crawling, accessibility audits, performance monitoring, and network analysis

        ### 🔧 Test Maintenance & Quality
        - **Self-Healing Tests**: Automatically repair broken tests when UI elements change
        - **Visual AI Testing**: Perform visual regression testing and UI analysis
        - **RoboCop Lint Checker**: Static code analysis for Python to enforce coding standards
        - **Smart Test Optimizer**: AI-driven test suite optimization to reduce execution time by up to 70%

        ### 🚀 Automation & Integration
        - **API Generation**: Generate API tests based on specifications or existing endpoints
        - **Performance Testing**: Tools for performance testing and optimization
        - **Security Penetration Testing**: Automated security testing and vulnerability scanning
        - **Cross-Platform Orchestrator**: Intelligent cross-browser and cross-platform testing automation

        ### 📊 DevOps & Monitoring
        - **Jenkins Dashboard**: Integration with Jenkins for CI/CD pipeline monitoring
        - **Pull Requests Reviewer**: AI-powered review of pull requests for code quality
        - **Database Insights**: Intelligent insights and optimization for database performance
        - **AI Environment Manager**: Smart test environment provisioning and data management

        ### 📝 Documentation & Analysis
        - **Auto Documentation**: Automate test documentation from code and test results
        - **Smart CX Navigator**: AI-driven navigation and insights for customer experience optimization
        - **Manual Test Analyzer**: Transform manual testing with AI-powered analysis and automation recommendations

        ### 🔮 Predictive Intelligence
        - **AI Bug Predictor**: Leverage AI to predict potential bugs before they occur in production using code complexity patterns, historical data, and developer patterns

        ### 💰 Time & Cost Savings
        Our advanced AI modules can save your team significant time and effort:
        - **70% reduction** in test execution time with Smart Test Optimizer
        - **Automated bug prediction** preventing production issues
        - **ROI calculator** showing payback periods for automation investments
        - **Cross-platform optimization** reducing testing costs across multiple environments

        Use the search bar above to find specific features or get help with any of these modules.
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

    st.markdown('</div>', unsafe_allow_html=True)

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

# Add 3D animation for star particles
import streamlit.components.v1 as components

components.html(
    """
    <div id='star-animation-container' style='position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1;'>
        <canvas id='star-animation'></canvas>
    </div>
    <script>
        const canvas = document.getElementById('star-animation');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const stars = Array.from({ length: 100 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            z: Math.random() * canvas.width,
        }));

        function drawStars() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            stars.forEach(star => {
                const perspective = canvas.width / (canvas.width + star.z);
                const x = star.x * perspective;
                const y = star.y * perspective;
                const size = perspective * 2;

                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fillStyle = 'white';
                ctx.fill();

                star.z -= 2;
                if (star.z < 0) {
                    star.z = canvas.width;
                }
            });
        }

        function animate() {
            drawStars();
            requestAnimationFrame(animate);
        }

        animate();

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
    """,
    height=0
)

# Ensure compatibility with Streamlit components
import streamlit.components.v1 as components

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 10px;">
    <p>THE VORTEX © 2025</p>
</div>
""", unsafe_allow_html=True)
