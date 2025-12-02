#!/usr/bin/env python3
"""
Setup script for Enhanced Intelligent Test Data Generator
Helps users configure the environment and verify dependencies
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"✅ {package_name} is installed")
        return True
    else:
        print(f"❌ {package_name} is not installed")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements_enhanced_test_generator.txt"
    
    if not requirements_file.exists():
        print("❌ Requirements file not found")
        return False
    
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT)
        print("✅ Tesseract OCR is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Tesseract OCR is not installed")
        print("📝 Install instructions:")
        print("   - macOS: brew install tesseract")
        print("   - Ubuntu: sudo apt-get install tesseract-ocr")
        print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def check_chrome_driver():
    """Check if Chrome and ChromeDriver are available"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        driver.quit()
        print("✅ Chrome and ChromeDriver are working")
        return True
    except Exception as e:
        print("❌ Chrome/ChromeDriver issue:", str(e))
        print("📝 Install instructions:")
        print("   - Install Google Chrome browser")
        print("   - ChromeDriver will be auto-managed by webdriver-manager")
        return False

def setup_azure_openai():
    """Guide user through Azure OpenAI setup"""
    print("\n🤖 Azure OpenAI Configuration")
    print("=" * 40)
    
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var} is set")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📝 Setup instructions:")
        print("1. Create an Azure OpenAI resource in Azure Portal")
        print("2. Deploy a model (e.g., gpt-4, gpt-35-turbo)")
        print("3. Set environment variables:")
        print('   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"')
        print('   export AZURE_OPENAI_API_KEY="your-api-key"')
        print('   export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"')
        print("\n4. Or create a .env file with these variables")
        return False
    else:
        print("✅ Azure OpenAI configuration looks good")
        return True

def test_azure_openai():
    """Test Azure OpenAI connection"""
    try:
        from azure_openai_client import AzureOpenAIClient
        
        client = AzureOpenAIClient()
        response = client.generate_response("Hello, world!", max_tokens=10)
        print("✅ Azure OpenAI connection test successful")
        return True
    except Exception as e:
        print(f"❌ Azure OpenAI connection test failed: {e}")
        return False

def create_sample_env_file():
    """Create a sample .env file"""
    env_content = """# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21

# Optional configurations
# SELENIUM_HEADLESS=true
# OCR_CONFIDENCE_THRESHOLD=60
# CRAWL_RATE_LIMIT=2
"""
    
    env_file = Path(__file__).parent / ".env.example"
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"📄 Created sample environment file: {env_file}")
    print("💡 Copy to .env and update with your actual values")

def run_setup():
    """Run the complete setup process"""
    print("🚀 Enhanced Intelligent Test Data Generator Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    print("\n📦 Checking Dependencies")
    print("-" * 25)
    
    # Core packages
    core_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("pytesseract", "pytesseract"),
        ("requests", "requests"),
        ("beautifulsoup4", "bs4"),
        ("selenium", "selenium"),
        ("plotly", "plotly")
    ]
    
    missing_packages = []
    for package_name, import_name in core_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❓ Install missing packages? ({', '.join(missing_packages)})")
        response = input("Type 'y' to install: ").lower().strip()
        if response == 'y':
            if not install_requirements():
                success = False
        else:
            success = False
    
    print("\n🔧 Checking External Dependencies")
    print("-" * 35)
    
    # Check Tesseract
    if not check_tesseract():
        success = False
    
    # Check Chrome/ChromeDriver
    if not check_chrome_driver():
        print("⚠️  Website traversal features will be limited without Chrome/ChromeDriver")
    
    # Azure OpenAI setup
    if not setup_azure_openai():
        print("⚠️  AI features will be disabled without Azure OpenAI configuration")
        success = False
    else:
        if not test_azure_openai():
            print("⚠️  Azure OpenAI configured but connection test failed")
    
    # Create sample environment file
    create_sample_env_file()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("✅ You can now run the Enhanced Intelligent Test Data Generator")
        print("\n📝 Next steps:")
        print("   1. Run: streamlit run intelligent_test_data_generation.py")
        print("   2. Or try the demo: streamlit run intelligent_test_data_generation_demo.py")
    else:
        print("⚠️  Setup completed with some issues")
        print("📋 Please address the issues above for full functionality")
    
    return success

if __name__ == "__main__":
    run_setup()
