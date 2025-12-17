#!/bin/bash

# GTMetrix-Style Performance Testing Suite - Installation Script
# This script installs all required dependencies for the performance testing suite

echo "🔧 GTMetrix-Style Performance Testing Suite - Installation"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if script is run from correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "Installation directory: $SCRIPT_DIR"

# Step 1: Check Python installation
echo ""
echo "📋 Step 1: Checking Python Installation"
echo "======================================="

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python found: $PYTHON_VERSION"
    
    # Check Python version (minimum 3.8)
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_status "Python version is compatible (3.8+)"
    else
        print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    print_info "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Step 2: Install Python dependencies
echo ""
echo "📦 Step 2: Installing Python Dependencies"
echo "========================================="

if [ -f "gtmetrix_requirements.txt" ]; then
    print_info "Installing Python packages from gtmetrix_requirements.txt..."
    
    if pip3 install -r gtmetrix_requirements.txt; then
        print_status "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        print_info "Try: pip3 install --user -r gtmetrix_requirements.txt"
        exit 1
    fi
else
    print_warning "gtmetrix_requirements.txt not found. Installing core packages..."
    
    CORE_PACKAGES="pandas numpy matplotlib seaborn requests beautifulsoup4 selenium jinja2 python-dateutil"
    
    if pip3 install $CORE_PACKAGES; then
        print_status "Core Python packages installed successfully"
    else
        print_error "Failed to install core Python packages"
        exit 1
    fi
fi

# Step 3: Check/Install Node.js
echo ""
echo "🟢 Step 3: Checking Node.js Installation"
echo "========================================"

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js found: $NODE_VERSION"
else
    print_warning "Node.js not found. Installing via NVM..."
    
    # Install NVM (Node Version Manager)
    if command -v curl &> /dev/null; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        
        # Source NVM
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
        
        # Install latest LTS Node.js
        nvm install --lts
        nvm use --lts
        
        print_status "Node.js installed via NVM"
    else
        print_error "curl is required to install Node.js via NVM"
        print_info "Please install Node.js manually: https://nodejs.org/"
        exit 1
    fi
fi

# Step 4: Install Lighthouse
echo ""
echo "🔍 Step 4: Installing Lighthouse"
echo "================================"

if command -v lighthouse &> /dev/null; then
    LIGHTHOUSE_VERSION=$(lighthouse --version)
    print_status "Lighthouse found: $LIGHTHOUSE_VERSION"
else
    print_info "Installing Lighthouse globally..."
    
    if npm install -g lighthouse; then
        print_status "Lighthouse installed successfully"
    else
        print_error "Failed to install Lighthouse"
        print_info "Try: sudo npm install -g lighthouse"
        exit 1
    fi
fi

# Step 5: Check Chrome/Chromium
echo ""
echo "🌐 Step 5: Checking Browser Installation"
echo "========================================"

CHROME_FOUND=false

# Check for Google Chrome
if command -v google-chrome &> /dev/null; then
    print_status "Google Chrome found"
    CHROME_FOUND=true
elif command -v google-chrome-stable &> /dev/null; then
    print_status "Google Chrome (stable) found"
    CHROME_FOUND=true
elif command -v chromium &> /dev/null; then
    print_status "Chromium found"
    CHROME_FOUND=true
elif command -v chromium-browser &> /dev/null; then
    print_status "Chromium browser found"
    CHROME_FOUND=true
elif [ -d "/Applications/Google Chrome.app" ]; then
    print_status "Google Chrome found (macOS)"
    CHROME_FOUND=true
fi

if [ "$CHROME_FOUND" = false ]; then
    print_warning "Chrome/Chromium not found"
    print_info "Please install Google Chrome or Chromium browser:"
    print_info "• Chrome: https://www.google.com/chrome/"
    print_info "• Chromium: https://www.chromium.org/"
fi

# Step 6: Verify installation
echo ""
echo "✅ Step 6: Verifying Installation"
echo "================================="

print_info "Testing core functionality..."

# Test Python imports
python3 -c "
import pandas, numpy, matplotlib, seaborn, requests, selenium
print('✅ All Python packages imported successfully')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "Python packages verified"
else
    print_error "Some Python packages are missing or broken"
fi

# Test Lighthouse
if lighthouse --version &> /dev/null; then
    print_status "Lighthouse is working"
else
    print_error "Lighthouse is not working properly"
fi

# Step 7: Create sample configuration
echo ""
echo "⚙️ Step 7: Creating Sample Configuration"
echo "======================================="

# Make scripts executable
chmod +x gtmetrix_style_performance_checker.py 2>/dev/null
chmod +x enhanced_gtmetrix_dashboard.py 2>/dev/null
chmod +x run_examples.sh 2>/dev/null

print_status "Scripts made executable"

# Step 8: Display next steps
echo ""
echo "🎉 Installation Complete!"
echo "========================"
print_status "GTMetrix-Style Performance Testing Suite is ready to use!"

echo ""
echo "📋 Next Steps:"
echo "============="
echo "1. 🧪 Run example tests:"
echo "   ./run_examples.sh"
echo ""
echo "2. 🔍 Test a single website:"
echo "   python3 gtmetrix_style_performance_checker.py --urls https://example.com"
echo ""
echo "3. 📊 Start the dashboard:"
echo "   python3 enhanced_gtmetrix_dashboard.py"
echo ""
echo "4. 📄 Read the documentation:"
echo "   cat GTMetrix_Style_Performance_Suite_README.md"
echo ""
echo "💡 Optional Enhancements:"
echo "========================"
echo "• Set up CrUX API key for real user metrics"
echo "• Configure custom performance thresholds"
echo "• Set up automated testing schedules"
echo ""
echo "🆘 Troubleshooting:"
echo "=================="
echo "• Check the README for common issues"
echo "• Ensure all browsers are up to date"
echo "• Verify network connectivity for testing"
echo ""
echo "Happy performance testing! 🚀"
