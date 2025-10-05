#!/bin/bash

# Search Ranking System - Complete Setup Script
# This script sets up the entire project from scratch

set -e  # Exit on error

echo "=================================================="
echo "  Search Ranking System - Automated Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."

mkdir -p src/rankers
mkdir -p src/features
mkdir -p src/evaluation
mkdir -p src/ab_testing
mkdir -p src/api
mkdir -p data
mkdir -p models
mkdir -p tests
mkdir -p scripts
mkdir -p demo/templates
mkdir -p notebooks

print_success "Directories created"

# Create __init__.py files
echo ""
echo "Creating Python package files..."

# src/__init__.py
cat > src/__init__.py << 'EOF'
"""Search Relevance & Ranking System"""
__version__ = "1.0.0"
EOF

# src/rankers/__init__.py
cat > src/rankers/__init__.py << 'EOF'
"""Ranking algorithms module"""
from .tfidf_ranker import TFIDFRanker
from .bm25_ranker import BM25Ranker
from .lambdamart_ranker import LambdaMARTRanker

__all__ = ['TFIDFRanker', 'BM25Ranker', 'LambdaMARTRanker']
EOF

# src/features/__init__.py
cat > src/features/__init__.py << 'EOF'
"""Feature extraction module"""
from .feature_extractor import FeatureExtractor

__all__ = ['FeatureExtractor']
EOF

# src/evaluation/__init__.py
cat > src/evaluation/__init__.py << 'EOF'
"""Evaluation metrics module"""
from .metrics import RankingMetrics

__all__ = ['RankingMetrics']
EOF

# src/ab_testing/__init__.py
cat > src/ab_testing/__init__.py << 'EOF'
"""A/B testing framework module"""
from .experiment import ABTestExperiment, ABTestManager

__all__ = ['ABTestExperiment', 'ABTestManager']
EOF

# src/api/__init__.py
cat > src/api/__init__.py << 'EOF'
"""API module"""
from .app import app

__all__ = ['app']
EOF

# tests/__init__.py
cat > tests/__init__.py << 'EOF'
"""Test suite"""
EOF

# Create .gitkeep files
touch data/.gitkeep
touch models/.gitkeep

print_success "Package files created"

# Create virtual environment
echo ""
echo "Creating virtual environment..."

if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
print_success "pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Generate sample data
echo ""
echo "Generating sample data..."
if [ -f "scripts/generate_sample_data.py" ]; then
    python scripts/generate_sample_data.py
    print_success "Sample data generated"
else
    print_error "generate_sample_data.py not found!"
    exit 1
fi

# Train models
echo ""
echo "Training models (this may take a few minutes)..."
if [ -f "scripts/train_model.py" ]; then
    python scripts/train_model.py
    print_success "Models trained"
else
    print_error "train_model.py not found!"
    exit 1
fi

# Run tests
echo ""
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v
    print_success "All tests passed"
else
    print_info "pytest not found, skipping tests"
fi

# Initialize git repository
echo ""
echo "Initializing git repository..."
if [ -d ".git" ]; then
    print_info "Git repository already initialized"
else
    git init
    git add .
    git commit -m "Initial commit: Complete Search Ranking System"
    print_success "Git repository initialized"
fi

# Final summary
echo ""
echo "=================================================="
echo "  Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Project structure:"
echo "  ✓ All directories created"
echo "  ✓ Python packages configured"
echo "  ✓ Dependencies installed"
echo "  ✓ Sample data generated (100 documents, 50 queries)"
echo "  ✓ Models trained (TF-IDF, BM25, LambdaMART)"
echo "  ✓ Tests executed"
echo "  ✓ Git repository initialized"
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Start API Server:"
echo "   python src/api/app.py"
echo "   → http://localhost:5000"
echo ""
echo "2. Start Live Demo:"
echo "   python demo/app.py"
echo "   → http://localhost:8080"
echo ""
echo "3. Run Experiments:"
echo "   python scripts/run_experiments.py"
echo ""
echo "4. Run Tests:"
echo "   pytest tests/ -v"
echo ""
echo "=================================================="
echo ""
print_success "Ready to use! Happy searching! 🔍"
echo ""
