#!/usr/bin/env python3
"""
Advanced Data Analytics Portfolio Setup Script

This script automates the setup and installation of the data analytics portfolio.
It handles environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class PortfolioSetup:
    """Setup class for the Advanced Data Analytics Portfolio"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = "3.9"
        self.venv_name = "venv"
        self.requirements_file = "requirements.txt"
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 80)
        print("🚀 Advanced Data Analytics Portfolio Setup")
        print("=" * 80)
        print("This script will set up your data analytics portfolio environment.")
        print("It includes:")
        print("  • Python virtual environment")
        print("  • All required dependencies")
        print("  • Sample data and models")
        print("  • Configuration files")
        print("=" * 80)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        current_version = sys.version_info
        
        if current_version.major < 3 or (current_version.major == 3 and current_version.minor < 8):
            print(f"❌ Python {self.python_version}+ is required. Current version: {current_version.major}.{current_version.minor}")
            return False
        
        print(f"✅ Python version {current_version.major}.{current_version.minor} is compatible")
        return True
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        print("🐍 Creating virtual environment...")
        
        venv_path = self.project_root / self.venv_name
        
        if venv_path.exists():
            print(f"⚠️  Virtual environment already exists at {venv_path}")
            response = input("Do you want to recreate it? (y/N): ").lower()
            if response == 'y':
                shutil.rmtree(venv_path)
            else:
                print("Using existing virtual environment")
                return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print(f"✅ Virtual environment created at {venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self):
        """Get the Python executable from virtual environment"""
        if platform.system() == "Windows":
            return self.project_root / self.venv_name / "Scripts" / "python.exe"
        else:
            return self.project_root / self.venv_name / "bin" / "python"
    
    def get_venv_pip(self):
        """Get the pip executable from virtual environment"""
        if platform.system() == "Windows":
            return self.project_root / self.venv_name / "Scripts" / "pip.exe"
        else:
            return self.project_root / self.venv_name / "bin" / "pip"
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        pip_executable = self.get_venv_pip()
        requirements_path = self.project_root / self.requirements_file
        
        if not requirements_path.exists():
            print(f"❌ Requirements file not found: {requirements_path}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([str(pip_executable), "install", "-r", str(requirements_path)], check=True)
            
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating project directories...")
        
        directories = [
            "data",
            "models", 
            "logs",
            "config",
            "notebooks",
            "docs",
            "tests"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Created {directory}/")
    
    def create_sample_data(self):
        """Create sample data files"""
        print("📊 Creating sample data...")
        
        data_dir = self.project_root / "data"
        
        # Create sample CSV data
        import pandas as pd
        import numpy as np
        
        # Sample stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stock_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        stock_data.to_csv(data_dir / "sample_stock_data.csv", index=False)
        
        # Sample sentiment data
        sentiment_data = pd.DataFrame({
            'text': [
                "I love this product! It's amazing.",
                "This is terrible, worst purchase ever.",
                "The service was okay, nothing special.",
                "Excellent customer support and fast delivery.",
                "Disappointed with the quality."
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        sentiment_data.to_csv(data_dir / "sample_sentiment_data.csv", index=False)
        
        print("  ✅ Created sample stock data")
        print("  ✅ Created sample sentiment data")
    
    def create_config_files(self):
        """Create configuration files"""
        print("⚙️  Creating configuration files...")
        
        config_dir = self.project_root / "config"
        
        # Create .env file
        env_content = """# Advanced Data Analytics Portfolio Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/analytics_db
MONGODB_URL=mongodb://localhost:27017/analytics

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Model Configuration
MODEL_CACHE_DIR=./models
MODEL_UPDATE_INTERVAL=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# External APIs
YAHOO_FINANCE_API_KEY=your_api_key_here
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
"""
        
        with open(config_dir / ".env.example", "w") as f:
            f.write(env_content)
        
        # Create config.py
        config_py_content = """# Configuration settings for the data analytics portfolio

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"

# Database Settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/analytics")

# Redis Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Model Settings
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
MODEL_UPDATE_INTERVAL = int(os.getenv("MODEL_UPDATE_INTERVAL", 3600))

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", "./logs/app.log"))

# External API Keys
YAHOO_FINANCE_API_KEY = os.getenv("YAHOO_FINANCE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Security Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-change-this")
"""
        
        with open(config_dir / "config.py", "w") as f:
            f.write(config_py_content)
        
        print("  ✅ Created .env.example")
        print("  ✅ Created config.py")
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        print("🧪 Running basic tests...")
        
        python_executable = self.get_venv_python()
        
        try:
            # Test imports
            test_script = """
import sys
import pandas as pd
import numpy as np
import streamlit as st
import fastapi
import tensorflow as tf
import torch
import plotly.graph_objects as go
import yfinance as yf

print("✅ All core dependencies imported successfully")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
"""
            
            result = subprocess.run([str(python_executable), "-c", test_script], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Test failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "=" * 80)
        print("🎉 Setup completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("\n1. Activate the virtual environment:")
        if platform.system() == "Windows":
            print(f"   {self.venv_name}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_name}/bin/activate")
        
        print("\n2. Run the dashboard:")
        print("   streamlit run dashboard/main_app.py")
        
        print("\n3. Start the API server:")
        print("   uvicorn api.main:app --reload")
        
        print("\n4. Access the applications:")
        print("   Dashboard: http://localhost:8501")
        print("   API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        
        print("\n5. Run tests:")
        print("   python -m pytest tests/")
        
        print("\n6. For Docker deployment:")
        print("   docker-compose -f deployment/docker-compose.yml up -d")
        
        print("\n📚 Documentation:")
        print("   - README.md: Project overview and setup")
        print("   - PROJECT_OVERVIEW.md: Detailed feature descriptions")
        print("   - docs/: Additional documentation")
        
        print("\n🔧 Configuration:")
        print("   - Copy config/.env.example to config/.env")
        print("   - Update API keys and settings in config/.env")
        
        print("\n" + "=" * 80)
        print("🚀 Your Advanced Data Analytics Portfolio is ready!")
        print("=" * 80)
    
    def setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create directories
        self.create_directories()
        
        # Create sample data
        self.create_sample_data()
        
        # Create config files
        self.create_config_files()
        
        # Run tests
        if not self.run_tests():
            print("⚠️  Some tests failed, but setup completed")
        
        # Print next steps
        self.print_next_steps()
        
        return True

def main():
    """Main setup function"""
    setup = PortfolioSetup()
    success = setup.setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
