#!/usr/bin/env python3
"""
Setup script for Trading Dashboard
This script helps install dependencies and verify the environment
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_imports():
    """Check if all required packages can be imported"""
    packages = {
        'streamlit': 'streamlit',
        'yfinance': 'yfinance', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'plotly': 'plotly'
    }
    
    missing = []
    for import_name, package_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {package_name} is available")
        except ImportError:
            print(f"✗ {package_name} is missing")
            missing.append(package_name)
    
    return missing

def main():
    print("Trading Dashboard Setup")
    print("=" * 30)
    
    # Check current imports
    missing = check_imports()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        
        for package in missing:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    else:
        print("\n✓ All packages are available!")
    
    # Final check
    print("\nFinal verification:")
    final_missing = check_imports()
    
    if not final_missing:
        print("\n🎉 Setup complete! You can now run the trading dashboard.")
        print("Run: streamlit run app.py")
    else:
        print(f"\n⚠️  Still missing: {', '.join(final_missing)}")
        print("Please install manually or check your Python environment.")

if __name__ == "__main__":
    main()