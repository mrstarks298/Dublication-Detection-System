#!/usr/bin/env python3
"""
Setup and Run Script for Smart Question Similarity Checker
This script helps set up the environment and run the Flask application.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'uploads',
        'outputs', 
        'csv_files',
        'templates',
        'static'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def create_template_file():
    """Create the HTML template file in the templates directory"""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Question Similarity Checker</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <div class="container mx-auto px-6 py-8">
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🧠 Smart Question Similarity Checker</h1>
            <p class="text-gray-600">Intelligently match questions to topic-specific databases and find similarities using AI</p>
        </div>
        
        <div class="bg-white rounded-xl shadow-lg p-8 text-center">
            <div class="text-6xl mb-4">🚀</div>
            <h2 class="text-2xl font-bold text-gray-800 mb-4">Welcome to Smart Question Checker!</h2>
            <p class="text-gray-600 mb-6">Upload your CSV files and JSON questions to get started with intelligent similarity analysis.</p>
            
            <div class="grid md:grid-cols-2 gap-6 mt-8">
                <div class="bg-blue-50 p-6 rounded-lg">
                    <h3 class="font-bold text-blue-800 mb-2">📚 CSV Database</h3>
                    <p class="text-blue-700 text-sm">Upload multiple CSV files containing questions with embeddings</p>
                </div>
                <div class="bg-green-50 p-6 rounded-lg">
                    <h3 class="font-bold text-green-800 mb-2">📝 JSON Questions</h3>
                    <p class="text-green-700 text-sm">Upload new questions with chapter hierarchy for smart matching</p>
                </div>
            </div>
            
            <div class="mt-8 p-4 bg-gray-50 rounded-lg">
                <p class="text-sm text-gray-600">
                    <strong>Note:</strong> This is a basic template. Please copy the full HTML template from the artifacts to get the complete interface.
                </p>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(template_content)
    print("✓ Created basic template file (replace with full template from artifacts)")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def main():
    print("🚀 Setting up Smart Question Similarity Checker")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create basic template
    create_template_file()
    
    # Install requirements
    if os.path.exists('requirements.txt'):
        if not install_requirements():
            print("Please install requirements manually using:")
            print("pip install -r requirements.txt")
    else:
        print("⚠ requirements.txt not found. Please create it first.")
    
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    print("\nNext steps:")
    print("1. Replace templates/index.html with the full template from artifacts")
    print("2. Make sure app.py is in the current directory")
    print("3. Run the application with: python app.py")
    print("4. Open http://localhost:5000 in your browser")
    print("\nFolder structure created:")
    print("  📁 uploads/     - For uploaded files")
    print("  📁 outputs/     - For generated results")
    print("  📁 csv_files/   - For CSV database files")
    print("  📁 templates/   - For HTML templates")
    
    # Ask if user wants to start the server
    try:
        start_server = input("\nDo you want to start the Flask server now? (y/N): ").lower().strip()
        if start_server == 'y':
            if os.path.exists('app.py'):
                print("Starting Flask server...")
                subprocess.run([sys.executable, 'app.py'])
            else:
                print("app.py not found. Please create it first.")
    except KeyboardInterrupt:
        print("\nSetup completed. You can start the server later with: python app.py")

if __name__ == "__main__":
    main()