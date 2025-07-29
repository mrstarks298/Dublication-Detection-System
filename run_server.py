#!/usr/bin/env python3
"""
Startup script for the Duplicate Question Detection Flask App
"""

import os
import sys
from app import app, load_model

def setup_environment():
    """Setup the environment for the Flask app"""
    
    # Create necessary directories
    directories = ['uploads', 'templates', 'static']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
    
    # Set environment variables
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('FLASK_APP', 'app.py')
    
    print("🔧 Environment setup complete")

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'flask',
        'sentence_transformers',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n🚨 Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to start the Flask app"""
    
    print("🚀 Starting Duplicate Question Detection Flask App")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Load the model on startup
    print("\n🤖 Loading AI model...")
    try:
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        print("Please make sure you have internet connection for first-time model download")
        sys.exit(1)
    
    # Start the Flask app
    print("\n🌐 Starting Flask server...")
    print("📍 Server will be available at: http://localhost:5005")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5005)),
            debug=os.environ.get('FLASK_ENV') == 'development',
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

