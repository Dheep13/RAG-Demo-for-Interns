"""
RAG Demo Setup Script
====================

This script helps set up the RAG demo environment automatically.
Run this after cloning the repository.

Usage: python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n{step}. {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.output}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        print("Please upgrade Python and try again.")
        return False

def create_virtual_environment():
    """Create and activate virtual environment."""
    print_step(2, "Creating Virtual Environment")
    
    venv_name = "rag_demo_env"
    
    if Path(venv_name).exists():
        print(f"⚠️ Virtual environment '{venv_name}' already exists")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response == 'y':
            print("Removing existing environment...")
            if os.name == 'nt':  # Windows
                run_command(f"rmdir /s /q {venv_name}", "Remove existing venv")
            else:  # Unix/Linux/Mac
                run_command(f"rm -rf {venv_name}", "Remove existing venv")
        else:
            print("Using existing virtual environment")
            return True
    
    # Create virtual environment
    if run_command(f"python -m venv {venv_name}", "Create virtual environment"):
        print(f"✅ Virtual environment '{venv_name}' created successfully!")
        
        # Print activation instructions
        print("\n💡 To activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"   {venv_name}\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            print(f"   source {venv_name}/bin/activate")
        
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False

def install_packages():
    """Install required packages."""
    print_step(3, "Installing Required Packages")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command("python -m pip install --upgrade pip", "Upgrade pip")
    
    # Install packages
    print("Installing packages from requirements.txt...")
    if run_command("pip install -r requirements.txt", "Install packages"):
        print("✅ All packages installed successfully!")
        return True
    else:
        print("❌ Failed to install some packages")
        print("💡 Try installing manually: pip install langchain langchain-openai langchain-chroma")
        return False

def setup_environment_file():
    """Set up the .env file."""
    print_step(4, "Setting Up Environment File")
    
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("⚠️ .env file already exists")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("Using existing .env file")
            return True
    
    if example_file.exists():
        # Copy example file
        with open(example_file, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ .env file created from template")
        print("\n🔑 IMPORTANT: Edit the .env file and add your API keys!")
        print("   Required: OPENAI_API_KEY")
        print("   Optional: GEMINI_API_KEY, HF_TOKEN")
        
        return True
    else:
        print("❌ .env.example file not found")
        print("Creating basic .env file...")
        
        with open(env_file, 'w') as f:
            f.write("# OpenAI API Key (Required)\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n\n")
            f.write("# Optional API Keys\n")
            f.write("GEMINI_API_KEY=your-gemini-api-key-here\n")
            f.write("HF_TOKEN=your-huggingface-token-here\n")
        
        print("✅ Basic .env file created")
        return True

def create_sample_documents_folder():
    """Create sample documents folder."""
    print_step(5, "Setting Up Sample Documents Folder")
    
    sample_dir = Path("sample_documents")
    
    if not sample_dir.exists():
        sample_dir.mkdir()
        print("✅ sample_documents folder created")
    else:
        print("✅ sample_documents folder already exists")
    
    # Check if there are any documents
    pdf_files = list(sample_dir.glob("*.pdf"))
    word_files = list(sample_dir.glob("*.docx")) + list(sample_dir.glob("*.doc"))
    
    if pdf_files or word_files:
        print(f"📄 Found {len(pdf_files)} PDF and {len(word_files)} Word documents")
    else:
        print("📄 No documents found - you can add PDF/Word files later")
        print("💡 The demo will work with built-in sample data")
    
    return True

def run_verification():
    """Run the setup verification."""
    print_step(6, "Verifying Setup")
    
    if Path("test_setup.py").exists():
        print("Running setup verification...")
        if run_command("python test_setup.py", "Verify setup"):
            print("✅ Setup verification completed!")
            return True
        else:
            print("⚠️ Some verification tests failed")
            print("💡 Check the output above for specific issues")
            return False
    else:
        print("⚠️ test_setup.py not found - skipping verification")
        return True

def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! 🎉")
    
    print("📋 Next Steps:")
    print("\n1. 🔑 Add your OpenAI API key to the .env file:")
    print("   - Edit .env file")
    print("   - Replace 'your-openai-api-key-here' with your actual key")
    print("   - Get key from: https://platform.openai.com/api-keys")
    
    print("\n2. 📄 (Optional) Add sample documents:")
    print("   - Copy PDF/Word files to sample_documents/ folder")
    print("   - Or use the built-in sample data")
    
    print("\n3. 🚀 Run the demos:")
    print("   - Interactive notebook: jupyter notebook Comprehensive_RAG_Demo.ipynb")
    print("   - Embedding analysis: python Enhanced_RAG_Demo_with_Embeddings.py")
    print("   - Complete demo: python multimodal_rag_demo.py")
    
    print("\n4. 🧪 Test your setup:")
    print("   - Run: python test_setup.py")
    print("   - This will verify everything is working")
    
    print("\n📚 Documentation:")
    print("   - Setup guide: docs/SETUP_GUIDE.md")
    print("   - Troubleshooting: docs/TROUBLESHOOTING.md")
    print("   - Learning path: docs/LEARNING_PATH.md")
    
    print("\n💡 Tips:")
    print("   - Start with the Jupyter notebook for best learning experience")
    print("   - Use virtual environment for clean package management")
    print("   - Check docs/ folder for detailed guides")

def main():
    """Main setup function."""
    print_header("RAG Demo Setup")
    print("This script will help you set up the RAG demo environment.")
    print("Make sure you have Python 3.8+ installed.")
    
    # Check if user wants to continue
    response = input("\nDo you want to continue with setup? (y/n): ").lower()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        check_python_version,
        create_virtual_environment,
        install_packages,
        setup_environment_file,
        create_sample_documents_folder,
        run_verification
    ]
    
    failed_steps = []
    
    for step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_func.__name__)
        except Exception as e:
            print(f"❌ Error in {step_func.__name__}: {e}")
            failed_steps.append(step_func.__name__)
    
    # Summary
    if failed_steps:
        print_header("Setup Completed with Issues")
        print("❌ Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n💡 Check the error messages above and:")
        print("   - See docs/TROUBLESHOOTING.md for solutions")
        print("   - Try running individual steps manually")
        print("   - Contact support if issues persist")
    else:
        print_next_steps()

if __name__ == "__main__":
    main()
