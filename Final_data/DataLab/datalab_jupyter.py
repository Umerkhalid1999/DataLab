#!/usr/bin/env python3
"""
Embedded Jupyter Server for DataLab
Provides a full Jupyter environment within the DataLab platform
"""

import os
import sys
import subprocess
import threading
import time
import socket
from pathlib import Path

class JupyterServerManager:
    def __init__(self, base_dir=None, port=8888):
        if base_dir is None:
            # Use current directory or find DataLab directory
            current_dir = Path.cwd()
            if current_dir.name == "DataLab":
                self.base_dir = current_dir
            elif (current_dir / "DataLab").exists():
                self.base_dir = current_dir / "DataLab"
            else:
                self.base_dir = current_dir
        else:
            self.base_dir = Path(base_dir).resolve()
            
        self.port = port
        self.process = None
        self.server_url = f"http://localhost:{port}"
        self.notebook_dir = self.base_dir / "notebooks"
        self.token = "datalab-jupyter-token-2024"
        
        # Ensure notebooks directory exists
        self.notebook_dir.mkdir(exist_ok=True)
        print(f"📁 Jupyter notebooks directory: {self.notebook_dir}")
        
    def find_available_port(self, start_port=8888):
        """Find an available port for Jupyter"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available ports found")
    
    def is_jupyter_installed(self):
        """Check if Jupyter is installed"""
        try:
            result = subprocess.run(['jupyter', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_jupyter(self):
        """Install Jupyter if not present"""
        print("Installing Jupyter...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'jupyter', 'jupyterlab', 'notebook'], 
                          check=True, timeout=300)
            print("✅ Jupyter installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Jupyter: {e}")
            return False
    
    def create_jupyter_config(self):
        """Create Jupyter configuration"""
        config_dir = self.base_dir / ".jupyter"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "jupyter_lab_config.py"
        config_content = f'''
# DataLab Jupyter Configuration
c = get_config()

# Server settings
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = {self.port}
c.ServerApp.open_browser = False
c.ServerApp.token = '{self.token}'
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Directory settings
c.ServerApp.root_dir = r'{self.notebook_dir}'
c.ServerApp.preferred_dir = r'{self.notebook_dir}'

# Security settings
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True

# Content settings
c.ContentsManager.allow_hidden = True

# Notebook settings
c.NotebookApp.tornado_settings = {{
    'headers': {{
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:5000"
    }}
}}
'''
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return config_file
    
    def start_server(self):
        """Start the Jupyter server"""
        if not self.is_jupyter_installed():
            print("Jupyter not found. Installing...")
            if not self.install_jupyter():
                return False
        
        # Find available port
        self.port = self.find_available_port(self.port)
        self.server_url = f"http://localhost:{self.port}"
        
        # Create config
        config_file = self.create_jupyter_config()
        
        print(f"Starting Jupyter server on {self.server_url}")
        
        try:
            # Start Jupyter Lab
            cmd = [
                sys.executable, '-m', 'jupyter', 'lab',
                '--config', str(config_file),
                '--no-browser',
                '--allow-root',
                f'--port={self.port}',
                f'--ServerApp.token={self.token}',
                f'--ServerApp.root_dir={self.notebook_dir}'
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.base_dir)
            )
            
            # Wait for server to start
            self.wait_for_server()
            print(f"✅ Jupyter server started successfully")
            print(f"🔗 Access URL: {self.server_url}?token={self.token}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Jupyter server: {e}")
            return False
    
    def wait_for_server(self, timeout=30):
        """Wait for Jupyter server to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/api", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        
        raise TimeoutError("Jupyter server failed to start within timeout")
    
    def stop_server(self):
        """Stop the Jupyter server"""
        if self.process:
            print("Stopping Jupyter server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print("✅ Jupyter server stopped")
    
    def get_embed_url(self):
        """Get URL for embedding in iframe"""
        return f"{self.server_url}/lab?token={self.token}"
    
    def is_running(self):
        """Check if server is running"""
        return self.process is not None and self.process.poll() is None
    
    def create_sample_notebook(self, model_name="Sample", model_code=""):
        """Create a sample notebook with ML model code"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {model_name} Model - DataLab Export\n",
                        "\n",
                        "This notebook contains your optimized machine learning model from DataLab.\n",
                        "\n",
                        "## Getting Started\n",
                        "1. Run each cell in order\n",
                        "2. Modify parameters as needed\n",
                        "3. Add your own analysis\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import required libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.metrics import classification_report, confusion_matrix\n",
                        "\n",
                        "print('Libraries imported successfully!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load your dataset\n",
                        "# Replace with your actual data file path\n",
                        "# df = pd.read_csv('your_dataset.csv')\n",
                        "\n",
                        "print('Add your dataset loading code here')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": model_code if model_code else "# Your ML model code will appear here\nprint('Model code from DataLab export')"
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Next Steps\n",
                        "\n",
                        "You can now:\n",
                        "- Modify the model parameters\n",
                        "- Add new features\n",
                        "- Create visualizations\n",
                        "- Export results\n",
                        "\n",
                        "Happy coding! 🚀"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        import json
        notebook_file = self.notebook_dir / f"{model_name.replace(' ', '_')}_model.ipynb"
        with open(notebook_file, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        return str(notebook_file)

# Global server instance
jupyter_manager = None

def get_jupyter_manager():
    """Get or create Jupyter server manager"""
    global jupyter_manager
    if jupyter_manager is None:
        jupyter_manager = JupyterServerManager()
    return jupyter_manager

def start_jupyter_server():
    """Start Jupyter server"""
    manager = get_jupyter_manager()
    if not manager.is_running():
        success = manager.start_server()
        if success:
            # Start in background thread
            def run_server():
                try:
                    manager.process.wait()
                except Exception as e:
                    print(f"Jupyter server error: {e}")
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
        return success
    return True

def stop_jupyter_server():
    """Stop Jupyter server"""
    manager = get_jupyter_manager()
    manager.stop_server()

def get_jupyter_embed_url():
    """Get Jupyter URL for embedding"""
    manager = get_jupyter_manager()
    return manager.get_embed_url()

if __name__ == "__main__":
    # Test the Jupyter server
    print("Testing Jupyter Server...")
    
    manager = JupyterServerManager()
    
    if manager.start_server():
        print(f"✅ Server running at: {manager.get_embed_url()}")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_server()
    else:
        print("❌ Failed to start server")
