
# DataLab Jupyter Configuration
c = get_config()

# Server settings
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False
c.ServerApp.token = 'datalab-jupyter-token-2024'
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Directory settings
c.ServerApp.root_dir = r'C:\Users\hp\Desktop\Final_data\Final_data\DataLab\notebooks'
c.ServerApp.preferred_dir = r'C:\Users\hp\Desktop\Final_data\Final_data\DataLab\notebooks'

# Security settings
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True

# Content settings
c.ContentsManager.allow_hidden = True

# Notebook settings
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:5000"
    }
}
