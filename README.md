# DataLab - Final Year Project

A comprehensive data analysis and machine learning platform built with Flask and Python.

## 🚀 Features

- **Data Upload & Management**: Support for CSV, Excel, and JSON files
- **Data Preprocessing**: Advanced data cleaning and transformation using PyCaret
- **Data Visualization**: Interactive charts and plots using Plotly, Matplotlib, and Seaborn
- **AI Assistant**: OpenAI-powered chat assistant for data analysis guidance
- **User Authentication**: Firebase-based user authentication system
- **Data Quality Analysis**: Comprehensive data quality reports and statistics
- **Exploratory Data Analysis (EDA)**: Automated EDA with univariate, bivariate, and multivariate analysis

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.10)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite, Firebase
- **Machine Learning**: PyCaret, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **AI Integration**: OpenAI API
- **Authentication**: Firebase Admin SDK

## 📋 Prerequisites

- Python 3.10
- Git
- Firebase account and credentials
- OpenAI API key (optional, for AI features)

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Umerkhalid1999/Final-Year-Project.git
   cd Final-Year-Project
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv310
   venv310\Scripts\activate

   # Linux/Mac
   python3 -m venv venv310
   source venv310/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   cd Final_data/DataLab
   pip install -r requirements.txt
   ```

4. **Configure Firebase**:
   - Place your Firebase credentials JSON file in the `templates/` directory
   - Update the filename in `config.py` or `main.py` if different

5. **Set environment variables** (optional):
   ```bash
   set OPENAI_API_KEY=your_openai_api_key
   set FIREBASE_CONFIG_PATH=path_to_your_firebase_credentials
   ```

6. **Run the application**:
   ```bash
   python main.py
   ```

7. **Access the application**:
   - Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
Final_data/
├── DataLab/
│   ├── main.py                 # Main Flask application
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── static/                # CSS and JavaScript files
│   ├── templates/             # HTML templates
│   ├── uploads/               # Uploaded datasets
│   ├── utils/                 # Utility modules
│   └── routes/                # Route modules
├── venv310/                   # Python 3.10 virtual environment
└── README.md                  # This file
```

## 🎯 Usage

1. **Register/Login**: Create an account or login using the authentication system
2. **Upload Data**: Upload your datasets (CSV, Excel, JSON)
3. **Explore Data**: View data previews, quality reports, and summary statistics
4. **Preprocess**: Clean and transform your data using automated suggestions
5. **Visualize**: Create interactive visualizations and charts
6. **AI Assistant**: Get help and insights using the AI-powered assistant

## 🔒 Security Notes

- Firebase credentials are excluded from version control
- Environment variables should be used for sensitive configurations
- Database files and logs are excluded from the repository

## 🚀 Quick Start Scripts

### Run without activation (no .bat/.ps1 required)

- **Option A: Use the project venv directly (PowerShell/CMD)**

  PowerShell:
  ```powershell
  & ".\venv310\Scripts\python.exe" ".\Final_data\DataLab\main.py"
  ```

  CMD:
  ```cmd
  .\venv310\Scripts\python.exe .\Final_data\DataLab\main.py
  ```

- **Option B: Use system Python (ensure deps installed)**
  ```powershell
  cd .\Final_data\DataLab
  python -m pip install -r requirements.txt
  python .\main.py
  ```

Notes:
- You don’t need to activate the virtualenv if you call its `python.exe` directly.
- If you prefer `flask run`, set `FLASK_APP=main.py` and run from `Final_data/DataLab`.

## 📝 License

This project is part of a Final Year Project and is for educational purposes.

## 👨‍💻 Author

**Umer Khalid** - [GitHub](https://github.com/Umerkhalid1999)

## 🤝 Contributing

This is a final year project. For any questions or suggestions, please open an issue or contact the author.

---

**Note**: Make sure to configure Firebase credentials and OpenAI API key before running the application. 