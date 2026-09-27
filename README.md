# House Price Prediction

A comprehensive end-to-end machine learning project for predicting house prices. This project includes a full pipeline ranging from exploratory data analysis and data preprocessing to model training, evaluation, and deployment using a Streamlit dashboard.

## 🚀 Features
- **Data Exploration & EDA**: Detailed Jupyter notebooks covering initial exploration, visualizations, and insights.
- **Robust Preprocessing Pipeline**: Custom Python scripts for data cleaning, outlier removal, and feature engineering.
- **Model Training & Optimization**: Implementation of various machine learning models with hyperparameter tuning.
- **Experiment Tracking**: Integrated with **MLflow** for tracking model parameters, metrics, and artifacts.
- **Interactive Dashboard**: A **Streamlit** web application for real-time house price predictions.
- **Containerization**: Fully Dockerized environment using `docker-compose` for seamless setup and reproducibility.

## 📁 Project Structure

```
├── dashboard/               # Streamlit application for the UI
├── data/                    # Raw and processed datasets
├── models/                  # Saved machine learning models
├── notebooks/               # Step-by-step Jupyter notebooks
│   ├── 01_exploration.py
│   ├── 02_eda.py
│   ├── 03_modeling.py
│   ├── 04_validation_optimization.py
│   ├── 05_evaluation.py
│   └── 06_interpretation.py
├── src/                     # Source code for the ML pipeline
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── remove_outliers.py
│   ├── save_model.py
│   └── split_data.py
├── mlruns/                  # MLflow tracking directory
├── Dockerfile               # Docker image configuration
├── docker-compose.yaml      # Docker Compose setup for the Streamlit app
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

## 🛠️ Tech Stack
- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: MLflow
- **Web App**: Streamlit
- **Containerization**: Docker, Docker Compose

## ⚙️ Quick Start

### Prerequisites
Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your machine.

### Running the Application
1. **Clone the repository:**
   ```bash
   git clone https://github.com/BOUCHAIB-EL-HAIDI/A2_S1_B2_House_Price_Prediction.git
   cd A2_S1_B2_House_Price_Prediction
   ```

2. **Start the environment using Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the Application:**
   Open your browser and navigate to `http://localhost:8501` to use the Streamlit dashboard.

### Running Locally (Without Docker)
If you prefer to run it locally without Docker:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 📊 Experiment Tracking
To view the MLflow UI and compare different model runs, run the following command in your terminal:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000`.
