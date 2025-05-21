# Genomic Analysis Django

A web application for interactive genomic data analysis built with Django, Scikit-learn, and Pandas.  
Users can upload CSV files containing genomic data and receive automatic analysis, including:

- **Target column detection** (classification or clustering as appropriate)
- **Data preprocessing** (imputation, scaling, encoding)
- **Feature selection** and **feature importance visualization**
- **Model training and evaluation** (Decision Tree classifier by default)
- **Class distribution and clustering visualization**
- **Confusion matrix and classification report**
- **Downloadable reports**

## Features

- **Upload CSV Genomic Data:** Easily upload your dataset via the web interface.
- **Automatic Target Detection:** The system detects the best target column for supervised tasks.
- **Classification & Clustering:** Runs classification (if suitable target found) or clustering (if not).
- **Feature Importance:** Visualizes which features are most important for prediction.
- **Data Visualization:** See bar charts of class distribution, PCA plots of clusters, decision tree visualization, and more.
- **Report Download:** Download a plain-text classification report.
- **Handles Missing Data:** Uses imputation and scaling for robust results.
- **Class Imbalance Support:** Uses SMOTE to balance classes for fairer model evaluation.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/riyaaaa19/genomic-analysis-django.git
   cd genomic-analysis-django
   ```

2. **Create a virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run migrations (if using Django models):**
   ```sh
   python manage.py migrate
   ```

5. **Start the development server:**
   ```sh
   python manage.py runserver
   ```

6. **View in your browser:**
   ```
   http://127.0.0.1:8000/
   ```

## Usage

1. Open the web app.
2. Upload your CSV genomic data file.
3. Wait for the analysis and visualizations.
4. Download results or reports if needed.

**Note:** Only numeric columns are used for analysis. The app will ignore text columns (other than the target).

## File Structure

```
genomic-analysis-django/
├── analysis/
│   ├── templates/
│   │   └── analysis/
│   │       ├── index.html
│   │       └── result.html
│   ├── static/
│   │   └── results/
│   ├── views.py
│   └── ...
├── genomics_project/
│   ├── settings.py
│   └── ...
├── requirements.txt
├── manage.py
└── README.md
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE) (or specify your license here)

---

*Created by [riyaaaa19](https://github.com/riyaaaa19)*
