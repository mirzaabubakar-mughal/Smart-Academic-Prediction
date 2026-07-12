# EduLens – Student Academic Performance Prediction

EduLens is a web-based machine learning application that predicts students' academic performance using multiple supervised learning algorithms. The project demonstrates a complete machine learning workflow, including synthetic data generation, preprocessing, model training, evaluation, visualization, and report generation.

The system is designed for educational and research purposes to compare different machine learning models and analyze their prediction performance on student academic data.

## Features

- Generate and use a synthetic student dataset.
- Preprocess data before training.
- Balance class distribution using SMOTE.
- Reduce feature dimensions using PCA.
- Train multiple supervised learning models.
- Compare model performance using evaluation metrics.
- Predict student grades.
- Generate performance reports.
- Store training history and predictions in SQL Server.

## Machine Learning Models

- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network

## Data Visualization

The system provides several visualizations to help analyze model performance, including:

- Confusion Matrix
- ROC Curve
- Model Comparison Charts
- Grade Distribution
- Correlation Heatmap
- PCA Visualization
- t-SNE Visualization

## Tech Stack

#### Backend

- Python
- FastAPI
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

#### Frontend

- React
- HTML
- CSS

#### Database

- Microsoft SQL Server

## Project Structure

````text
```text
Smart Academic Prediction/
│
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── ml_models.py
│   ├── generate_dataset.py
│   ├── requirements.txt
│   ├── data/
│   │   └── student_data.csv
│
├── frontend/
│   └── edulens/
│       ├── package.json
│       ├── vite.config.js
│       ├── index.html
│       ├── src/
│       │   ├── main.jsx
│       │   ├── App.jsx
│       │   ├── App.css
│       │   ├── index.css
│       │   ├── components/
│       │   │   ├── Dataset.jsx
│       │   │   ├── Training.jsx
│       │   │   ├── Prediction.jsx
│       │   │   ├── Visualization.jsx
│       │   │   ├── Report.jsx
│       │   │   └── History.jsx
│       │   └── utils/
│       │       └── api.js
│
├── README.md
└── LICENSE
````

## Installation

```bash
git clone https://github.com/mirzaabubakar-mughal/Smart-Academic-Prediction.git

cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Generate the dataset:

```bash
python generate_dataset.py
```

Run the backend:

```bash
python app.py
```

Run the frontend:

```bash
cd frontend
cd edulens

npm run dev
```

## Machine Learning Pipeline

1. Load a dataset.
2. Preprocess the data.
3. Handle class imbalance using SMOTE.
4. Apply PCA for dimensionality reduction.
5. Train machine learning models.
6. Evaluate model performance.
7. Predict student grades.
8. Generate visualizations and reports.

## Dataset

This project uses a **synthetic dataset** containing academic and behavioral information such as:

- Midterm Marks
- Sessional Marks
- Final Exam Marks
- Study Hours
- Sleep Hours
- Previous CGPA
- Class Participation
- Final Grade

No real student data is used.

## Results

The application compares multiple machine learning algorithms using standard evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

It also provides visual reports for model comparison and prediction analysis.

## Future Improvements

- Deep learning models
- Explainable AI (XAI)
- Real-world educational datasets
- Upload Dataset
- User authentication
- Automated model selection

## Author

**Mirza Abubakar**

## License

This project is licensed under the **MIT License**.
