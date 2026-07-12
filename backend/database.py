import pyodbc
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any

class SQLServerDB:
    """SQL Server database handler for storing ML training and prediction results"""
    
    def __init__(self):
        # Database connection configuration
        self.connection_string = (
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost;'
            'DATABASE=EduLensML;'
            'Trusted_Connection=yes;'
        )
        self.connection = None
    
    def connect(self) -> bool:
        """
        Establish connection to SQL Server database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = pyodbc.connect(self.connection_string)
            print("✅ Database connected successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Database connection error: {e}")
            print("   Please check:")
            print("   1. SQL Server is running")
            print("   2. Database 'EduLensML' exists")
            return False
    
    def get_training_history(self, model_name: Optional[str] = None, limit: int = 10) -> pd.DataFrame:
        """
        Retrieve training history from database
        
        Args:
            model_name: Optional filter for specific model
            limit: Maximum number of records to return
            
        Returns:
            pandas.DataFrame: Training history records
        """
        try:
            query = f"""
                SELECT TOP {limit} ModelName, Accuracy, Precision, Recall, F1Score,
                       TrainingDuration, DatasetSize, TestSize, TrainingDate
                FROM ModelTrainingResults
            """
            if model_name:
                query += f" WHERE ModelName = '{model_name}'"
            query += " ORDER BY TrainingDate DESC"
            
            return pd.read_sql(query, self.connection)
        except Exception as e:
            print(f"Error in get_training_history: {e}")
            return pd.DataFrame()
    
    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieve prediction history from database
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            pandas.DataFrame: Prediction history records
        """
        try:
            query = f"""
                SELECT TOP {limit} PredictionDate, MidtermMarks, SessionalMarks,
                       FinalExamMarks, ClassParticipation, StudyHours, SleepHours,
                       PreviousCGPA, PredictedGrade, TotalMarks, Percentage
                FROM PredictionHistory
                ORDER BY PredictionDate DESC
            """
            return pd.read_sql(query, self.connection)
        except Exception as e:
            print(f"Error in get_prediction_history: {e}")
            return pd.DataFrame()
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Retrieve latest model comparison data
        
        Returns:
            pandas.DataFrame: Model comparison records sorted by accuracy
        """
        try:
            query = """
                SELECT ModelName, Accuracy, Precision, Recall, F1Score, TrainingDate
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY ModelName ORDER BY TrainingDate DESC) as rn
                    FROM ModelTrainingResults
                ) t
                WHERE rn = 1
                ORDER BY Accuracy DESC
            """
            return pd.read_sql(query, self.connection)
        except Exception as e:
            print(f"Error in get_model_comparison: {e}")
            return pd.DataFrame()
    
    def save_training_results(
        self, 
        model_name: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        training_duration: int,
        dataset_size: int,
        test_size: int
    ) -> bool:
        """
        Save model training results to database
        
        Args:
            model_name: Name of the trained model
            accuracy: Model accuracy score
            precision: Model precision score
            recall: Model recall score
            f1: Model F1 score
            training_duration: Training time in seconds
            dataset_size: Total dataset size
            test_size: Test set size
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO ModelTrainingResults 
                (ModelName, Accuracy, Precision, Recall, F1Score, 
                 TrainingDuration, DatasetSize, TestSize, TrainingDate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, model_name, accuracy, precision, recall, f1, 
                training_duration, dataset_size, test_size)
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving training results: {e}")
            return False
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Save student prediction results to database
        
        Args:
            prediction_data: Dictionary containing prediction data
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO PredictionHistory 
                (MidtermMarks, SessionalMarks, FinalExamMarks, ClassParticipation,
                 StudyHours, SleepHours, PreviousCGPA, PredictedGrade, 
                 TotalMarks, Percentage, Remarks, PredictionDate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """,
                prediction_data.get('midterm_marks'),
                prediction_data.get('sessional_marks'),
                prediction_data.get('final_exam_marks'),
                prediction_data.get('class_participation'),
                prediction_data.get('study_hours'),
                prediction_data.get('sleep_hours'),
                prediction_data.get('previous_cgpa'),
                prediction_data.get('predicted_grade'),
                prediction_data.get('total_marks'),
                prediction_data.get('percentage'),
                prediction_data.get('remarks')
            )
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def save_best_model_metrics(
        self,
        model_name: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        confusion_matrix_2d: List[List[int]],
        roc_auc_score: float
    ) -> bool:
        """
        Save best model metrics to database
        
        Args:
            model_name: Name of the best performing model
            accuracy: Model accuracy score
            precision: Model precision score
            recall: Model recall score
            f1: Model F1 score
            confusion_matrix_2d: 2D confusion matrix as list of lists
            roc_auc_score: ROC AUC score
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO BestModelMetrics 
                (ModelName, Accuracy, Precision, Recall, F1Score, 
                 ConfusionMatrix2D, RocAucScore, CreatedDate)
                VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, model_name, accuracy, precision, recall, f1,
                json.dumps(confusion_matrix_2d), roc_auc_score)
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving best model metrics: {e}")
            return False