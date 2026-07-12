import pandas as pd
import numpy as np
import os
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)

def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic student academic performance dataset
    
    Args:
        n_samples: Number of student records to generate
        
    Returns:
        pandas.DataFrame: Synthetic student dataset
    """
    data = []
    
    for _ in range(n_samples):
        # Generate correlated features
        previous_cgpa = np.random.uniform(1.0, 4.0)
        study_hours = np.random.uniform(1, 12)
        participation = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3])
        
        # Base score influenced by previous performance and study habits
        base_score = previous_cgpa * 15 + study_hours * 2
        
        # Generate marks with some randomness
        mid = np.random.normal(base_score * 0.4, 3)
        mid = np.clip(mid, 0, 25)
        
        sessional = np.random.normal(base_score * 0.4, 3)
        sessional = np.clip(sessional, 0, 25)
        
        final = np.random.normal(base_score * 0.8, 4)
        final = np.clip(final, 0, 50)
        
        # Apply participation effect
        if participation == 'High':
            mid *= 1.05
            sessional *= 1.05
            final *= 1.05
        elif participation == 'Low':
            mid *= 0.95
            sessional *= 0.95
            final *= 0.95
        
        # Clamp after participation adjustment
        mid = np.clip(mid, 0, 25)
        sessional = np.clip(sessional, 0, 25)
        final = np.clip(final, 0, 50)
        
        # Calculate total marks and assign grade
        total = mid + sessional + final
        
        if total >= 80:
            grade = 'A'
        elif total >= 70:
            grade = 'B'
        elif total >= 60:
            grade = 'C'
        elif total >= 50:
            grade = 'D'
        else:
            grade = 'F'
        
        data.append({
            'midterm_marks': round(mid, 2),
            'sessional_marks': round(sessional, 2),
            'final_exam_marks': round(final, 2),
            'class_participation': participation,
            'study_hours_perday': round(study_hours, 2),
            'sleep_hours_perday': round(np.clip(np.random.normal(7, 1.5), 4, 10), 2),
            'previous_cgpa': round(previous_cgpa, 2),
            'grade': grade
        })
    
    return pd.DataFrame(data)

def save_dataset(df: pd.DataFrame, filepath: str = 'data/student_data.csv') -> None:
    """
    Save dataset to CSV file
    
    Args:
        df: pandas DataFrame to save
        filepath: Path where to save the CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Print dataset statistics and information
    
    Args:
        df: pandas DataFrame to analyze
    """
    print("Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print("\nGrade Distribution:")
    print(df['grade'].value_counts().sort_index())
    print("\nDataset Statistics:")
    print(df.describe())

if __name__ == '__main__':
    # Generate and save dataset
    df = generate_dataset(2000)
    save_dataset(df)
    print_dataset_info(df)