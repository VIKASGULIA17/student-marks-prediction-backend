import numpy as np
import pandas as pd
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

preprocessor = joblib.load('preprocessor.pkl')
study_bins = joblib.load('study_bins.pkl')

def handle_missing_value(df):
    numerical_col = df.select_dtypes(include=['int64','float64']).columns
    for col in numerical_col:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    categorial_col = df.select_dtypes(include=['object']).columns
    for col in categorial_col:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def engineer_features(df, mode='train', study_bins=None):
    df['Total_entertainment_period'] = df['social_media_hours'] + df['netflix_hours']
    df['Study_to_entertainment_ratio'] = df['study_hours_per_day'] / (df['Total_entertainment_period'] + .1)
    df['sleep_adequacy'] = np.where(df['sleep_hours'] >= 7, 'Adequate', 'Inadequate')
    df['Life_balance_score'] = (
        (df['study_hours_per_day'] / 5) +
        (df['sleep_hours'] / 10) +
        (df['exercise_frequency'] / 7) -
        (df['Total_entertainment_period'] / 10)
    )
    if mode == 'train':
        df['study_habit_category'], study_bins = pd.qcut(
            df['study_hours_per_day'], q=4,
            labels=['Low', 'Moderate', 'High', 'Intense'],
            retbins=True
        )
    else:
        if study_bins is None:
            raise ValueError("study_bins must be provided for prediction mode.")
        df['study_habit_category'] = pd.cut(
            df['study_hours_per_day'], bins=study_bins,
            labels=['Low', 'Moderate', 'High', 'Intense'],
            include_lowest=True
        )
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 18, 21, 25, 50],
        labels=['Under 18', '18-21', '22-25', 'Over 25']
    )
    # print("Study bins:", study_bins)
    return df, study_bins

def preprocess_new_data(new_df, preprocessor, study_bins):
    new_df = handle_missing_value(new_df)
    new_df, _ = engineer_features(new_df, mode='predict', study_bins=study_bins)
    new_df.drop(columns=['social_media_hours', 'netflix_hours', 'age'], inplace=True, errors='ignore')
    if 'student_id' in new_df.columns:
        new_df = new_df.drop(['student_id'], axis=1)
    new_data_processed = preprocessor.transform(new_df)
    return new_data_processed

# data = {
#     'student_id': ['S001'],
#     'age': [20],
#     'gender': ['Male'],
#     'study_hours_per_day': [5],
#     'social_media_hours': [2],
#     'netflix_hours': [1],
#     'part_time_job': ['No'],
#     'attendance_percentage': [90],
#     'sleep_hours': [7],
#     'diet_quality': ['Good'],
#     'exercise_frequency': [3],
#     'parental_education_level': ["Bachelor"],
#     'internet_quality': ['Good'],
#     'mental_health_rating': [8],
#     'extracurricular_participation': ['Yes'],
# }

# test_df = pd.DataFrame(data)

# preprocess_new_data(test_df,preprocessor,study_bins)