from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from preprocessing import preprocess_new_data
import traceback



# Load model and preprocessors once
model = joblib.load('best_student_exam_score_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
study_bins = joblib.load('study_bins.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    studentId: str
    age: int
    gender: str
    studyHours: int
    socialMediaHours: int
    netflixHours: int
    partTimeJob: str
    attendance: int
    sleepHours: int
    dietQuality: str
    exerciseFrequency: int
    parentalEducation: str
    internetQuality: str
    mentalHealthRating: int
    extracurricular: str

@app.post("/predict")
def predict(data: StudentData):
    try:
        input_df = pd.DataFrame([{
            'student_id': data.studentId,
            'age': data.age,
            'gender': data.gender,
            'study_hours_per_day': data.studyHours,
            'social_media_hours': data.socialMediaHours,
            'netflix_hours': data.netflixHours,
            'part_time_job': data.partTimeJob,
            'attendance_percentage': data.attendance,
            'sleep_hours': data.sleepHours,
            'diet_quality': data.dietQuality,
            'exercise_frequency': data.exerciseFrequency,
            'parental_education_level': data.parentalEducation,
            'internet_quality': data.internetQuality,
            'mental_health_rating': data.mentalHealthRating,
            'extracurricular_participation': data.extracurricular
        }])

        processed_input = preprocess_new_data(input_df, preprocessor, study_bins)
        prediction = model.predict(processed_input)[0]
        predicted_score = np.clip(prediction, 0, 100)

        return {"predicted_score": round(predicted_score, 2)}

    except Exception as e:
        print("🔥 Exception in /predict:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



