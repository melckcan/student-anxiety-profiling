# Student Anxiety Profiling System 

A rule-based expert system that profiles university students 
into 7 distinct anxiety risk categories to support counselors 
in delivering personalized interventions.

Developed as an academic research project at the Istanbul Technical University 
University, analyzing 77 students using DASS-21 anxiety assessments.

## What It Does

- Classifies students into 7 risk profiles (Academic Risk, 
  Financial Stress, Trauma/Bullying, Relationship Anxiety, 
  Digital Stress, Low Risk, High Risk Unclassified)
- Uses Waterfall Logic — prioritizes the most critical risk 
  factor first (e.g., trauma overrides academic issues)
- Generates personalized intervention recommendations instantly
- Achieves R² = 0.456, explaining 45.6% of anxiety variation

## Tech Stack

Python 3.9 · Pandas · NumPy · Scikit-learn · 
Streamlit · Matplotlib · Seaborn

## How to Run

pip install -r requirements.txt
streamlit run app.py

## Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.456 |
| Silhouette Score | 0.247 |
| Students Classified | 60%+ |

## Clinical Note

This system supports clinicians — it does not replace them.

## Authors

Didem Uzun & Melek Can — ITU Data Science & Analytics
