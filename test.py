from tensorflow import keras
import joblib
import numpy as np
import pandas as pd

model=keras.models.load_model("D:\\NLP\\ResumeScreeningApplication\\artifacts\\training\\trained_model.h5")

text_preprocessor=joblib.load("D:\\NLP\\ResumeScreeningApplication\\artifacts\\data_preprocess\\text_preprocessor.pkl")

prediction=model.predict(input)

Categories=['Arts', 'Mechanical Engineer', 'DevOps Engineer', 'Hadoop', 'ETL Developer', 'Blockchain', 'Civil Engineer', 'Electrical Engineering', 'PMO', 'SAP Developer', 'HR', 'DotNet Developer', 'Python Developer', 'Operations Manager', 'Data Science', 'Database', 'Business Analyst', 'Web Designing', 'Testing', 'Health and fitness', 'Network Security Engineer', 'Automation Testing', 'Sales', 'Java Developer', 'Advocate']


print(Categories[np.argmax(prediction)])