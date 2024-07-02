from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#kolone cije podatke model uzima u razmatranje
FEATURES = ["bloodpressure", "glucose"]

#kolona cije podatke model trazi
TARGET = "diabetes"

#cita .csv fajl i vraca DataFrame objekat od tog celog fajla
def load_file():
    df = pd.read_csv('data/diabetes.csv')
    return pd.DataFrame(df)

#izdeli DataFrame na X_train, X_test, X_val, y_train, y_test, y_val
def preprocess_data_frame(dataframe):
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe[FEATURES], dataframe[TARGET], test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

#pomocna fja koja kreira preprocessor (za kolone koje imaju brojevne vrednosti -> StandardScaler(); 
#                                                                 za stringove -> CountVectorizer())
def create_preprocessor(features):
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
        # ('num', 'passthrough', features)
    ])

    return preprocessor

#kreira preprocessor, i onda kreira pipeline gde prvo podatke provlaci kroz preprocessor i onda ih daje Naive Bayesu
def create_pipeline():
    preprocessor = create_preprocessor(FEATURES)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())        #bolji za brojeve, complement je bolji za tekstove
    ])
    return pipeline

def train(pipeline, x_tr, y_tr, y_val, X_val):
    print("training starting...")

    pipeline.fit(x_tr, y_tr)

    print("training finished")

    y_pred = pipeline.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
    print(f'Accuracy of the model: {accuracy * 100:.2f}%')

#fja koja preddictuje da li dijabetes ima ili nema
def predict_outcome(glucose, blood_pressure):
    input_data = pd.DataFrame([[glucose, blood_pressure]], columns=FEATURES)
    has_diabetes = pipeline.predict(input_data)[0]
    return has_diabetes

if __name__ == "__main__":
    dataframe = load_file()

    print("dataset size: ", len(dataframe))

    x_tr, x_test, X_val, y_tr, y_tst, y_val = preprocess_data_frame(dataframe)

    print("X tr: ", len(x_tr))
    print("X test: ", len(x_test))
    print("X val: ", len(X_val))
    print("Y tr: ", len(y_tr))
    print("Y test: ", len(y_tst))
    print("Y val: ", len(y_val))

    pipeline = create_pipeline()

    print("pipeline created, starting training...")

    train(pipeline, x_tr, y_tr, y_val, X_val)

    
    new_glucose = 41
    new_blood_pressure = 86
    predicted_outcome = predict_outcome(new_glucose, new_blood_pressure)
    print(predicted_outcome)
