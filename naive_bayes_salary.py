from sklearn.naive_bayes import ComplementNB, GaussianNB, CategoricalNB, BernoulliNB
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

#kolone cije podatke model uzima u razmatranje
FEATURES = ["age","workclass","fnlwgt","education","education num","marital status","occupation","relationship","race","sex","cgain","closs","hr per week","country"]
#kolona cije podatke model trazi
TARGET = "income"

#cita .csv fajl i vraca DataFrame objekat od tog celog fajla
def load_file():
    df = pd.read_csv('data/adult.csv')
    df.dropna()
    return pd.DataFrame(df)

#resice vrednosti koje nisu dobro upisane u tabeli
def handle_values(df):
    incomes = df['income']
    for income in incomes:
        if income == "<=50K":
            income = False
        else:
            income = True
    

#izdeli DataFrame na X_train, X_test, X_val, y_train, y_test, y_val
def preprocess_data_frame(dataframe):
    X = dataframe.drop('income', axis=1)
    y = dataframe['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


class DenseTransformer:
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, "toarray") else X

#pomocna fja koja kreira preprocessor (za kolone koje imaju brojevne vrednosti -> StandardScaler(); 
#                                                                 za stringove -> CountVectorizer())
def create_preprocessor():
    numeric_features = ['age', 'fnlwgt', 'education num', 'cgain', 'closs', 'hr per week']
    categorical_features = ['workclass', 'education', 'marital status', 'occupation', 'relationship', 'race', 'sex', 'country']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('to_dense', DenseTransformer())  # Convert sparse matrix to dense
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('text', categorical_transformer, categorical_features)
    ])

    return preprocessor

#kreira preprocessor, i onda kreira pipeline gde prvo podatke provlaci kroz preprocessor i onda ih daje Naive Bayesu
def create_pipeline():
    preprocessor = create_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', BernoulliNB())        #bernulijev bolji od gaussian, complement ne moze zbog negativnih vr
    ])
    return pipeline

def train(pipeline, x_tr, y_tr, y_val, X_val):
    print("training starting...")

    pipeline.fit(x_tr, y_tr)

    print("training finished")

    y_pred = pipeline.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
    print(f'Accuracy of the model: {accuracy * 100:.2f}%')
    return pipeline

#fja koja preddictuje da li ce imati platu vecu od 50K ili ne
def predict_outcome(pipeline, age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,cgain,closs,hr_per_week,country):
    input_data = pd.DataFrame([[ age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,cgain,closs,hr_per_week,country]], columns=FEATURES)
    has_diabetes = pipeline.predict(input_data)[0]
    return has_diabetes

def TRAIN(filename):
    dataframe = load_file()

    x_tr, x_test, X_val, y_tr, y_tst, y_val = preprocess_data_frame(dataframe)

    pipeline = create_pipeline()

    print("model created, starting training...")

    model = train(pipeline, x_tr, y_tr, y_val, X_val)
    joblib_file = "naive_bayes_models/salary/" + filename + ".joblib"
    joblib.dump(model, joblib_file)

def PREDICT(filename, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, cgain, closs, hr_per_week, country):

    pipeline = joblib.load("naive_bayes_models/salary/" + filename + ".joblib")
    predicted_outcome = predict_outcome(pipeline, age, workclass, fnlwgt, education, education_num,marital_status, occupation, relationship, race, sex, cgain, closs, hr_per_week,  country)
    print(predicted_outcome)


if __name__ == "__main__":
    dataframe = load_file()

    print("dataset size: ", len(dataframe))

    handle_values(dataframe)

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


    predicted_outcome = predict_outcome(pipeline, 15, "State-gov", 215646, "Bachelors", 13,"Married-civ-spouse","Other-service", "Husband", "White", "Male", 2174, 0, 40, "United States")
    print(predicted_outcome)
