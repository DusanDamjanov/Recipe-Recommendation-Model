import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


#ucitaj fajl i vrati ga kao DataFrame
def load_data():
    data = pd.read_csv("data/car_evaluation.csv")
    return data

#ocisti podatke da u svim kolonama i redovima imamo samo dozvoljene vrednosti
def clean_data(data):

    valid_values_dict = {
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', 'more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high'],
        'class': ['unacc', 'acc', 'good', 'vgood']
    }
    
    for column, valid_values in valid_values_dict.items():
        data = data[data[column].isin(valid_values)]
    return data

#ako kolona generalno ima sta sem brojeva bice tipa 'object' i onda daj je label encoderu da je enkodira u brojeve
def encode_object_columns(dataset, filename):
    label_encoders = {}
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            dataset[column] = label_encoders[column].fit_transform(dataset[column])
            if column == "class":   
                joblib.dump(label_encoders[column], "random_forest_models/car_evaluation/label_encoder" + filename + ".joblib") 
    
    return dataset

#podeli na train, test, val
def split_data(dataset):
    y = dataset['class']
    X = dataset.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

#kreiraj model
def build_model():
    model =  RandomForestClassifier(n_estimators=100, random_state=42)
    return model

#istreniraj ga
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def TRAIN(filename):
    dataset = load_data()
    print(dataset.keys())
    # print("dataset loaded......len: ", len(dataset))

    dataset_cleaned = clean_data(dataset)

    # print("dataset cleaned....len: ", len(dataset_cleaned))


    dataset_encoded = encode_object_columns(dataset_cleaned, filename)
    # print("dataset encoded")
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(dataset_encoded)
    # print("dataset splitted")
    model = build_model()
    print("beggining training...")
    model = train_model(model, X_train, y_train)

    y_pred = model.predict(X_val)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
      
    joblib_file = "random_forest_models/car_evaluation/" + filename + ".joblib"
    joblib.dump(model, joblib_file)


def CLASIFY(buying, maint, doors, persons, lug_boot, safety, filename):
    
    # Load the model
    model = joblib.load("random_forest_models/car_evaluation/" + filename + ".joblib")
    
    recipe_df = pd.DataFrame({
        "buying": [buying],
        "maint": [maint],
        "doors": [doors],
        "persons": [persons],
        "lug_boot": lug_boot,
        "safety": safety
    })
    
    # Preprocess the recipe data (this should match your preprocessing steps)
    # recipe_cleaned = clean_data(recipe_df)
    recipe_encoded = encode_object_columns(recipe_df, filename)
    
    # Make prediction
    prediction = model.predict(recipe_encoded)
    labelenc = joblib.load("random_forest_models/car_evaluation/label_encoder" + filename + ".joblib")
    prediction = labelenc.inverse_transform(prediction)
    print(prediction)
    return prediction

if __name__ == "__main__":
    dataset = load_data()
    print(dataset.keys())
    print("dataset loaded......len: ", len(dataset))

    dataset_cleaned = clean_data(dataset)

    print("dataset cleaned....len: ", len(dataset_cleaned))


    dataset_encoded = encode_object_columns(dataset_cleaned)
    print("dataset encoded")
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(dataset_encoded)
    print("dataset splitted")
    model = build_model()
    print("beggining training...")
    model = train_model(model, X_train, y_train)

    y_pred = model.predict(X_val)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))





