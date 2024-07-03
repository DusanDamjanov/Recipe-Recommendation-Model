import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# 1. Load JSON data
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 2. Extract features and target values
def extract_data(data):
    ingredients = [' '.join(item['text'] for item in recipe['ingredients']) for recipe in data]
    instructions = [' '.join(step['text'] for step in recipe['instructions']) for recipe in data]
    levels = [recipe['level'] for recipe in data]

    print("ingredients example: ", ingredients[0])
    print("instructions example: ", instructions[0])
    print("level example: ", levels[0])
    
    df = pd.DataFrame({
        'ingredients': ingredients,
        'instructions': instructions,
        'level': levels
    })
    return df

def encode_object_columns(dataset):
    label_encoders = {}
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            dataset[column] = label_encoders[column].fit_transform(dataset[column])
    
    return dataset

def split_data(dataset):
    y = dataset['level']
    X = dataset.drop('level', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val


def build_model():
    model =  RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def TRAIN(filename):
    data = load_json_file("data/modified.json")
    df = extract_data(data)
    print("dataset loaded and prepared")
    encoded_df = encode_object_columns(df)
    
    print("dataset encoded")

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(encoded_df)
    print("dataset splitted")
    model = build_model()
    print("beggining training...")
    model = train_model(model, X_train, y_train)
    y_pred = model.predict(X_val)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    
    joblib_file = "random_forest_models/recipes/" + filename + ".joblib"
    joblib.dump(model, joblib_file)



def CLASIFY(recipe, model_path):
    
    # Load the model
    model = joblib.load(model_path)
    
    # Assume the recipe is a dictionary and convert it to a DataFrame
    recipe_df = pd.DataFrame([recipe])
    
    # Preprocess the recipe data (this should match your preprocessing steps)
    recipe_encoded = encode_object_columns(recipe_df)
    
    # Make prediction
    prediction = model.predict(recipe_encoded)
    
    return prediction

if __name__ == "__main__":
    data = load_json_file("data/modified.json")
    df = extract_data(data)
    print("dataset loaded and prepared")
    encoded_df = encode_object_columns(df)
    
    print("dataset encoded")

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(encoded_df)
    print("dataset splitted")
    model = build_model()
    print("beggining training...")
    model = train_model(model, X_train, y_train)
    y_pred = model.predict(X_val)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    
    joblib_file = "rf_model.joblib"
    joblib.dump(model, joblib_file)

  