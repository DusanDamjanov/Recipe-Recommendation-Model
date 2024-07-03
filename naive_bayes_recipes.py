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
import joblib

UNIQUE_TOKEN_TITLE = "<TIT>"
UNIQUE_TOKEN_INGREDIENTS = "<ING>"
UNIQUE_TOKEN_INSTRUCTIONS = "<INS>"
UNIQUE_TOKEN_END = "*" 
MAX_RECIPE_LEN = 2000 
TARGET = "cuisine"
FEATURES = "merged_recipe"

def process_json(dict): 
    ingredients_list = []
    for ingredient in dict['ingredients']:
        ingredients_list.append(ingredient['text'])
    
    dict['ingredients'] = ingredients_list

    manual = ""
    for instruction in dict['instructions']:
        manual += instruction['text'] + "\n"
    
    dict['instructions'] = manual[:-1]

#otvara .json fajl, ucita iz fajla, preprocesuje svaki recept iz fajla [process_json] i za kraj sve to strpa u listu 
def load_dataset(silent = True):
    recipes_dataset = []

    with open('data/modified.json', 'r') as f:
        recipes_list = json.load(f)
        recipes_keys = [key for key in recipes_list[0]]
        recipes_keys.sort()

        for recipe in recipes_list:
            process_json(recipe)
            recipes_dataset.append(recipe)

        if silent is False:
            print('===========================================')
            print('Number of examples: ', len(recipes_list), '\n')
            print('Example object keys:\n', recipes_keys, '\n')
            print('Example object:\n', recipes_list[0], '\n')
            print('Required keys:\n')
            print('  title: ', recipes_list[0]['title'], '\n')
            print('  ingredients: ', recipes_list[0]['ingredients'], '\n')
            print('  instructions: ', recipes_list[0]['instructions'])
            print('\n\n')
    
    return recipes_dataset

#pretvara recept u jedinstven string kako bi to sve trebalo otprilike da izgleda a pritom i uvodi u taj string posebne tokene od gore
def merge_recipe_string(recipe):    #TODO: OVU FUNKCIJU POSLE AKO MOZES MALO SAMO OVAJ FORMATI IZMENI I ZBOG PLAGIJATA I INACE ODVRATNO JE
    title_format = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')

    ingredients_format = ""
    for ingredient in ingredients:
        ingredients_format += " - " + ingredient + "\n"
    
    instruction_format = ""
    for instruction in instructions:
        instruction_format += " - " + instruction + "\n"
    
    return UNIQUE_TOKEN_TITLE + " " + title_format + "\n\n" + UNIQUE_TOKEN_INGREDIENTS + "\n\n" + ingredients_format + "\n" + UNIQUE_TOKEN_INSTRUCTIONS + "\n\n" + instruction_format

def insert_merged_in_dataset(merged_recipes, dataset):
    for i in range(len(dataset)):
        dataset[i]['merged_recipe'] = merged_recipes[i]
    
    return dataset

def transform_data(dataset):
    df = pd.DataFrame(dataset)
    columns_to_remove = ["url", "id", "title", "ingredients", "instructions", "partition", "level"]
    df = df.drop(columns=columns_to_remove)
    print(df.keys())
    return df


def preprocess_data_frame(df):
    
    X_train, X_test, y_train, y_test = train_test_split(df[['merged_recipe']], df['cuisine'], test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def create_preprocessor(features):

    preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), features)
        # ('num', 'passthrough', features)
    ])

    return preprocessor

def create_pipeline():
    preprocessor = create_preprocessor(FEATURES)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ComplementNB())        #bolji za brojeve, complement je bolji za tekstove
    ])
    return pipeline

def train(pipeline, x_tr, y_tr, y_val, X_val):
    print("training starting...")

    # print(x_tr.head())
    # print(y_tr.head())
    # print("Shape of training data:", x_tr.shape)
    # print("Shape of expected data:", y_tr.shape)


    pipeline.fit(x_tr, y_tr)

    print("training finished")

    y_pred = pipeline.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
    print(f'Accuracy of the model: {accuracy * 100:.2f}%')
    return pipeline

def predict_outcome(pipeline, ingredients, instructions):
    input_data = pd.DataFrame([[ingredients, instructions]], columns=FEATURES)
    cuisine = pipeline.predict(input_data)[0]
    return cuisine

if __name__ == "__main__":
    dataset = load_dataset()
    recipes_merged = [merge_recipe_string(recipe) for recipe in dataset]
    
    dataset = insert_merged_in_dataset(recipes_merged, dataset)


    df = transform_data(dataset)

    x_tr, x_test, X_val, y_tr, y_tst, y_val = preprocess_data_frame(df)

    pipeline = create_pipeline()
    train(pipeline, x_tr, y_tr, y_val, X_val)

    # new_glucose = 41
    # new_blood_pressure = 86
    # predicted_outcome = predict_outcome(pipeline, new_glucose, new_blood_pressure)
    # print(predicted_outcome)  OVO NIJE RESENO DOK NE USPOSTAVIM DA MI MODEL DAJE UVEK ING I INS
