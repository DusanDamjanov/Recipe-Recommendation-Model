import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

UNIQUE_TOKEN_TITLE = "<TIT>"
UNIQUE_TOKEN_INGREDIENTS = "<ING>"
UNIQUE_TOKEN_INSTRUCTIONS = "<INS>"
UNIQUE_TOKEN_END = "*" 
MAX_RECIPE_LEN = 2000 

# # 1. Load JSON data
# def load_json_file(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data

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

# # 2. Extract features and target values
# def extract_data(data):
#     ingredients = [' '.join(item['text'] for item in recipe['ingredients']) for recipe in data]
#     instructions = [' '.join(step['text'] for step in recipe['instructions']) for recipe in data]
#     levels = [recipe['level'] for recipe in data]

#     print("ingredients example: ", ingredients[0])
#     print("instructions example: ", instructions[0])
#     print("level example: ", levels[0])
    
#     df = pd.DataFrame({
#         'ingredients': ingredients,
#         'instructions': instructions,
#         'level': levels
#     })
#     return df

def encode_object_columns(dataset):
    dataset = pd.DataFrame(dataset)
    columns_to_remove = ["url", "id", "title", "ingredients", "instructions", "partition", "cuisine"]
    if "id" in dataset.keys():
        dataset = dataset.drop(columns=columns_to_remove)
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


# def TRAIN(filename):
#     data = load_json_file("data/modified.json")
#     df = extract_data(data)
#     print("dataset loaded and prepared")
#     encoded_df = encode_object_columns(df)
    
#     print("dataset encoded")

#     X_train, X_test, X_val, y_train, y_test, y_val = split_data(encoded_df)
#     print("dataset splitted")
#     model = build_model()
#     print("beggining training...")
#     model = train_model(model, X_train, y_train)
#     y_pred = model.predict(X_val)

#     # Evaluate the model
#     print("Accuracy:", accuracy_score(y_val, y_pred))
#     print("\nClassification Report:\n", classification_report(y_val, y_pred))

    
#     joblib_file = "random_forest_models/recipes/" + filename + ".joblib"
#     joblib.dump(model, joblib_file)



def CLASIFY(recipe, filename):
    recipe_copy = recipe
    # Load the model
    model = joblib.load("random_forest_models/recipes/" + filename + ".joblib")
    
    # Assume the recipe is a dictionary and convert it to a DataFrame
    recipe_df = pd.DataFrame({
    "merged_recipe": [recipe]
    })
    
    # Preprocess the recipe data (this should match your preprocessing steps)
    recipe_encoded = encode_object_columns(recipe_df)
    
    # Make prediction
    prediction = model.predict(recipe_encoded)

    print("\nLevel:" , prediction)
    return recipe_copy

if __name__ == "__main__":

    # dataset = load_dataset()
    # recipes_merged = [merge_recipe_string(recipe) for recipe in dataset]
    # dataset = insert_merged_in_dataset(recipes_merged, dataset)

    # # data = load_json_file("data/modified.json")
    # # df = extract_data(data)
    # print("dataset loaded and prepared")
    # encoded_df = encode_object_columns(dataset)
    
    # print("dataset encoded")

    # X_train, X_test, X_val, y_train, y_test, y_val = split_data(encoded_df)
    # print("dataset splitted")
    # model = build_model()
    # print("beggining training...")
    # model = train_model(model, X_train, y_train)
    # y_pred = model.predict(X_val)

    # # Evaluate the model
    # print("Accuracy:", accuracy_score(y_val, y_pred))
    # print("\nClassification Report:\n", classification_report(y_val, y_pred))

    
    # joblib_file = "rf_model.joblib"
    # joblib.dump(model, joblib_file)

    CLASIFY("<TIT> Worlds Best Mac and Cheese", "rf_model")

  