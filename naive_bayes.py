from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# # Treniranje CNB modela
# def train_naive_bayes_instructions(vectorizer_dict):
#     print("----------------kreiranje CNB Ingredients---------------------")
#     X_train = vectorizer_dict['X_train']
#     y_train = vectorizer_dict['y_train_instructions']
#     cnb_model = ComplementNB()
#     print("-----------------treniranje CNB Instructions-------------------")
#     cnb_model.fit(X_train, y_train)
#     print("-----------------trening gotov-------------------")
#     return cnb_model

# def train_naive_bayes_ingredients(vectorizer_dict):
#     print("----------------kreiranje CNB Ingredients---------------------")
#     X_train = vectorizer_dict['X_train']
#     y_train = vectorizer_dict['y_train_ingredients']
#     cnb_model = ComplementNB()
#     print("-----------------treniranje CNB Ingredients-------------------")
#     cnb_model.fit(X_train, y_train)
#     print("------------------trening gotov-------------------------------")
#     return cnb_model


# def validate_model(model, X_val, y_val, name):
#     y_val_pred = model.predict(X_val)
#     print("-----------------------EVALUACIJA CNB-----------------------------------")
#     print(f'{name} CNB Accuracy:', accuracy_score(y_val, y_val_pred))
#     print(f'{name} CNB F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
#     print(f'{name} CNB Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))
#     print("--------------------------------------------------------------")


def loadFile():
# Load JSON data
    with open('data/recipes.json', 'r') as f:
        recipes_data = json.load(f)

    # Extract ingredients from each recipe
    categories_train = []
    ingredients_list_train = []
    categories_test = []
    ingredients_list_test = []
    categories_val = []
    ingredients_list_val = []
    
    for recipe in recipes_data:

        # title = recipe['title']
        partition = recipe['partition']
        # Extract instructions as a single string
        # instructions = '\n'.join([step['text'] for step in recipe['instructions']])
        # Extract ingredients as a list of strings

        if partition ==" train":
            categories_train.append(recipe['category'])
            ingredients_list_train.append(recipe['ingredients'])
        elif partition == "test":
            categories_test.append(recipe['category'])
            ingredients_list_test.append(recipe['ingredients'])
        else:
            categories_val.append(recipe['category'])
            ingredients_list_val.append(recipe['ingredients'])
    
    data_split_dict = {
        'categories_train': categories_train,
        'categories_test': categories_test,
        'categories_val': categories_val,
        'ingredients_train': ingredients_list_train,
        'ingredients_test': ingredients_list_test,
        'ingredients_val': ingredients_list_val
    }

    return data_split_dict

def vectorize_ingredients(data_split_dict):
    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(data_split_dict['ingredients_train'])
    Y_train = data_split_dict['categories_train']
    X_test = tfidf_vectorizer.fit_transform(data_split_dict['ingredients_test'])
    Y_test = data_split_dict['categories_test']
    X_val = tfidf_vectorizer.fit_transform(data_split_dict['ingredients_val'])
    Y_val = data_split_dict['categories_val']

    vector_dict = {
        'vectorizer': tfidf_vectorizer,
        'x_train': X_train,
        'x_test': X_test,
        'x_val': X_val,
        'y_train': Y_train,
        'y_test': Y_test,
        'y_val': Y_val
    }

    return vector_dict

def create_model():
    return ComplementNB()

def predict_category(new_ingredients, vectorizer, cnb):
    X_new = vectorizer.transform([new_ingredients])
    return cnb.predict(X_new)[0]


if __name__ == "__main__":
    data_dict = loadFile()
    vector_dict = vectorize_ingredients(data_dict)
    cnb_model = create_model()
    cnb_model.fit(vector_dict['x_train'], vector_dict['y_train'])

    print("sprovodjenje testa: ")
    prediction = cnb_model.predict(vector_dict['x_val'])
    accuracy = accuracy_score(vector_dict['y_val'], prediction)
    print(f"tacnos modela: {accuracy * 100:.2f}")
    torch.save(cnb_model, 'naive_bayes.pth')