import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import word_tokenize



# ucitava fajl i stvara data frame
def loadFile():
    # Učitavanje podataka iz JSON fajla
    with open('data/recipes.json', 'r') as f:
        data = json.load(f)

    # Konvertovanje JSON podataka u DataFrame
    data = pd.DataFrame(data)
    data = data.drop(columns=['url', 'id'])

    # Pregled neprocesuiranih podataka
    print("-------------NEPROCESUIRANI-----------------------")
    print(data.head())
    print("------------------------------------")

    return data

# Funkcija za predprocesiranje teksta
def preprocess_text(text, stop_words):
    if isinstance(text, list):
        processed_string = ""
        for item in text:
            if isinstance(item, dict) and 'text' in item:
                tokens = word_tokenize(item['text'])
                processed_item = ' '.join([word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words])
                processed_string += processed_item + " "

        return processed_string
    else:       #ovaj else je obavezan jer ce ovde uci za svaki title --> u prevodu ingredients i instructions resava ovo gore a title 
                #resava ovaj else
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
        return ' '.join(tokens)


#TODO: mozda ovo moze i bolje da se odradi...jer se gube neki delici oko proporcija hrane kada se podaci procesuiraju
# ucitava i deli podatke na 'train', 'val', 'test' skupove
def load():
    # Preuzimanje potrebnih resursa za NLTK
    # nltk.download('stopwords')  
    # nltk.download('punkt')
    stop_words = set(stopwords.words('english'))    #izvuci sve reci engleskog jezika koje su 'nebitne' za filtriranje


    data = loadFile()

    # Primena predprocesiranja na kolone 'ingredients', 'instructions' i 'title'
    data['ingredients'] = data['ingredients'].apply(lambda x: preprocess_text(x, stop_words))
    data['instructions'] = data['instructions'].apply(lambda x: preprocess_text(x, stop_words))
    data['title'] = data['title'].apply(lambda x: preprocess_text(x, stop_words))

    print("-------------PROCESUIRANI-----------------------")
    columns_to_show = ['title', 'ingredients', 'instructions']  
    print(data[columns_to_show].head())
    print("KOLONE", data.columns)
    print("------------------------------------")

    # Filtriranje podataka na osnovu kolone 'partition'
    train_data = data[data['partition'] == 'train']
    val_data = data[data['partition'] == 'val']
    test_data = data[data['partition'] == 'test']

    data_splits_dict = {
            'train_data': train_data,   #DATA FRAME
            'test_data': test_data,     #DATA FRAME
            'val_data': val_data        #DATA FRAME
        }
    
    #-------------------------DEBUG-----------------------------------------
    if isinstance(data_splits_dict['train_data'], pd.DataFrame):
        print("data frame je podskup")
    #-------------------------------------------------------------------------


    return data_splits_dict




# primi sve kolone podataka koji su se pre toga izdelili u ['train', 'test', 'val'] i uzme te skupove podataka isfiltrira da sadrze
# samo ['ingredients', 'title' i 'instructions'] i vrati recnik i dalje izdeljen na ['train', 'test', 'val'] 





# #usage u LSTM!!
# #TODO: OVA IMALA GRESKU OKO 1-DIM I 2-DIM ARRAYA TAKO DA TREBA PROVERITI ALI NIJE VAZNO SADA ZA NAIVA BAYES DOK RADIM
# def get_relevant_columns(train_data, test_data, val_data):
#      # Reshape arrays to 2-dimensional if they are 1-dimensional
#     train_ingredients = train_data['ingredients'].reshape(-1, 1)
#     train_title = train_data['title'].reshape(-1, 1)
#     train_instructions = train_data['instructions'].reshape(-1, 1)

#     test_ingredients = test_data['ingredients'].reshape(-1, 1)
#     test_title = test_data['title'].reshape(-1, 1)
#     test_instructions = test_data['instructions'].reshape(-1, 1)

#     val_ingredients = val_data['ingredients'].reshape(-1, 1)
#     val_title = val_data['title'].reshape(-1, 1)
#     val_instructions = val_data['instructions'].reshape(-1, 1)

#     # Concatenate the reshaped arrays along axis 1
#     train_data_relevant = np.concatenate((train_ingredients, train_title, train_instructions), axis=1)
#     test_data_relevant = np.concatenate((test_ingredients, test_title, test_instructions), axis=1)
#     val_data_relevant = np.concatenate((val_ingredients, val_title, val_instructions), axis=1)

#     relevant_data_dict = {
#             'train_data_relevant': train_data_relevant,
#             'test_data_relevant': test_data_relevant,
#             'val_data_relevant': val_data_relevant
#         }
#     return relevant_data_dict



