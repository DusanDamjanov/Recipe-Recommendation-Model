import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
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
    print(data.head())
    return data

# Funkcija za predprocesiranje teksta
def preprocess_text(text, stop_words):
    if isinstance(text, list):
        processed_list = []
        for item in text:
            if isinstance(item, dict) and 'text' in item:
                tokens = word_tokenize(item['text'])
                processed_item = ' '.join([word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words])
                processed_list.append({'text': processed_item})

            else:
                processed_item = ' '.join([word.lower() for word in word_tokenize(str(item)) if word.isalnum() and word.lower() not in stop_words])
                processed_list.append(processed_item)
        return processed_list
    else:
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

    
    columns_to_show = ['title', 'ingredients', 'instructions']  
    print(data[columns_to_show].head())
    print("KOLONE", data.columns)

    # Filtriranje podataka na osnovu kolone 'partition'
    train_data = data[data['partition'] == 'train']
    val_data = data[data['partition'] == 'val']
    test_data = data[data['partition'] == 'test']
    return tuple(train_data, val_data, test_data)






