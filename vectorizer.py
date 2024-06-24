from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


#  - glavni zadatak je da do sadasnje podatke koje smo tokenizovali i procesuirali i imamo u vidu teksta pretvorimo u numericke
#  vrednosti pomocu ovog vektorizera, to se desava na taj nacin da on od stringa uz pomoc fit transforma uzme i izvuce reci sve
# bez duplikata, zatim kad izvuce reci bez duplikata u jedan veliki recnik ide recenicu po recenicu i gleda koliko se puta neka
# rec pojavila u recenici. npr ako vektor ima 5 reci [i, am, person, good, guy] onda ce nasa recenica 'i am good guy' nakon f-je 
# fit_transform().toarray() izgledati [1, 1, 0, 1, 1] 

#  - koliko sam razumeo ovo ce se generalno koristiti u naive bayesu i random forestu najvise jer se oni koriste da iz ovih podataka
# predvide nesto novo. Videcu posle sta tacno predvidjam time pa samim tim da li je ime 'Recipe Name' adekvatno jer mi se cini 
# da nece biti ali trebalo bi istraziti ovaj proces jos...
def vectorize(data):

    train_data, val_data, test_data = data

    vectorizer = CountVectorizer()
    
    X_train_ingredients = vectorizer.fit_transform(train_data['Ingredients']).toarray()
    X_train_instructions = vectorizer.fit_transform(train_data['Instructions']).toarray()
    X_train_title = vectorizer.fit_transform(train_data['Title']).toarray()
    X_train = np.concatenate((X_train_title, X_train_ingredients, X_train_instructions), axis=1)
    y_train = train_data['Recipe Name']

    X_val_ingredients = vectorizer.transform(val_data['Ingredients']).toarray()
    X_val_instructions = vectorizer.transform(val_data['Instructions']).toarray()
    X_val_title = vectorizer.transform(val_data['Title']).toarray()
    X_val = np.concatenate((X_val_title, X_val_ingredients, X_val_instructions), axis=1)
    y_val = val_data['Recipe Name']

    X_test_ingredients = vectorizer.transform(test_data['Ingredients']).toarray()
    X_test_instructions = vectorizer.transform(test_data['Instructions']).toarray()
    X_test_title = vectorizer.transform(test_data['Title']).toarray()
    X_test = np.concatenate((X_test_title, X_test_ingredients, X_test_instructions), axis=1)
    y_test = test_data['Recipe Name']

    return tuple(X_train, y_train, X_val, y_val, X_test, y_test)


  


        
    