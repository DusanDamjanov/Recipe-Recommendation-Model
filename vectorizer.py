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


# def vectorize2(data_dict):

#     train_data = data_dict['train_data']
#     val_data = data_dict['val_data']
#     test_data = data_dict['test_data']

#     vectorizer = CountVectorizer()


#     print("-----------------vectorizer kreiran---------------")

#     X_train_ingredients = vectorizer.fit_transform(train_data['ingredients']).toarray()
#     X_train_instructions = vectorizer.fit_transform(train_data['instructions']).toarray()
#     X_train_title = vectorizer.fit_transform(train_data['title']).toarray()
#     X_train = np.concatenate((X_train_title, X_train_ingredients, X_train_instructions), axis=1)

#     print("----------train podaci vektorizovani----------------------")

#     X_val_ingredients = vectorizer.transform(val_data['ingredients']).toarray()
#     X_val_instructions = vectorizer.transform(val_data['instructions']).toarray()
#     X_val_title = vectorizer.transform(val_data['title']).toarray()
#     X_val = np.concatenate((X_val_title, X_val_ingredients, X_val_instructions), axis=1)

#     print("-----------------val podaci vektorizovani-----------------")

#     X_test_ingredients = vectorizer.transform(test_data['ingredients']).toarray()
#     X_test_instructions = vectorizer.transform(test_data['instructions']).toarray()
#     X_test_title = vectorizer.transform(test_data['title']).toarray()
#     X_test = np.concatenate((X_test_title, X_test_ingredients, X_test_instructions), axis=1)

#     print("--------------test podaci vektrizovani-----------")

#     #ono sto ce se predvidjati
#     y_train_ingredients = train_data['ingredients']
#     y_train_instructions = train_data['instructions']
    
#     y_val_ingredients = val_data['ingredients']
#     y_val_instructions = val_data['instructions']
    
#     y_test_ingredients = test_data['ingredients']
#     y_test_instructions = test_data['instructions']

#     vectorizer_dict = {
#             'X_train': X_train,
#             'X_val': X_val,
#             'X_test': X_test,
#             'y_val_ingredients': y_val_ingredients,
#             'y_val_instructions': y_val_instructions,
#             'y_test_ingredients': y_test_ingredients,
#             'y_test_instructions': y_test_instructions,
#             'y_train_ingredients': y_train_ingredients,
#             'y_train_instructions': y_train_instructions,
#             #TODO: falice jos za title kad budem dodavao u random forest
#             'vectorizer': vectorizer
#         }
#     return vectorizer_dict



def vectorize(data_dict):
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    test_data = data_dict['test_data']

    vectorizer = CountVectorizer()

    print("-----------------vectorizer kreiran---------------")

    # Combine text data into a single string
    train_combined = (train_data['title'] + ' ' + train_data['ingredients'] + ' ' + train_data['instructions']).tolist()
    val_combined = (val_data['title'] + ' ' + val_data['ingredients'] + ' ' + val_data['instructions']).tolist()
    test_combined = (test_data['title'] + ' ' + test_data['ingredients'] + ' ' + test_data['instructions']).tolist()
    print("------------podaci kombinovani--------------------")
    
    # Fit the vectorizer on the training data and transform all datasets
    X_train = vectorizer.fit_transform(train_combined).toarray()
    X_val = vectorizer.transform(val_combined).toarray()
    X_test = vectorizer.transform(test_combined).toarray()

    print("----------podaci vektorizovani----------------------")

    y_train_ingredients = train_data['ingredients']
    y_train_instructions = train_data['instructions']
    y_traing_title = train_data['title']
    
    y_val_ingredients = val_data['ingredients']
    y_val_instructions = val_data['instructions']
    y_val_title = val_data['title']
    
    y_test_ingredients = test_data['ingredients']
    y_test_instructions = test_data['instructions']
    y_test_title = test_data['title']

    vectorizer_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_val_ingredients': y_val_ingredients,
        'y_val_instructions': y_val_instructions,
        'y_val_title': y_val_title,
        'y_test_ingredients': y_test_ingredients,
        'y_test_instructions': y_test_instructions,
        'y_val_instructions': y_val_instructions,
        'y_train_ingredients': y_train_ingredients,
        'y_train_instructions': y_train_instructions,
        'y_train_title': y_traing_title,
        'vectorizer': vectorizer
    }
    return vectorizer_dict