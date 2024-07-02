import loader as loader
import vectorizer as vectorizer
import naive_bayes as naive
import random_forest as rf
import LSTM as LSTM
import lstmpytorch as lstmpt
import torch
import random

import subprocess
import os
import tensorflow as tf
def print_menu():
    print("===============MENU================")
    print("1. Train new LSTM model")
    print("2. Train new Naive Bayes model")
    print("3. Train new Random Forest model")
    print("4. Generate recipe")
    print("5. List all LSTM models")
    print("5. List all Naive Bayes models")
    print("7. List all Random Forest models")
    print("X. Exit app")

def menu():
    input = ""
    while input.upper() != "X":
        print_menu()
        input = input("Select your option: ")
        if input == "1":
            pass
        elif input == "2":
            pass
        elif input == "3":
            pass
        elif input == "4":
            pass
        elif input == "5":
            pass
        elif input == "6":
            pass
        elif input == "7":
            pass
        
        


if __name__=="__main__":
    pass





    # print(torch.cuda.is_available())      SAMO TEST DA LI JE CUDA AVAILABLE ---> OBRISATI POSLE

    # tokenized_recipes = lstmpt.tokenize_recipes()
    # data_splits_dict = loader.load()
    # print(data_splits_dict)
    # print(type(data_splits_dict['train_data']))
    # print(type(data_splits_dict['train_data']['title']))
    # print("\n\n\n\n\n\n")
    # # Generating new recipe text
    # seed_text = "Blueberry Cheesecake Bars"
    # next_words = 50  # Number of words to generate

    # # Training the LSTM model
    # lstm_dict = lstmpt.train_lstm(data_splits_dict, num_epochs=4, batch_size=32)
    # generated_text = lstmpt.generate_text(seed_text, next_words, lstm_dict)
    # print("Generated Text:")
    # print(generated_text)




    # vectorizer_dict = vectorizer.vectorize(data_splits_dict)
    #relevant_data_dict = loader.get_relevant_columns(data_splits_dict['train_data'], data_splits_dict['test_data'], data_splits_dict['val_data'])
    

    # nb_ingredients = naive.train_naive_bayes_ingredients(vectorizer_dict)
    # nb_instructions = naive.train_naive_bayes_instructions(vectorizer_dict)

    # naive.validate_model(nb_ingredients, vectorizer_dict['X_val'], vectorizer_dict['y_val_ingredients'], 'Ingredients')
    # naive.validate_model(nb_instructions, vectorizer_dict['X_val'], vectorizer_dict['y_val_instructions'], "Instructions")
    
    # rf_title = rf.train_random_forest(vectorizer_dict)
    # rf.validate_model_rf(rf_title, vectorizer_dict['X_val'], vectorizer_dict['y_val_title'], "Title", vectorizer_dict)



    # lstm_dict = LSTM.train_lstm(data_splits_dict)
    # seed_text = "Chocolate cake with"
    # new_recipe = LSTM.generate_text(seed_text, 50, lstm_dict)





















    # # Example list of strings (each string is a separate document)
    # documents = [
    #     "This is the first document.",
    #     "This document is the second document.",
    #     "And this is the third one.",
    #     "Is this the first document?"
    # ]

    # vectorizer = CountVectorizer()

    # # Fit the vectorizer to the documents and transform them into a document-term matrix
    # X = vectorizer.fit_transform(documents)

    # # Convert the matrix to an array and print it
    # print(X.toarray())

    # # Print the feature names (vocabulary)
    # print(vectorizer.get_feature_names_out())

