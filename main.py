import naive_bayes_salary as naive
import random_forest_car_eval as rf
import random_forest_recipes as RFRECIPES
import random_forest_car_eval as RF
import LSTM as LSTM
import naive_bayes_salary as NB
import naive_bayes_recipes as NBRECIPES
import os
import tensorflow as tf
from LSTM import RecipeLSTM

def list_directory_contents(directory_path):
    try:
        contents = os.listdir(directory_path)
        for file in contents:
            print("File: ", file)
    except FileNotFoundError:
        return f"Error: The directory '{directory_path}' does not exist."
    except PermissionError:
        return f"Error: You do not have permission to access '{directory_path}'."

def print_menu():
    print("===============MENU================")
    print("1. Train new LSTM model")
    print("2. Train new Naive Bayes (recipe) model")
    print("3. Train new Naive Bayes (salary) model")
    print("4. Train new Random Forest (recipe) model")  #TODO CONNECT TO MAIN LSTM
    print("5. Train new Random Forest (car_evaluation model")   #TODO: END GENERATING FUNCTION HERE
    print("6. Generate recipe")
    print("7. List all LSTM models")
    print("8. List all Naive Bayes (recipe) models")
    print("9. List all Naive Bayes (salary) models")
    print("10. List all Random Forest (car evaluation) models")
    print("11. List all Random Forest (recipe) models")
    print("12. Evaluate salary")
    print("13. Evaluate cars")
    print("X. Exit app")

def menu():
    option = ""
    while option.upper() != "X":
        print()
        print()
        print_menu()
        option = input("Select your option: ")
        if option == "1":
            filename = input("Please enter filename: ")
            LSTM.TRAIN(filename)
        elif option == "2":
            filename = input("Please enter filename")
            NBRECIPES.TRAIN(filename)
        elif option == "3":
            filename = input("Please enter filename: ")
            NB.TRAIN(filename)
        elif option == "4":
            filename = input("Please enter filename: ")
            RFRECIPES.TRAIN(filename)
        elif option == "5":
            filename = input("Please enter filename: ")
            RF.TRAIN(filename)
        elif option == "6":
            filename_lstm = input("Enter which lstm model to use: ")
            temperature_lstm = input("Enter lstm temperature: [0.5, 0.7, 1, 1.5]")
            start_text = input("Enter starting text for lstm: [Should start with <TIT> ] ")
            maxlen_lstm = input("Enter maximum lenght for recipe: ")
            filename_rf = input("Enter which random forest model to use: ")
            filename_nb = input("Enter which naive bayes model to use: ")
            generated_recipe = LSTM.GENERATE(filename_lstm, start_text, int(maxlen_lstm), float(temperature_lstm))
            generated_recipe = RFRECIPES.CLASIFY(generated_recipe, filename_rf)
            NBRECIPES.GENERATE(filename_nb, generated_recipe)
            pass
        elif option == "7":
            list_directory_contents("lstm_models")
        elif option == "8":
            list_directory_contents("naive_bayes_models/recipes")
        elif option == "9":
            list_directory_contents("naive_bayes_models/salary") 
        elif option == "10":
            list_directory_contents("random_forest_models/car_evaluation")
        elif option == "11":
            list_directory_contents("random_forest_models/recipes")
        elif option == "12":
            filename = input("Please enter filename: ")
            age = input("Please enter age [0+]: ")
            workclass = input("Please enter workclass[Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked]: ")
            fnlwgt = input("Please enter fnlwgt[0+]: ")
            education = input("Please enter education[Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool]: ")
            education_num = input("Please enter education number [0+]: ")
            marital_status = input("Please enter marital status [Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.]: ")
            occupation = input("Please enter occupation [Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.]: ")
            relationship = input("Please enter relationship [Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.]: ")
            race = input("Please enter race [White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.]: ")
            sex = input("Please enter sex [Male, Female]: ")
            cgain = input("Please enter cgain [0+]: ")
            closs = input("Please enter closs [0+]: ")
            hr_per_week = input("Please enter hours per week [0+]: ")
            country = input("Please enter country [United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.]: ")
            NB.PREDICT(filename, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, cgain, closs, hr_per_week, country)
        elif option == "13":
            filename = input("Please enter filename: ")
            buying = input("Please enter buying price [vhigh, high, med, low] ")
            maint = input("Please enter maintnance price [vhigh, high, med, low]: ")
            doors = input("Please enter doors [2, 3, 4, more]: ")
            persons = input("Please enter persons [2, 4, more]: ")
            lug_boot = input("Please enter lug_boot [small, med, big]: ")
            safety = input("Please enter safety [low, med, high]: ")
            RF.CLASIFY(buying, maint, doors, persons, lug_boot, safety, filename)

        


if __name__=="__main__":
    menu()





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

