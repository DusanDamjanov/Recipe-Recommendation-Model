import loader as loader
import vectorizer as vectorizer
import naive_bayes as naive

from sklearn.feature_extraction.text import CountVectorizer


if __name__=="__main__":
    data_splits_dict = loader.load()
    
    #relevant_data_dict = loader.get_relevant_columns(data_splits_dict['train_data'], data_splits_dict['test_data'], data_splits_dict['val_data'])
    
    vectorizer_dict = vectorizer.vectorize(data_splits_dict)

    nb_ingredients = naive.train_naive_bayes_ingredients(vectorizer_dict)
    nb_instructions = naive.train_naive_bayes_instructions(vectorizer_dict)

    naive.validate_model(nb_ingredients, vectorizer_dict['X_val'], vectorizer_dict['y_val_ingredients'], 'Ingredients')
    naive.validate_model(nb_instructions, vectorizer_dict['X_val'], vectorizer_dict['y_val_instructions'], "Instructions")
    


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

