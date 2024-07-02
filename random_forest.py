# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# import consts as const
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# # Treniranje Random Forest modela
# def train_random_forest(vectorizer_dict):
#     print("------------------kreiranje random foresta-------------------")
#     X_train = vectorizer_dict['X_train']
#     y_train = vectorizer_dict['y_train_title']
    
#     #----------------------------------------------MOJ KOD--------------------------------------------------------------------------
#     rf_model = RandomForestClassifier(n_estimators=const.RF_estimators_num, random_state=const.RF_random_state, class_weight='balanced')
#     print("------------------treniranje random foresta-------------------")
#     rf_model.fit(X_train, y_train)
#     print("------------------trening gotov----------------------------")
#     return rf_model
#     #-------------------------------------------------------------------------------------------------------------------------------

#     #-------------------------------------GPT OPTIMIZACIJA PARAMETARA-------------------------------------------------------------------
#     # # Reduce dimensionality using PCA
#     # pca = PCA(n_components=100)
#     # X_train_pca = pca.fit_transform(X_train)
#     # vectorizer_dict['X_train_pca'] = X_train_pca
#     # vectorizer_dict['pca'] = pca

#     # best_rf_model = perform_grid_search(X_train_pca, y_train)
#     # print("------------------Training Complete------------------------")
#     # return best_rf_model
#     #-----------------------------------------------------------------------------------------------------------------------------------


# def validate_model_rf(model, X_val, y_val, name, vectorizer_dict):

#     #-------------------------------MOJ KOD--------------------------------------------------------
#     y_val_pred = model.predict(X_val)
#     print("-----------------------EVALUACIJA RF-----------------------------------")
#     print(f'{name} Random Forest Accuracy:', accuracy_score(y_val, y_val_pred))
#     print(f'{name} Random Forest F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
#     print(f'{name} Random Forest Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))
#     print("--------------------------------------------------------------")
#     #----------------------------------------------------------------------------------------------------

#     #------------------------------------------GPT OPTIMIZACIJA PARAMETARA----------------------------------------------------
#     # pca = vectorizer_dict['pca']
#     # X_val_pca = pca.transform(X_val)
    
#     # y_val_pred = model.predict(X_val_pca)
#     # print("-----------------------EVALUATION RF-----------------------------------")
#     # print(f'{name} Random Forest Accuracy:', accuracy_score(y_val, y_val_pred))
#     # print(f'{name} Random Forest F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
#     # print(f'{name} Random Forest Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))
#     # print("--------------------------------------------------------------")
#     #-----------------------------------------------------------------------------------------------------------------------


# #-----------------------------------------------GPT OPTIMIZACIJA PARAMETARA------------------------------------------------
# # Perform hyperparameter tuning using Grid Search
# # def perform_grid_search(X_train, y_train):
# #     param_grid = {
# #         'n_estimators': [50, 100, 200],
# #         'max_depth': [None, 10, 20, 30],
# #         'min_samples_split': [2, 5, 10],
# #         'min_samples_leaf': [1, 2, 4],
# #         'bootstrap': [True, False],
# #         'max_features': ['auto', 'sqrt', 'log2']
# #     }

# #     rf_model = RandomForestClassifier(random_state=const.RF_random_state, class_weight='balanced')
# #     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# #     grid_search.fit(X_train, y_train)

# #     print(f"Best parameters found: {grid_search.best_params_}")
# #     return grid_search.best_estimator_
# #--------------------------------------------------------------------------------------------------------------------------


FEATURES = []
TARGET = ""

def load_data():
    data = pd.read_csv("data/car_evaluation.csv")
    return data

#ocisti podatke da u svim kolonama i redovima imamo samo dozvoljene vrednosti
def clean_data(data):

    valid_values_dict = {
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', 'more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high'],
        'class': ['unacc', 'acc', 'good', 'vgood']
    }
    
    for column, valid_values in valid_values_dict.items():
        data = data[data[column].isin(valid_values)]
    return data

def encode_object_columns(dataset):
    label_encoders = {}
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            dataset[column] = label_encoders[column].fit_transform(dataset[column])
    
    return dataset

def split_data(dataset):
    y = dataset['class']
    X = dataset.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

def build_model():
    model =  RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    dataset = load_data()
    print(dataset.keys())
    print("dataset loaded......len: ", len(dataset))

    dataset_cleaned = clean_data(dataset)

    print("dataset cleaned....len: ", len(dataset_cleaned))


    dataset_encoded = encode_object_columns(dataset_cleaned)
    print("dataset encoded")
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(dataset_encoded)
    print("dataset splitted")
    model = build_model()
    print("beggining training...")
    model = train_model(model, X_train, y_train)

    y_pred = model.predict(X_val)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))





