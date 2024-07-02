# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# import consts as const

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




