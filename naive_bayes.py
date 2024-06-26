from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Treniranje CNB modela
def train_naive_bayes_instructions(vectorizer_dict):
    print("----------------kreiranje CNB Ingredients---------------------")
    X_train = vectorizer_dict['X_train']
    y_train = vectorizer_dict['y_train_instructions']
    cnb_model = ComplementNB()
    print("-----------------treniranje CNB Instructions-------------------")
    cnb_model.fit(X_train, y_train)
    print("-----------------trening gotov-------------------")
    return cnb_model

def train_naive_bayes_ingredients(vectorizer_dict):
    print("----------------kreiranje CNB Ingredients---------------------")
    X_train = vectorizer_dict['X_train']
    y_train = vectorizer_dict['y_train_ingredients']
    cnb_model = ComplementNB()
    print("-----------------treniranje CNB Ingredients-------------------")
    cnb_model.fit(X_train, y_train)
    print("------------------trening gotov-------------------------------")
    return cnb_model


def validate_model(model, X_val, y_val, name):
    y_val_pred = model.predict(X_val)
    print("-----------------------EVALUACIJA CNB-----------------------------------")
    print(f'{name} CNB Accuracy:', accuracy_score(y_val, y_val_pred))
    print(f'{name} CNB F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
    print(f'{name} CNB Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))
    print("--------------------------------------------------------------")
