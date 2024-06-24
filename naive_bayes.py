from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Treniranje CNB modela
def train_naive_bayes(X_train, y_train):
    cnb_model = ComplementNB()
    cnb_model.fit(X_train, y_train)
    return cnb_model


# ovaj deo upotrebe treba videti kasnije detaljnije kako ce funckionisati [kodovi sa gpt se ne podudaraju u potpunosti ]
# a i ovo ce se pozivati kod kreiranja novog recepta valjda
# Validacija CNB modela
# y_val_pred = cnb_model.predict(X_val)
# print('CNB Accuracy:', accuracy_score(y_val, y_val_pred))
# print('CNB F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
# print('CNB Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))

q=5