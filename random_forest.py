from sklearn.ensemble import RandomForestClassifier
import consts as const

# Treniranje Random Forest modela
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(const.RF_estimators_num, const.RF_random_state)
    rf_model.fit(X_train, y_train)


# ovaj deo upotrebe treba videti kasnije detaljnije kako ce funckionisati [kodovi sa gpt se ne podudaraju u potpunosti ]
# a i ovo ce se pozivati kod kreiranja novog recepta valjda
# # Validacija Random Forest modela
# y_val_pred = rf_model.predict(X_val)
# print('RF Accuracy:', accuracy_score(y_val, y_val_pred))
# print('RF F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
# print('RF Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))
