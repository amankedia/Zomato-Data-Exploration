from sklearn.svm import SVC
from joblib import dump
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def buildAndSaveModel(X_train, X_test, y_train, y_test, path):

    print("Building model")
    classifier = SVC(kernel='linear', random_state=0, probability=True)
    classifier.fit(X_train, y_train)

    print("Validating model")
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: ", acc)
    precision = precision_score(y_test, y_pred)
    print("Precision: ", precision)
    recall = recall_score(y_test, y_pred)
    print("Recall: ", recall)
    f1_score_val = f1_score(y_test, y_pred)
    print("F1 Score: ", f1_score_val)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix")
    print(cm)

    dump(classifier, filename = path + "/" + 'svm_classifier.joblib')

    return acc, precision, recall, f1_score_val