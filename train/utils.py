import datetime
import os
from sklearn.model_selection import train_test_split

def splitData(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)

    print("There are ", (y_train[y_train == 0]).count(), "and", (y_test[y_test == 0]).count(),
          "samples belongng to class 0 in train and test set respectively")

    print("There are ", (y_train[y_train == 1]).count(), "and", (y_test[y_test == 1]).count(),
          "samples belongng to class 0 in train and test set respectively")

    return X_train, X_test, y_train, y_test

def createDirectoryForDataStorage():
    ts = datetime.datetime.now().timestamp()
    # print(str(int(ts)))
    # print(os.getcwd())
    path = os.path.join(os.getcwd() + "/" + str(int(ts)))
    print("The model and supporting files would be stored in folder named: ", path)
    os.mkdir(path)
    return path

def createFileForStoringValues(path):
    filename = path + "/" + 'file_for_storage.txt'
    return filename

def saveData(filename_for_storage, line):
    with open(filename_for_storage, 'a') as outfile:
        outfile.write(line)
        outfile.write("\n")
    return

def saveDictData(path, dict_type, dict):
    with open(path + "/" + dict_type + ".txt", 'w') as outfile:
        outfile.write(str(dict))
    return