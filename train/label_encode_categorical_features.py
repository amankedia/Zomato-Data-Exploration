from joblib import dump
from sklearn.preprocessing import LabelEncoder

def labelEncodeListedInTypeFeature(train, test, path):

    le_listed_in_type = LabelEncoder()
    train = le_listed_in_type.fit_transform(train)
    test = le_listed_in_type.transform(test)
    dump(le_listed_in_type, filename = path + "/" + 'le_listed_in_type.joblib')
    return train, test

def labelEncodeListedInCityFeature(train, test, path):

    le_listed_in_city = LabelEncoder()
    train = le_listed_in_city.fit_transform(train)
    test = le_listed_in_city.transform(test)
    dump(le_listed_in_city, filename = path + "/" + 'le_listed_in_city.joblib')
    return train, test