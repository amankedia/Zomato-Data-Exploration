from sklearn.preprocessing import OneHotEncoder
from joblib import dump

def oneHotEncode(train, test, features, path):

    onehotencoder = OneHotEncoder(categorical_features = features)
    train = onehotencoder.fit_transform(train).toarray()
    test = onehotencoder.transform(test).toarray()
    dump(onehotencoder, filename = path + "/" + 'onehotencoder.joblib')

    return train, test