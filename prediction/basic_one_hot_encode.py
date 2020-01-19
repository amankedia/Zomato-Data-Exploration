from joblib import load

def loadCityListingLabelEncoder():
    le_listed_in_city = load("prediction/le_listed_in_city.joblib")
    return le_listed_in_city

def loadTypeListingLabelEncoder():
    le_listed_in_type = load("prediction/le_listed_in_type.joblib")
    return le_listed_in_type

def loadOneHotEncoder():
    ohe = load("onehotencoder.joblib")
    return ohe

def oneHotEncode(array):

    le_listed_in_type = loadTypeListingLabelEncoder()
    le_listed_in_city = loadCityListingLabelEncoder()
    # print(le_listed_in_city.classes_)
    # print(le_listed_in_type.classes_)

    type = [array[5]]
    loc = [array[6]]

    array[5] = le_listed_in_type.transform(type)[0]
    array[6] = le_listed_in_city.transform(loc)[0]

    # print(array)

    ohe = load("prediction/onehotencoder.joblib")
    array =  ohe.transform([array]).toarray()


    return array


# if __name__ == '__main__':
#
#     array = [1, 0, -0.249901, -0.165918, 0.330432, 'Dine-out', 'Kammanahalli', 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#             0, 0]
#
#     array = oneHotEncode(array)
