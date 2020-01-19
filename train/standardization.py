from sklearn.preprocessing import StandardScaler
from joblib import dump

def standardizeNumericAttributes(train, test, path):

    print("Standardizing Numeric Attributes")
    sc = StandardScaler()
    train[['rate', 'votes', 'approx_cost(for two people)']] = sc.fit_transform(train[['rate', 'votes', 'approx_cost(for two people)']])
    test[['rate', 'votes', 'approx_cost(for two people)']] = sc.transform(test[['rate', 'votes', 'approx_cost(for two people)']])
    dump(sc, filename = path + "/" + 'std_scaler.bin', compress=True)

    return train, test