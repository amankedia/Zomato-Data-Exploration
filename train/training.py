import pandas as pd
from train.attribute_validation_and_cleaning import onlineOrderMapping, tableBookingMapping
from train.missing_data_numerical import replaceNewAndHyphenWithNaNInRatings, replaceNaNWithMeanRatingAndConvertToFloat, removeCommaAndReplaceNaNWithMeanForApproxCost
from train.standardization import standardizeNumericAttributes
from train.missing_data_categorical import fillMissingCuisineWithMostFrequentCuisineType, fillMissingRestaurantTypeWithMostFrequentRestaurantType
from train.cuisine_and_restaurant_dict_creation import createAndSaveCuisineDict, createAndSaveRestaurantDict
from train.cuisine_and_restaurant_one_hot_encode import oneHotEncodeCuisine, oneHotEncodeRestaurant
from train.label_encode_categorical_features import labelEncodeListedInTypeFeature, labelEncodeListedInCityFeature
from train.one_hot_encode_categorical_features import oneHotEncode
from train.build_and_save_model import buildAndSaveModel
from train.utils import splitData, createDirectoryForDataStorage, createFileForStoringValues

def train_classifier(data):

    df = pd.read_csv(data)

    path = createDirectoryForDataStorage()

    filename_for_storage = createFileForStoringValues(path)

    #Drop columns which are not required
    df.drop("url", axis=1, inplace=True)
    df.drop("address", axis=1, inplace=True)
    df.drop("name", axis=1, inplace=True)
    df.drop("phone", axis=1, inplace=True)

    df.drop("dish_liked", axis=1, inplace=True)
    df.drop("reviews_list", axis=1, inplace=True)
    df.drop("menu_item", axis=1, inplace=True)
    df.drop("location", axis=1, inplace=True)

    df["online_order"] = onlineOrderMapping(df["online_order"])
    df["book_table"] = tableBookingMapping(df["book_table"])

    print("Analysis values in the Rating column")
    print("Entries with hyphen in Rating", df.rate[df.rate == '-'].count())
    print("Entries with NEW in Rating", df.rate[df.rate == 'NEW'].count())
    print("Entries with nan in Rating", df.rate.isnull().sum())

    df["rate"] = replaceNewAndHyphenWithNaNInRatings(df["rate"])

    y = df.pop('clusters')

    print("Splitting Data")
    X_train, X_test, y_train, y_test = splitData(df, y)

    '''
    Cleaning, Filling NaN and Standardizing Numeric attributes rating, approx cost and votes next
    '''

    print("Replacing NaN values in rating and approx cost with mean values of training data respectively")
    X_train["rate"], X_test["rate"] = replaceNaNWithMeanRatingAndConvertToFloat(X_train["rate"], X_test["rate"], filename_for_storage)

    X_train["approx_cost(for two people)"], X_test["approx_cost(for two people)"] = removeCommaAndReplaceNaNWithMeanForApproxCost(X_train["approx_cost(for two people)"], X_test["approx_cost(for two people)"], filename_for_storage)

    print("Standardizing numeric attributes")
    X_train, X_test = standardizeNumericAttributes(X_train, X_test, path)

    '''
    Cleaning, Filling NaN and Building Custom One Hot Encoding For Cuisines and Rest_type attributes next
    '''
    df_cuisines = df[df['cuisines'].notnull()]

    cuisines_set, cuisine_dict = createAndSaveCuisineDict(df_cuisines['cuisines'], path)

    X_train['cuisines'], X_test['cuisines'] = fillMissingCuisineWithMostFrequentCuisineType(X_train['cuisines'],
                                                                                            X_test['cuisines'], filename_for_storage)
    oneHotEncodedCuisineDFTrain = oneHotEncodeCuisine(X_train["cuisines"], cuisines_set, cuisine_dict)
    oneHotEncodedCuisineDFTest = oneHotEncodeCuisine(X_test["cuisines"], cuisines_set, cuisine_dict)

    X_train = pd.concat([X_train, oneHotEncodedCuisineDFTrain], axis=1)
    X_test = pd.concat([X_test, oneHotEncodedCuisineDFTest], axis=1)

    X_train.pop('cuisines')
    X_test.pop('cuisines')


    df_rest_type = df[df['rest_type'].notnull()]

    rest_type_set, restaurant_dict = createAndSaveRestaurantDict(df_rest_type['rest_type'], path)

    X_train['rest_type'], X_test['rest_type'] = fillMissingRestaurantTypeWithMostFrequentRestaurantType(X_train['rest_type'], X_test['rest_type'], filename_for_storage)

    oneHotEncodedRestDFTrain = oneHotEncodeRestaurant(X_train["rest_type"], rest_type_set, restaurant_dict)
    oneHotEncodedRestDFTest = oneHotEncodeRestaurant(X_test["rest_type"], rest_type_set, restaurant_dict)

    X_train = pd.concat([X_train, oneHotEncodedRestDFTrain], axis=1)
    X_test = pd.concat([X_test, oneHotEncodedRestDFTest], axis=1)

    X_train.pop('rest_type')
    X_test.pop('rest_type')

    '''
    Cleaning Building Off The Shelf One Hot Encoding from sklearn For listed_in(city) and listed_in(type) attributes next
    '''

    X_train["listed_in(city)"], X_test["listed_in(city)"] = labelEncodeListedInCityFeature(X_train["listed_in(city)"], X_test["listed_in(city)"], path)

    X_train["listed_in(type)"], X_test["listed_in(type)"] = labelEncodeListedInTypeFeature(X_train["listed_in(type)"], X_test["listed_in(type)"], path)

    X_train, X_test = oneHotEncode(X_train, X_test, [5,6], path)

    '''
    Building, Evaluating and Saving Model
    '''

    acc, precision_score, recall_score, f1_score = buildAndSaveModel(X_train, X_test, y_train, y_test, path)

    return acc, precision_score, recall_score, f1_score, path