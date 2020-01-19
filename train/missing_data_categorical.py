from train.utils import saveData

def mostFrequentRestaurant(rest_type, filename_for_storage):

    most_frequent_restaurant_type = rest_type.mode().iloc[0]
    line = "Most frequent restaurant type is:" + most_frequent_restaurant_type
    print(line)
    saveData(filename_for_storage, line)
    return most_frequent_restaurant_type

def fillMissingRestaurantTypeWithMostFrequentRestaurantType(train_rest_type, test_rest_type, filename_for_storage):

    most_frequent_restaurant_type = mostFrequentRestaurant(train_rest_type, filename_for_storage)
    train_rest_type = train_rest_type.fillna(most_frequent_restaurant_type)
    test_rest_type = test_rest_type.fillna(most_frequent_restaurant_type)

    return train_rest_type, test_rest_type


def mostFrequentCuisine(cuisine, filename_for_storage):

    most_frequent_cuisine_type = cuisine.mode().iloc[0]
    line = "Most frequent cuisine type is:" + most_frequent_cuisine_type
    print(line)
    saveData(filename_for_storage, line)
    return most_frequent_cuisine_type

def fillMissingCuisineWithMostFrequentCuisineType(train_cuisine, test_cuisine, filename_for_storage):

    most_frequent_cuisine_type = mostFrequentCuisine(train_cuisine, filename_for_storage)
    train_cuisine = train_cuisine.fillna(most_frequent_cuisine_type)
    test_cuisine = test_cuisine.fillna(most_frequent_cuisine_type)

    return train_cuisine, test_cuisine