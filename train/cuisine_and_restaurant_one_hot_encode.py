import pandas as pd

def oneHotEncodeCuisine(data, cuisine_set, cuisine_dict):
    indexes = data.index
    cuisine_encoded_list = []
    for i, restaurant_cuisine in enumerate(data):
        val = restaurant_cuisine.split(',')
        l = [0]*len(cuisine_set)
        for value in val:
            l[cuisine_dict[value.strip()]] = 1
        cuisine_encoded_list.append(l)
    df = pd.DataFrame(cuisine_encoded_list, columns = cuisine_set, index = indexes)
    return df

def oneHotEncodeRestaurant(data, restaurant_set, restaurant_dict):
    indexes = data.index
    restaurant_serving_encoded_list = []
    for i, restaurant_serving in enumerate(data):
        val = restaurant_serving.split(',')
        l = [0]*len(restaurant_set)
        for value in val:
            l[restaurant_dict[value.strip()]] = 1
        restaurant_serving_encoded_list.append(l)
    df = pd.DataFrame(restaurant_serving_encoded_list, columns = restaurant_set, index = indexes)
    return df