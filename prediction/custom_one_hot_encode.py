from prediction.cuisine_info import cuisines_dict
from prediction.restaurant_info import restaurant_dict

def oneHotEncodeCuisine(cuisines):
    cuisine_list = cuisines.split(',')
    cuisines_one_hot_encoding = [0]*len(cuisines_dict)
    for cui in cuisine_list:
        cuisines_one_hot_encoding[cuisines_dict[cui.strip()]] = 1
    return cuisines_one_hot_encoding


def oneHotEncodeRestaurant(rest_type):
    restaurant_serving_list = rest_type.split(',')
    restaurant_serving_one_hot_encoding = [0]*len(restaurant_dict)
    for serving in restaurant_serving_list:
        restaurant_serving_one_hot_encoding[restaurant_dict[serving.strip()]] = 1
    return restaurant_serving_one_hot_encoding

# if __name__ == '__main__':
#     print(oneHotEncodeCuisine("North Indian, Tex-Mex"))
#     print(oneHotEncodeRestaurant("Bhojanalya,Food Court, Bakery"))