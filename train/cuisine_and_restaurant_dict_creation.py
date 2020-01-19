from train.utils import saveDictData

def createCuisineSet(cuisines):

    cuisines_set = set()
    for entry in cuisines:
        val = entry.split(',')
        for value in val:
            cuisines_set.add(value.strip())

    return cuisines_set

def createAndSaveCuisineDict(cuisines, path):

    cuisines_set = createCuisineSet(cuisines)
    cuisine_dict = {}
    for i, cuisine in enumerate(cuisines_set):
        cuisine_dict[cuisine] = i

    saveDictData(path, dict_type="cuisine", dict=cuisine_dict)

    return cuisines_set, cuisine_dict

def createRestTypeSet(rest_type):

    rest_type_set = set()
    for entry in rest_type:
        val = entry.split(',')
        for value in val:
            rest_type_set.add(value.strip())

    return rest_type_set

def createAndSaveRestaurantDict(rest_type, path):

    rest_type_set = createRestTypeSet(rest_type)
    restaurant_dict = {}
    for i, restaurant in enumerate(rest_type_set):
        restaurant_dict[restaurant] = i

    saveDictData(path, dict_type = "restaurant", dict = restaurant_dict)

    return rest_type_set, restaurant_dict