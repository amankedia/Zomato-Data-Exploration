import numpy as np

rating_mean = 3.6988367561085296
approx_cost_mean = 555.1805221030095
most_frequent_restaurant_type = "Quick Bites"
most_frequent_cuisine = "North Indian"


def onlineOrderMapping(online_order):
    if online_order.lower() == "yes":
        return 1
    return 0

def bookTableMapping(book_table):
    if book_table.lower() == "yes":
        return 1
    return 0

def cleanRating(rate):
    if rate == "-" or rate == "NEW" or rate == "" or rate == np.nan:
        return rating_mean
    return float(rate.split('/')[0])

def processApproxCost(approx_cost):
    if approx_cost == np.nan or approx_cost == "":
        return approx_cost_mean
    approx_cost = float(approx_cost.replace(',', ''))
    return approx_cost

def checkRestaurantMissingValue(rest_type):
    if rest_type == "" or rest_type == np.nan:
        return most_frequent_restaurant_type
    return rest_type

def checkCuisineMissingValue(cuisines):
    if cuisines == "" or cuisines == np.nan:
        return most_frequent_cuisine
    return cuisines