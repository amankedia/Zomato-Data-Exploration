import numpy as np
from train.utils import saveData

def replaceNewAndHyphenWithNaNInRatings(ratings):
    ratings = ratings.replace('-', np.nan)
    ratings = ratings.replace('NEW', np.nan)
    return ratings

def calculateRatingsMean(ratings, filename_for_storage):

    ratings = ratings.dropna()
    ratings = ratings.apply(lambda x : float(x.split('/')[0]))
    ratings_mean = ratings.mean()
    line = "Mean rating of training data is:" + str(ratings_mean)
    print(line)
    saveData(filename_for_storage, line)
    return ratings_mean

def replaceNaNWithMeanRatingAndConvertToFloat(training_rating, test_rating, filename_for_storage):

    mean_rating_of_training_data = calculateRatingsMean(training_rating, filename_for_storage)
    training_rating = training_rating.replace(np.nan, (str(mean_rating_of_training_data) + "/5"))
    test_rating = test_rating.replace(np.nan, (str(mean_rating_of_training_data) + "/5"))
    training_rating = training_rating.apply(lambda x: float(x.split('/')[0]))
    test_rating = test_rating.apply(lambda x: float(x.split('/')[0]))

    return training_rating, test_rating

def calculateApproxCostMean(approx_cost, filename_for_storage):

    approx_cost = approx_cost.dropna().str.replace(',', '').astype(float)
    approx_cost_mean = approx_cost.mean()
    line = "Mean approx cost of training data is:" + str(approx_cost_mean)
    print(line)
    saveData(filename_for_storage, line)
    return approx_cost_mean

def removeCommaAndReplaceNaNWithMeanForApproxCost(training_approx_cost, test_approx_cost, filename_for_storage):

    approx_cost_mean = calculateApproxCostMean(training_approx_cost, filename_for_storage)

    training_approx_cost = training_approx_cost.str.replace(',', '').astype(float)
    test_approx_cost = test_approx_cost.str.replace(',', '').astype(float)

    training_approx_cost = training_approx_cost.replace(np.nan, str(approx_cost_mean))
    test_approx_cost = test_approx_cost.replace(np.nan, str(approx_cost_mean))

    return training_approx_cost, test_approx_cost