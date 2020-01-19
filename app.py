from flask import Flask, request
from joblib import load

from prediction.attribute_validation_and_cleaning import onlineOrderMapping, bookTableMapping, cleanRating, \
    processApproxCost, checkRestaurantMissingValue, checkCuisineMissingValue
from prediction.basic_one_hot_encode import oneHotEncode
from prediction.custom_one_hot_encode import oneHotEncodeRestaurant, oneHotEncodeCuisine
from prediction.preprocess_numerical_attributes import applyScalar

from train.training import train_classifier

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():

    url = request.form["url"]
    address = request.form["address"]
    name = request.form["address"]
    online_order = request.form["online_order"]
    book_table = request.form["book_table"]
    rate = request.form["rate"]
    votes = request.form["votes"]
    phone = request.form["phone"]
    location = request.form["location"]
    rest_type = request.form["rest_type"]
    dish_liked = request.form["dish_liked"]
    cuisines = request.form["cuisines"]
    approx_cost = request.form["approx_cost(for two people)"]
    reviews_list = request.form["reviews_list"]
    menu_item = request.form["menu_item"]
    type_listed_in = request.form["listed_in(type)"]
    city_listed_in = request.form["listed_in(city)"]


    online_order = onlineOrderMapping(online_order)
    book_table = bookTableMapping(book_table)
    rate = cleanRating(rate)
    approx_cost = processApproxCost(approx_cost)

    rate, votes, approx_cost = applyScalar(rate, votes, approx_cost)

    rest_type = checkRestaurantMissingValue(rest_type)
    cuisines = checkCuisineMissingValue(cuisines)

    rest_type = oneHotEncodeRestaurant(rest_type)
    cuisines = oneHotEncodeCuisine(cuisines)

    arr = convertAttributesToFeatureArray(online_order, book_table, rate, votes, approx_cost, type_listed_in, city_listed_in, cuisines, rest_type)

    arr = oneHotEncode(arr)

    prediction = model.predict(arr)

    if prediction[0] == 0:
        return "Vegetarian"
    else:
        return "Non-vegetarian"

@app.route('/train', methods=['POST'])
def train():
    data = request.files['data']
    acc, precision_score, recall_score, f1_score, path = train_classifier(data)
    final_output =  "The Accuracy of the trained model is: " + str(acc) + "\nPrecision: " + str(precision_score) + "\nRecall: " + str(recall_score) + "\nF1 Score: " + str(f1_score) + "\nThe model along with supporting files has been saved at: " + str(path)
    return final_output

def convertAttributesToFeatureArray(online_order, book_table, rate, votes, approx_cost, type_listed_in, city_listed_in, cuisines, rest_type):
    arr = [online_order, book_table, rate, votes, approx_cost, type_listed_in, city_listed_in]
    arr.extend(cuisines)
    arr.extend(rest_type)
    # print(arr)
    # print(len(arr))
    return arr


def loadModel():
    classifier = load("prediction/svm_classifier.joblib")
    return classifier



if __name__ == '__main__':
    model = loadModel()

    # convertAttributesToFeatureArray(1, 0, -0.249901, -0.165918, 0.330432, "Dine-out", "Kammanahalli", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    app.run(port=5000)