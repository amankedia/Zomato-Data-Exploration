from joblib import load

def applyScalar(rate, votes, approx_cost):

    # print(rate, votes, approx_cost)

    standardScalar = load("prediction/std_scaler.bin")
    # print(standardScalar.mean_)
    # print(standardScalar.scale_)
    rate = (rate - standardScalar.mean_[0]) / standardScalar.scale_[0]
    votes = (float(votes) - standardScalar.mean_[1]) / standardScalar.scale_[1]
    approx_cost = (approx_cost - standardScalar.mean_[2]) / standardScalar.scale_[2]

    # print(rate, votes, approx_cost)

    return rate, votes, approx_cost