import numpy as np
import cPickle as pickle
import pandas as pd
import PreProcessData as PPD
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from numpy.linalg import norm
#helper functions
def score_classifier(prob_matrix,true_labels, LC):
    '''
    Compute R^2 for a classifier.  The drop off location in R^2 is computed by taking 
    the expectation with respect to the predict class probabilities.  
    params:
        prob_matrix-- a matrix of prediction probabilities
            prob_matrix[i,j] is the predicted probability sample i belongs to class j
        actual_labels-- the actual labels for the test data
        LC -- a location cluster object
    '''
    pred_locs = LC.expected_locations(prob_matrix=prob_matrix,end=True)
    true_locs = LC.end_MS.cluster_centers_[true_labels]
    return compute_rsquared(y_true=true_locs,y_pred=pred_locs)


def compute_rsquared(y_true,y_pred):
    '''
    compute the rsquared value for points in R^d 
    params:
        y_true -- a [num_samples,d] numpy array of actual values
        y_pred -- a [num_samples,d] numpy array of predicted values
    '''
    SSres = (norm(y_true - y_pred,axis=1) ** 2).sum()
    SStot = (norm(y_true - y_true.mean(axis=0),axis=1) ** 2).sum()
    # print "SSres: {0}.    SStot: {1}".format(SSres,SStot)
    # print "SSres/SStot: {}".format(SSres/SStot)
    return 1 - SSres/SStot

def save_model(model,model_file="models.p"):
    pickle.dump(model, open(model_file, "wb" ) )


if __name__ == "__main__":
    TRAIN_PATH="data/hw1_train.csv"
    TEST_PATH="data/hw1_test.csv"

    data = PPD.PreProcessData(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        target=["dropoff_lat","dropoff_lng"],
        base_features =[])#["begintrip_lat","begintrip_lng","day_of_week","hour_of_day"])

    RF = RandomForestRegressor(n_estimators=100,criterion="mse")
    RF.fit(X=data.X_train,y=data.y_train)
    r_squared = RF.score(X=data.X_test,y=data.y_test)
    print "For the RandomForestRegressor, the out of sample R^2: {}".format(r_squared)
    model_dict = {}
    model_dict["model"] = RF
    model_dict["location_clusters"]=data.clusters
    model_dict["count_features"]=data.CountFeatures
    model_dict["base_features"]=data.base_features
    print "saving model"
    save_model(model=model_dict,model_file="model_dict.p")

    '''
    Code to train RandomForestClassifier.  Work in Progress
    '''
    # data = PPD.PreProcessData(
    #     train_path=TRAIN_PATH,
    #     test_path=TEST_PATH,
    #     target=["dropoff_labels"],
    #     base_features =[])#["begintrip_lat","begintrip_lng","day_of_week","hour_of_day"])

    # RC = RandomForestClassifier(n_estimators=100)
    # print "training random forest"
    # RC.fit(X=data.X_train,y=data.y_train)
    # print "random forest trained"
    # prob_matrix = RC.predict_proba(data.X_test)
    # r_squared = score_classifier(prob_matrix=prob_matrix,true_labels=data.y_test, LC= data.clusters)
    # print r_squared
    
    # print "Saving model."
    # #save model
    # save_model(model=RC,model_file="model.p")

