"""
File to read in a test csv from command line and score prediction accuracy.
Reads in csvs in the original file form, so the test df is also preprocessed before making predictions
"""
import cPickle as pickle
import numpy as np
import pandas as pd
import time

def add_time_features(df):
    time_format = "%Y-%m-%d_%H:%M:%S"
    df["begintrip_at"]=df["begintrip_at"].apply(lambda x: time.strptime(x,"%Y-%m-%d_%H:%M:%S"))
    df["hour_of_day"] = df["begintrip_at"].apply(lambda x: x.tm_hour)
    df["day_of_week"] = df["begintrip_at"].apply(lambda x: x.tm_wday)
    return df

def clean_data(df):
    numeric_cols=['begintrip_lat', 'begintrip_lng','dropoff_lat','dropoff_lng']
    for col in numeric_cols:
        df[col] = df[col].convert_objects(convert_numeric=True)
    old_len = df.shape[0]
    df.dropna(axis=0,inplace=True)
    #reset index after dropping na
    df = df.reset_index()
    new_len = df.shape[0]
    return df

def preprocess_test_data(test_df,location_clusters):
    test_df = clean_data(test_df)
    test_df["begin_labels"] = location_clusters.predict_labels(
        x=test_df["begintrip_lng"],
        y=test_df["begintrip_lat"],
        start=True
        )
    test_df["dropoff_labels"] = location_clusters.predict_labels(
        x=test_df["dropoff_lng"],
        y=test_df["dropoff_lat"],
        start=False
        )
    #test_df = add_time_features(test_df)
    #add label columns 
    return test_df

def make_data_arrays(df,CF):
    count_features = CF.make_features_from_new_data(in_data=df["begin_labels"])
    #combine the base features and count features
    #X = np.hstack((base_features,count_features))
    target=["dropoff_lat","dropoff_lng"]
    y=df[target].as_matrix()
    return count_features, y


if __name__ == "__main__":
    
    MODEL_FILE="model_dict.p"
    model_dict = pickle.load(open(MODEL_FILE,"rb"))
    model = model_dict["model"]
    clusters=model_dict["location_clusters"]
    count_features=model_dict["count_features"]
    base_features=model_dict["base_features"]
    while True:
        test_path = raw_input(
            "\n"
            +"Please enter the path of a csv of test data\n"
            + "Type 'quit' to exit.\n"
            )
        if test_path == "quit":
            break
        else:
            test_df = pd.read_csv(test_path)
            test_df =preprocess_test_data(test_df =test_df,
                                        location_clusters=clusters)
        X_test,y_test=make_data_arrays(df=test_df,
                                        CF=count_features,
                                       )
        r_squared = model.score(X=X_test,y=y_test)
        print "For the RandomForestRegressor, the out of sample R^2: {}".format(r_squared)
