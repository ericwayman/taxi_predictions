'''

'''
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import RandomForestRegressor 
import LocationClusters as LC
import CountFeatures as CF

class PreProcessData:
    #add arguments for feature columns, target columns
    def __init__(self, train_path,test_path,target):
        self.target =target
        #preprocess train data
        train_df =pd.read_csv(train_path)
        self.clusters = LC.LocationClusters(
            x_start=train_df["begintrip_lng"],
            y_start=train_df["begintrip_lat"],
            x_end=train_df["dropoff_lng"],
            y_end=train_df["dropoff_lat"]
            )
        self.train_df = self.preprocess_train_data(train_df)
        self.CountFeatures = CF.CountFeatures(in_feat=list(self.train_df["begin_labels"]),
                                            out_feat=list(self.train_df["drop_off_labels"]))
        self.X_train, self.y_train=self.train_data(self.train_df)

        #preprocess test data
        self.test_df = pd.read_csv(test_path)
        self.test_df = self.preprocess_test_data(self.test_df)

        self.X_test, self.y_test=self.test_data(self.test_df)

    #necessary because the dropoff fields get read as strs in the test data for some reason
    #need to reset index after dropping
    def clean_data(self,df):
        numeric_cols=['begintrip_lat', 'begintrip_lng','dropoff_lat','dropoff_lng']
        for col in numeric_cols:
            df[col] = df[col].convert_objects(convert_numeric=True)
        old_len = df.shape[0]
        df.dropna(axis=0,inplace=True)
        #reset index after dropping na
        df = df.reset_index()
        new_len = df.shape[0]
        print "Dropped {} rows with NaN".format(old_len-new_len)
        return df

    def preprocess_train_data(self,train_df):
        train_df = self.clean_data(train_df)
        train_df["begin_labels"]=self.clusters.start_MS.labels_
        train_df["drop_off_labels"]=self.clusters.end_MS.labels_
        return train_df  

    def preprocess_test_data(self,test_df):
        test_df = self.clean_data(test_df)
        test_df["begin_labels"] = self.clusters.predict_labels(
            x=test_df["begintrip_lng"],
            y=test_df["begintrip_lat"],
            start=True
            )
        #add label columns 
        return test_df


    def train_data(self,train_df):
        X_train = self.CountFeatures.make_features_from_new_data(in_data=train_df["begin_labels"])
        y_train=train_df[self.target].as_matrix()
        return X_train, y_train

    def test_data(self,test_df):
        X_test = self.CountFeatures.make_features_from_new_data(in_data=test_df["begin_labels"])
        y_test=test_df[self.target].as_matrix()
        return X_test, y_test


