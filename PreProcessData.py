'''
A class to handle the preprocessing of the data.  
Includes cleaning the data and feature engineering.
Stores the train and test feature and target matrices
'''
import numpy as np
import pandas as pd
import time
#local imports
import LocationClusters as LC
import CountFeatures as CF

class PreProcessData:
    #add arguments for feature columns, target columns
    def __init__(self, train_path,test_path,target,base_features):
        self.target =target
        self.base_features = base_features
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
                                            out_feat=list(self.train_df["dropoff_labels"]))
        self.X_train, self.y_train=self.make_data_arrays(self.train_df)

        #preprocess test data
        self.test_df = pd.read_csv(test_path)
        self.test_df = self.preprocess_test_data(self.test_df)

        self.X_test, self.y_test=self.make_data_arrays(self.test_df)

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
        train_df["dropoff_labels"]=self.clusters.end_MS.labels_
        train_df = self.add_time_features(train_df)
        return train_df  

    def preprocess_test_data(self,test_df):
        test_df = self.clean_data(test_df)
        test_df["begin_labels"] = self.clusters.predict_labels(
            x=test_df["begintrip_lng"],
            y=test_df["begintrip_lat"],
            start=True
            )
        test_df["dropoff_labels"] = self.clusters.predict_labels(
            x=test_df["dropoff_lng"],
            y=test_df["dropoff_lat"],
            start=False
            )
        test_df = self.add_time_features(test_df)
        #add label columns 
        return test_df

    def add_time_features(self,df):
        time_format = "%Y-%m-%d_%H:%M:%S"
        df["begintrip_at"]=df["begintrip_at"].apply(lambda x: time.strptime(x,"%Y-%m-%d_%H:%M:%S"))
        df["hour_of_day"] = df["begintrip_at"].apply(lambda x: x.tm_hour)
        df["day_of_week"] = df["begintrip_at"].apply(lambda x: x.tm_wday)
        return df

    def make_data_arrays(self,df):
        features = df[self.base_features].as_matrix()
        count_features = self.CountFeatures.make_features_from_new_data(in_data=df["begin_labels"])
        #combine the base features and count features
        X = np.hstack((features,count_features))
        y=df[self.target].as_matrix()
        return X, y

