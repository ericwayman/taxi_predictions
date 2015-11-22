import pandas as pd
import PreProcessData as PPD
from sklearn.ensemble import RandomForestRegressor 

if __name__ == "__main__":
    TRAIN_PATH="data/hw1_train.csv"
    TEST_PATH="data/hw1_test.csv"
    data = PPD.PreProcessData(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        target=["dropoff_lat","dropoff_lng"])

    RF = RandomForestRegressor(n_estimators=100,criterion="mse")
    RF.fit(X=data.X_train,y=data.y_train)
    r_squared = RF.score(X=data.X_test,y=data.y_test)
    print "For the RandomForestRegressor, the out of sample R^2: {}".format(r_squared)