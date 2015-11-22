'''
A class to manage the mean-shift clustering of start and end locations

'''
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

class LocationClusters():
    def __init__(self,
        x_start,
        y_start,
        x_end,
        y_end
        ):
        self.start_MS=fit_mean_shift_object(x=x_start,y=y_start)
        self.end_MS = fit_mean_shift_object(x=x_end,y=y_end)




    def predict_labels(self,x,y,start):
        '''
        Given x,y lists of x and y coordinates, returns the labels of 
        the clusters each pair (x,y) belongs to.  
        params:
            x,y -- lists of x and y coordinates
            start -- boolean.  If True uses the starting clusters
            otherwise uses the end clusters
        '''
        if start == True:
            MS = self.start_MS
        else:
            MS = self.end_MS
        points = make_coordinates(x,y)
        return MS.predict(points)


#utility functions
def make_coordinates(x,y):
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

def fit_mean_shift_object(x,y,quantile=.005):
    '''
    given x,y lists of x and y coordinates of points,
    we fit a Meanshift cluster object to these points
    '''
    points = make_coordinates(x=x,y=y)
    bandwidth=estimate_bandwidth(points, quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
    ms.fit(points)
    return ms