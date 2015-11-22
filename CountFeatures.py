'''
A class to compute historical counts of categorical
features as new features
given two features input (I) and output (O).  We generate 
a table with rows indexed by O and columns indexed by I
row j,k is the historical counts of class k for O when I is class j
'''
from sklearn.metrics import confusion_matrix
from numpy import zeros

class CountFeatures:

    def __init__(self,in_feat,out_feat,num_in=None,num_out=None):
        '''
        params:
            --in_feat list like object of input feature values
            --out_feat list like object of output feature values
            --num_in = number of input feature classes
            --num_out = number of output feature classes

        if num_in or num_out is 0, then the answer is parsed from in_feat 
        or out_feat resp.

        '''
        self._input= in_feat
        self._output = out_feat
        if num_in == None:
            self._num_in = len(set(in_feat))
        else:
            self._num_in = num_in

        if num_out == None:
            self._num_out = len(set(out_feat))
        else:
            self._num_out = num_in
        self._labels = range(max(self._num_in,self._num_out))
        self._count_dict = self._make_count_dict()


    def _make_count_matrix(self):
        counts =  confusion_matrix(
                y_true=self._input,
                y_pred=self._output,
                labels=self._labels
                )
        return counts[:self._num_in,:self._num_out]

    def _make_count_dict(self):
        count_dict = {}
        count_matrix = self._make_count_matrix()
        input_labels = range(self._num_in)
        for i in input_labels:
            count_dict[i] = list(count_matrix[i,:])
        return count_dict


    def make_features_from_new_data(self,in_data):
        '''
        Return a matrix of count features from historical in_data
        for a list of input feature data 
        '''
        nrows = len(in_data)
        ncols = self._num_out
        count_features = zeros((nrows,ncols))
        #fill rows of count_features
        for i in range(nrows):
            count_features[i,:]=self._count_dict[in_data[i]]
        return count_features

