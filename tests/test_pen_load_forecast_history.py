import unittest
import numpy as np
from supermimic_research.mechanics.pen_load_forecast import examples


class TestLoadForecastHistory(unittest.TestCase):
    def test_previous_window_rotates_into_current_body_without_future_labels(self):
        x=np.zeros((4,102)); y=np.zeros((4,90))
        x[:,43:46]=[1,0,0]; x[:,46:49]=[0,1,0]
        x[1,43:46]=[0,1,0]; x[1,46:49]=[-1,0,0]
        y[0,78:90]=[1,0,0,0,1,0,1,2,3,4,5,6];y[1,78:90]=17
        X,Y=examples(x,y,[True,True,False,True],True)
        self.assertEqual(X.shape,(1,87))
        np.testing.assert_allclose(X[0,-12:],[0,-1,0,1,0,0,1,2,3,4,5,6])
        np.testing.assert_array_equal(Y[0,:12],np.full(12,17))
        y[1,78:90]=100
        np.testing.assert_array_equal(examples(x,y,[True,True,False,True],True)[0],X)

if __name__=='__main__':unittest.main()
