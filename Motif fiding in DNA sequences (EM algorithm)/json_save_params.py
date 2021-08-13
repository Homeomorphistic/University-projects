# written by Pawel Lorek, this file is needed to start generating data
import json

import numpy as np
 
import argparse 


tmp = np.array([[3/8,1/8,2/8,2/8],[1/10,2/10,3/10,4/10],[1/7,2/7,1/7,3/7]])
Theta = tmp.T

# background distribution
ThetaB = np.array([1/4,1/4,1/4,1/4])

def json_save_params(w=3, alpha=.5, k=10, ThetaB=ThetaB, Theta=Theta):
    params = {
        "w" : w,
        "alpha" : alpha,
        "k" : k,
        "Theta" : Theta.tolist(),
        "ThetaB" : ThetaB.tolist()
        }

    with open('params_set1.json', 'w') as outfile:
        json.dump(params, outfile)


json_save_params()

