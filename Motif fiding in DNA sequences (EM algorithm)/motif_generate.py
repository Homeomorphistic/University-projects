import json
import numpy as np

import argparse 


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()

def generate(param_file="params_set1.json", output_file="generated_data.json"):
    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    k = params['k']
    alpha = params['alpha']
    Theta = np.asarray(params['Theta'])
    ThetaB = np.asarray(params['ThetaB'])

    X = np.empty(shape=(k, w))
    Z = np.random.choice(2, k, p=[1-alpha, alpha])

    for i in range(k):
        if(Z[i]):
            for j in range(w):
                X[i, j] = np.random.choice(4, size=1, p=Theta[:, j]) + 1
        else:
            X[i, :] = np.random.choice(4, size=w, p=ThetaB) + 1

    gen_data = {
        "alpha" : alpha,
        "X" : X.tolist()
        }

    with open(output_file, 'w') as outfile:
        json.dump(gen_data, outfile)

generate(param_file, output_file)