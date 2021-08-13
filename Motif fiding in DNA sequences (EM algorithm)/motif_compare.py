import json_save_params
import motif_297759_generate
import motif_297759_estimate
import json
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def ParseArguments():
    parser = argparse.ArgumentParser(description="Estimator comparison")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Zapisane wyestymowane parametry  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output

param_file, output_file = ParseArguments()

def distance_tv(p):
    return np.sum(np.abs(p))/2

def average_tv(ThetaB, ThetaB_est, Theta, Theta_est, w):
    dtv1 = distance_tv(ThetaB - ThetaB_est)
    dtv2 = np.sum(np.apply_along_axis(distance_tv, 0, Theta - Theta_est))
    return (dtv1 + dtv2)/(w+1)

def generate_dist_dense(k, w, limit=.8):
    X = np.random.uniform(0, 1-limit, size=(k, w))
    ind = np.random.choice(range(4), w)
    X[ind, np.arange(w)] = np.random.uniform(limit, 1, size=w)
    return X/sum(X)

K = [10, 50, 70, 100, 150, 200, 500, 1000]
W = [5, 10, 15, 20, 25, 30, 35]
ALPHA = np.array([.1, .3, .5, .7, .9])

def compare_em(W=W, K=K, ALPHA=ALPHA, init_avg=False, dist_dense=False, name=""):
    if dist_dense:
        Theta_dist = generate_dist_dense(4, max(W))
        ThetaB_dist = generate_dist_dense(4, 1).flatten()
    else:
        Theta_dist = motif_297759_estimate.generate_dist(4, max(W))
        ThetaB_dist = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    for a in range(len(ALPHA)):
        alpha = ALPHA[a]
        results = np.empty((len(W), len(K)))
        for i in range(len(W)):
            w = W[i]
            for j in range(len(K)):
                k = K[j]
                json_save_params.json_save_params(w=w, alpha=alpha, k=k, ThetaB=ThetaB_dist, Theta=Theta_dist[:, :w])
                motif_297759_generate.generate()
                motif_297759_estimate.estimate(init_avg=init_avg)

                with open(param_file, 'r') as paramfile:
                    params = json.load(paramfile)

                with open(output_file, 'r') as outputfile:
                    estimated_params = json.load(outputfile)

                w = params["w"]
                ThetaB = np.asarray(params["ThetaB"])
                ThetaB_est = np.asarray(estimated_params["ThetaB"])
                Theta = np.asarray(params["Theta"])
                Theta_est = np.asarray(estimated_params["Theta"])

                results[i, j] = average_tv(ThetaB, ThetaB_est, Theta, Theta_est, w)

        print("alpha= ", alpha)
        results = pd.DataFrame(results.round(5))
        results.index = W
        results.columns = K
        #results.to_csv("alpha=" + str(alpha) + "_" + name + "_results.csv")
        print(results)

compare_em(name="dense_dist_random_init", init_avg=False, dist_dense=False)


#os.system("python json_save_params.py")
#exec(open("json_save_params.py").read())