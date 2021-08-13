import json
import numpy as np
import argparse

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    parser.add_argument('--iter', default=100, required=False, help='Liczba powtorzen EM')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha, args.iter

input_file, output_file, estimate_alpha, iter = ParseArguments()

def get_thetaB(x_i, ThetaB):
    w = len(x_i)
    theta = np.zeros(w)
    for j in range(w):
        theta[j] = ThetaB[x_i[j]-1]
    return theta

def get_theta(x_i, Theta):
    w = len(x_i)
    theta = np.zeros(w)
    for j in range(w):
        theta[j] = Theta[x_i[j] - 1, j]
    return theta

def generate_dist(k, w):
    dist = np.random.random((k, w))
    return dist / sum(dist)

def estimate(input_file="generated_data.json", output_file="estimated_params.json", epsilon = 0.01, init_avg=False, n_iter=100):
    with open(input_file, 'r') as inputfile:
        data = json.load(inputfile)

    alpha=data['alpha']
    X = np.asarray(data['X'], dtype=int)
    k, w = X.shape

    THETAS = np.zeros((4, w))
    THETASB = np.zeros(4)

    if init_avg or k>100:
        n_iter = 1
    for n in range(n_iter):

        Theta = np.zeros((4, w))
        ThetaB = np.zeros(4)
        if init_avg: #emp initialization
            for j in range(4):
                Theta[j, :] = sum(X==j+1)/k
            for j in range(4):
                ThetaB[j] = np.sum(X==j+1)/(w*k)
        else: #random initialization
            ThetaB[:(4-1)] = np.random.rand(4-1)/4
            ThetaB[4-1] = 1-np.sum(ThetaB)
            Theta = generate_dist(4, w)

        #EM algorithm
        diff = 1
        while(diff>epsilon):
            #Expectation step
            Q_0 = np.zeros(k)
            Q_1 = np.zeros(k)
            for i in range(k):
                Q_0[i] = (1-alpha) * np.prod(get_thetaB(X[i, :], ThetaB))
                Q_1[i] = alpha * np.prod(get_theta(X[i, :], Theta))
            P_TH = Q_0 + Q_1
            Q_0 = Q_0/P_TH
            Q_1 = Q_1/P_TH

            #Maximization step
            #Background
            lambdaB = w * np.sum(Q_0)
            theta_b_t = np.zeros(4)
            for s in range(4):
                s_freq = np.zeros(k)
                for i in range(k):
                    s_freq[i] = np.sum(X[i, :] == s+1)
                theta_b_t[s] = np.sum(Q_0*s_freq)/lambdaB

            #Motif
            lambda_r = np.sum(Q_1)
            theta_t = np.zeros((4,w))
            for s in range(4):
                for r in range(w):
                    theta_t[s, r] = np.sum(Q_1 * (X[:, r] == s+1)) / lambda_r

            diff = (np.abs(ThetaB - theta_b_t)).sum() + (np.abs(Theta - theta_t)).sum()
            ThetaB = theta_b_t
            Theta = theta_t

        THETASB += ThetaB /n_iter
        THETAS += Theta /n_iter

    estimated_params = {
        "alpha" : alpha,            # "przepisujemy" to alpha, one nie bylo estymowane
        "Theta" : THETAS.tolist(),   # westymowane
        "ThetaB" : THETASB.tolist()  # westymowane
        }

    with open(output_file, 'w') as outfile:
        json.dump(estimated_params, outfile)

estimate(input_file, output_file, init_avg=False, n_iter=iter)
