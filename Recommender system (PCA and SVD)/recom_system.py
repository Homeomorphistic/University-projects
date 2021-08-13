import numpy as np
import argparse
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import sklearn.model_selection as select
import time
import pandas as pd

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--train', required=True, help='train_file.csv')
    parser.add_argument('--test', required=True, help='test_file.csv')
    parser.add_argument('--alg', default="SVD2", required=False, help='NMF, SVD1, SVD2, SGD')
    parser.add_argument('--result', required=True, help='result_file.csv')
    args = parser.parse_args()
    return args.train, args.test, args.alg, args.result

train, test, alg, result = ParseArguments()

#Quality of the system
def RMSE(Z, T, T_ind):
    return (np.sum((Z[T_ind[:, 0], T_ind[:, 1]] - T[T_ind[:, 0], T_ind[:, 1]])**2) / len(T_ind))**(1/2)

#Searching for the best dimension for procedure
def test_proc(Z, T, T_ind, proc, proc_name):
    results = np.empty((0, 2))
    R = np.append(np.arange(1, 21, step=1), np.arange(40, 201, step=20))
    #R = np.arange(5, 31, step=5)
    print(R)
    for r in R:
        start = time.time()
        Z_approximated = proc(Z, r)
        end = time.time()
        print("r=", r, "time=", end - start)
        print("RMSE=", RMSE(Z_approximated, T, T_ind))
        results = np.append(results, np.array([[end - start, RMSE(Z_approximated, T, T_ind)]]), axis=0)

    results = pd.DataFrame(results.round(5))
    results.index = R
    results.columns = ["time", "RMSE"]
    print(results)
    results.to_csv(proc_name+"_results.csv")
    return results
'''
#Loading data
ratings = pd.read_csv('ratings.csv')

#Dividing data into train and test sets
train_ratings, test_ratings = select.train_test_split(ratings, test_size=.1, stratify=ratings['userId'])

#Saving the data
train_ratings.to_csv('train_ratings.csv', index=False)
test_ratings.to_csv('test_ratings.csv', index=False)
'''
#Sparse matrices
train_ratings = np.genfromtxt(train, delimiter=",", dtype=int, skip_header=1, usecols=(0, 1, 2))
test_ratings = np.genfromtxt(test, delimiter=",", dtype=int, skip_header=1, usecols=(0, 1, 2))

n = len(np.unique(train_ratings[:, 0]))
d = len(np.unique(train_ratings[:, 1]))

#Table with new indices
movies_id = np.unique(train_ratings[:, 1]).reshape(d, 1)
movies_id = np.append(movies_id, np.arange(d).reshape(d, 1), axis=1)
movies_id = pd.DataFrame(movies_id)
movies_id.columns = ["movieId", "newId"]

#Adding new column to train
train_ratings = pd.DataFrame(train_ratings)
train_ratings.columns = ["userId", "movieId", "rating"]
train_ratings = train_ratings.join(movies_id.set_index("movieId"), on="movieId", how="inner")

#Avg ratings for every user
ratings_avg = train_ratings[['userId', 'rating']].groupby(by=["userId"]).mean()
ratings_me = train_ratings[['userId', 'rating']].groupby(by=["userId"]).median()

ratings_avg = np.array(ratings_avg).reshape(n, 1)
ratings_me = np.array(ratings_me).reshape(n, 1)

train_ratings = np.array(train_ratings)

#Adding new column to test
test_ratings = pd.DataFrame(test_ratings)
test_ratings.columns = ["userId", "movieId", "rating"]
test_ratings = test_ratings.join(movies_id.set_index("movieId"), on="movieId", how="inner")
test_ratings = np.array(test_ratings)

Z = np.empty((n, d)) #+ ratings_avg
T = np.empty((n, d))
Z[train_ratings[:, 0]-1, train_ratings[:, 3]] = train_ratings[:, 2]
T[test_ratings[:, 0]-1, test_ratings[:, 3]] = test_ratings[:, 2]
T_ind = np.transpose(np.array([test_ratings[:, 0]-1, test_ratings[:, 3]]))

#Non-negative matrix factorization
def NMF_proc(Z, r, avg=0):
    model = NMF(n_components=r, init="random", random_state=0)
    W = model.fit_transform(Z + avg*(Z==0))
    H = model.components_
    return np.dot(W, H)
#test_proc(Z, T, T_ind, NMF_proc, "NMF_0")
#test_proc(Z, T, T_ind, lambda Z, r: NMF_proc(Z,r,np.mean(ratings_avg)), "NMF_avg")
#test_proc(Z, T, T_ind, lambda Z, r: NMF_proc(Z,r,np.median(ratings_me)), "NMF_me")
#test_proc(Z, T, T_ind, lambda Z, r: NMF_proc(Z,r,ratings_avg), "NMF_user_avg")
#test_proc(Z, T, T_ind, lambda Z, r: NMF_proc(Z,r,ratings_me), "NMF_user_me")

#Singular Value Decomposition 1
def SVD1_proc(Z, r, avg=0):
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(Z + avg*(Z==0))
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z + avg*(Z==0)) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    return np.dot(W, H)
#test_proc(Z, T, T_ind, SVD1_proc, "SVD1_0")
#test_proc(Z, T, T_ind, lambda Z, r: SVD1_proc(Z,r,np.mean(ratings_avg)), "SVD1_avg")
#test_proc(Z, T, T_ind, lambda Z, r: SVD1_proc(Z,r,np.median(ratings_me)), "SVD1_me")
#test_proc(Z, T, T_ind, lambda Z, r: SVD1_proc(Z,r,ratings_avg), "SVD1_user_avg")
#test_proc(Z, T, T_ind, lambda Z, r: SVD1_proc(Z,r,ratings_me), "SVD1_user_me")

#Singular Value Decomposition 2
def SVD2_proc(Z, r, avg = 0, max_it=100, eps=0.001):
    Z_prev = Z.copy() + avg*(Z==0)
    diff = 1
    count = 0
    while diff >= eps and count != max_it:
        Z_next = SVD1_proc(Z_prev, r)
        Z_next *= (Z==0)
        Z_next += Z
        #diff = round((np.abs(Z_next - Z_prev)).mean(), 5)
        diff = RMSE(Z_prev, T, T_ind) - RMSE(Z_next, T, T_ind)
        #print("count= ", count, "diff= ", diff)
        Z_prev = Z_next.copy()
        count += 1
    return Z_prev #np.abs(Z_prev)
#test_proc(Z, T, T_ind, lambda Z, r: SVD2_proc(Z,r,eps=0.01), "SVD2_0")
#test_proc(Z, T, T_ind, lambda Z, r: SVD2_proc(Z,r,np.mean(ratings_avg)), "SVD2_avg")
#test_proc(Z, T, T_ind, lambda Z, r: SVD2_proc(Z,r,np.median(ratings_me)), "SVD2_me")
#test_proc(Z, T, T_ind, lambda Z, r: SVD2_proc(Z,r,ratings_avg, eps=0.00001), "SVD2_user_avg")
#test_proc(Z, T, T_ind, lambda Z, r: SVD2_proc(Z,r,ratings_me, eps=0.0001), "SVD2_user_me")

#Stochastic gradient descent

rmse = -1
print(alg+" = ...")
if(alg == 'NMF'):
    rmse = RMSE(NMF_proc(Z, 18, ratings_avg), T, T_ind)
elif(alg == 'SVD1'):
    rmse = RMSE(SVD1_proc(Z, 11, ratings_avg), T, T_ind)
elif(alg == 'SVD2'):
    rmse = RMSE(SVD2_proc(Z, 2, ratings_avg, eps=0.00001), T, T_ind)
else:
    print('Not implemented yet')
print(alg+" = ", rmse)
np.savetxt(result, np.array(rmse).reshape(1,1))
