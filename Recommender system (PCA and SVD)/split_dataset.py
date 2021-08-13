import pandas as pd
import sklearn.model_selection as sml
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time  # mierzenie czasu
import math  # sqrt
import statistics  # mean
import random  # random.choices

nmf = np.zeros((10, 30))
SVD = np.zeros((10, 30))
SVD2 = np.zeros((10, 30))
for k in range(10):
    t1 = time.time()
    df = pd.read_csv('ratings.csv')
    train, test = sml.train_test_split(df, test_size=0.1, stratify=df['userId'])

    users = sorted(set(df['userId']))
    no_users = len(users)
    movies = sorted(set(df['movieId']))
    no_movies = len(movies)
    '''
    # średnia z wszystkich ocen (Domyślnie: 1.04, NMF: 0.98, SVD: 0.98, SVD2: 0.86)
    Z = np.zeros((no_users, no_movies)) + statistics.mean(train["rating"])  
    '''

    # średnia z ocen danego użytkownika (Domyślnie: 0.94, NMF: 0.91, SVD: 0.91, SVD2: 0.86)
    Z = np.zeros((no_users, no_movies))
    for index in range(no_users):
        Z[index, :] += statistics.mean(train.iloc[np.where(train["userId"] == users[index])[0], 2])

    '''
    # mediana z ocen danego użytkownika (Domyślnie: 0.98, NMF: 0.94, SVD: 0.94, SVD2: 0.87)
    Z = np.zeros((no_users, no_movies))
    for index in range(no_users):
        Z[index, :] += statistics.median(train.iloc[np.where(train["userId"] == users[index])[0], 2])
    '''

    '''
    # losowa próba z ocen danego użytkownika (Domyślnie: 1.32, NMF: 0.92, SVD: 0.92, SVD2: 0.86)
    Z = np.zeros((no_users, no_movies))
    for index in range(no_users):
        Z[index, :] = train.iloc[np.where(train["userId"] == users[index])[0], 2].sample(n=no_movies, replace=True)
    '''

    '''
    # średnia z ocen danego filmu (Domyślnie: 0.97, NMF: 0.94, SVD: 0.94, SVD2: 0.88)
    Z = np.zeros((no_users, no_movies))
    for index in range(no_movies):  
        tmp = np.where(train["movieId"] == movies[index])[0]
        if(len(tmp) > 0):
            Z[:, index] += statistics.mean(train.iloc[tmp, 2])
        else:
            Z[:, index] += statistics.mean(train["rating"])
    '''

    moviesId_train = np.zeros((len(train), 1), dtype=int)
    moviesId_test = np.zeros((len(test), 1), dtype=int)

    for index in range(no_movies):
        moviesId_train[np.where(movies[index] == train["movieId"])[0]] = index + 1
        moviesId_test[np.where(movies[index] == test["movieId"])[0]] = index + 1
    train.insert(2, "new_movieId", moviesId_train, True)
    test.insert(2, "new_movieId", moviesId_test, True)
    train.to_csv('train_ratings.csv', index=False)
    test.to_csv('test_ratings.csv', index=False)
    # print("Koniec pierwszej pętli")

    Z[train["userId"] - 1, train["new_movieId"] - 1] = train["rating"]

    Z_df = pd.DataFrame(data=Z, columns=list(movies), index=list(users))
    Z_df.to_csv('Z.csv', sep=";", decimal=",")
    # print("Macierz Z została zapisana")


    def rmse(z_app, test_set):
        return math.sqrt(1 / len(test_set) *
                         sum(pow(z_app[test_set["userId"] - 1, test_set["new_movieId"] - 1] - test["rating"], 2)))


    # print("Wartość RMSE bez użycia algorytmów:", rmse(Z, test))
    # print("")
    # print("")

    for r in range(30):
        start = time.time()
        model = NMF(n_components=int(r + 1), init='random', random_state=0)
        W = model.fit_transform(Z)
        H = model.components_
        Z_approximated = np.dot(W, H)
        end = time.time()
        nmf[k, r] = rmse(z_app=Z_approximated, test_set=test)
        # print("RMSE:", rmse(z_app=Z_approximated, test_set=test), "Czas:", end - start)
    # print("Koniec NMF")
    # print("")
    # print("")

    for r in range(30):
        start = time.time()
        svd = TruncatedSVD(n_components=r, random_state=42)
        svd.fit(Z)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z) / svd.singular_values_
        H = np.dot(Sigma2, VT)
        Z_approximated = np.dot(W, H)
        end = time.time()
        SVD[k, r] = rmse(z_app=Z_approximated, test_set=test)
        # print("RMSE:", rmse(z_app=Z_approximated, test_set=test), "Czas:", end - start)
    # print("Koniec SVD")
    # print("")
    # print("")

    for r in range(30):
        epsilon1 = 10
        epsilon2 = 9
        tmp = Z.copy()
        start = time.time()
        while epsilon1 - epsilon2 > pow(0.1, 6):
            epsilon1 = epsilon2
            svd = TruncatedSVD(n_components=r, random_state=42)
            svd.fit(tmp)
            Sigma2 = np.diag(svd.singular_values_)
            VT = svd.components_
            W = svd.transform(tmp) / svd.singular_values_
            H = np.dot(Sigma2, VT)
            Z_approximated = np.dot(W, H)
            epsilon2 = rmse(z_app=Z_approximated, test_set=test)
            tmp = Z_approximated
            tmp[train["userId"] - 1, train["new_movieId"] - 1] = Z[train["userId"] - 1, train["new_movieId"] - 1]
        end = time.time()
        SVD2[k, r] = rmse(z_app=Z_approximated, test_set=test)
        # print("RMSE:", rmse(z_app=Z_approximated, test_set=test), "Czas:", end - start)
    t2 = time.time()
    print(k, "Czas:", t2 - t1)
    # print("Koniec SVD2")

nmf_df = pd.DataFrame(data=nmf)
nmf_df.to_csv('nmf.csv', sep=";", decimal=",")
SVD_df = pd.DataFrame(data=SVD)
SVD_df.to_csv('svd.csv', sep=";", decimal=",")
SVD2_df = pd.DataFrame(data=SVD2)
SVD2_df.to_csv('svd2.csv', sep=";", decimal=",")
