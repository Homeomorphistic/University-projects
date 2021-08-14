import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import kstest
from scipy.stats import norm
from scipy.special import *

#GENERATORS
def LCG(M, a, c, seed):
    while True:
        seed = (a * seed + c) % M
        yield seed

def GLCG(M=2**10, a_vec=np.array([3,7,68]), seed_vec=np.array([1,2,3])):
    while True:
        rnd = ( (a_vec * seed_vec).sum() ) % M
        seed_vec = np.append(seed_vec[1:], rnd)
        yield rnd

def excel(seed=0):
    while True:
        seed = (0.9821 * seed + 0.211327) % 1
        yield seed

def RC4(m=32, K = [0]):
    i, j = 0, 0
    S = np.arange(m)
    while True:
        i = (i + 1) % m
        j = int(j + S[i] + K[i % len(K)]) % m
        tmp = S[i]
        S[i] = S[j]
        S[j] = tmp
        yield S[(S[i] + S[j]) % m]

def RC4_keys(KEYS, m=32, n=2**18):
    i, j = 0, 0
    while True:
        if i % n == 0:
            #print(KEYS[j % KEYS.shape[0], :])
            gen = RC4(m, KEYS[j % KEYS.shape[0], :])
            j += 1
        i += 1
        yield next(gen)

def keys(n=100, m=32):
    K = np.zeros((n-1,  m))
    for i in range(m):
        j = 0
        while j < m-1 and j + i*(m-1) < n-1:
            K[j + i*(m-1), m-i-1] = j+1
            j += 1
    return np.append(np.zeros((1, m)), K, axis=0)

def PCG64(seed):
    np.random.seed(seed)
    while True:
        yield np.round(np.random.rand(),8)

def irrational_gen(const, seed=0):
    i = seed
    while True:
        i += 1
        yield const[i]
####################################################################################################

#CONVERTING FUNCTIONS
def read_const_to_str(file):
    const = np.genfromtxt(file, dtype="str")
    str = ''
    for i in range(len(const)):
        str += const[i]
    return str

def str_to_array(str):
    arr = np.full((len(str)), 0)
    for i in range(len(str)):
        arr[i] = int(str[i])
    return arr

def int_to_bin(rnd, max):
    rnd_bin = np.array([np.binary_repr(int(x), max) for x in rnd])#rnd_bin = np.array([bin(int(x))[2:] for x in rnd])
    str = ''
    for i in range(len(rnd_bin)):
        str += rnd_bin[i]
    return str_to_array(str)

def float_to_bin(rnd, max=32):
    rnd_bin = np.array([ np.binary_repr(int(np.floor(x*(2**max))), max) for x in rnd ])
    str = ''
    for i in range(len(rnd_bin)):
        str += rnd_bin[i]
    return str_to_array(str)

def to_bin(rnd, max, type):
    if (type == 'float'):
        rnd = float_to_bin(rnd, max)
    elif (type == 'int'):
        rnd = int_to_bin(rnd, max)
    return rnd

####################################################################################################

#PREPARING DATA
def generate_rnd(gen, n=1000):
    rnd = np.zeros(n)
    for i in range(n):
        rnd[i] = next(gen)
    return rnd

def compute_nrs_in_bins(partition, points):
    bins = np.zeros(len(partition) - 1)
    for i in range(len(bins)):
        ffrom = partition[i]
        tto = partition[i + 1]
        bins[i] = ((points > ffrom) & (points <= tto)).sum()
    return bins

#TESTS
def chi_test(rnd, min=0, max=1):
    if max==1:
        partition = np.arange(11)/10
        partition[0] = -1E-7
    else:
        partition = (min-1) + np.arange(max - min + 1) #+2?
    counts = compute_nrs_in_bins(partition, rnd)
    chi, p = chisquare(counts)
    return p

def ks_test(rnd, max=1):
    rnd = rnd/max
    stat, p = kstest(rnd, cdf="uniform")
    return p

def freq_monobit_test(B, max, type):
    B = to_bin(B, max, type)
    x = 2*B - 1
    n = len(B)
    stat = n**(-1/2) * x.sum()
    return 2*(1-norm.cdf(np.abs(stat)))

def freq_block_test(B, M, max, type):
    B = to_bin(B, max, type)
    N = int(np.floor(len(B)/M))
    pi = np.zeros(N)
    for i in range(N):
        for j in range(M):
            pi[i] += B[i*M + j]
    pi = pi/M
    stat = 4*M* ((pi-1/2)**2).sum()
    return gammaincc(N/2, stat/2)

def test_gen(gen, test, n=2**20, r=1000, gen_name='', test_name=''):
    #rnd = generate_rnd(gen, n*r)
    p_values = np.empty(r+1)
    for i in range(r):
        #p_values[i+1] = test(rnd[(i*n):((i+1)*n)])
        p_values[i + 1] = test(generate_rnd(gen, n))
        print('r= ', i+1, 'pval= ', p_values[i + 1])
    p_values[0] = chi_test(p_values)
    print(p_values)
    np.savetxt(gen_name+"_"+test_name+"_pvalues.csv", p_values)
    return p_values[0]
####################################################################################################

#TESTING
#PCG64, EXCEL TESTTING
#gen = PCG64(0)
#gen = excel()
#gen = excel(1812433253)
#print(test_gen(gen, lambda r: freq_monobit_test(r, max=32, type='float'), n=2**(18-5), r=100, gen_name="excel18", test_name="MONO")) #
#print(test_gen(gen, lambda r: freq_block_test(r, M=2**12, max=32, type='float'), n=2**(18-5), r=100, gen_name="excel18", test_name="BLOCK"))
#print(test_gen(gen, lambda r: chi_test(r, max=1), n=2**18, r=100, gen_name="excel18", test_name="CHI2"))
#print(test_gen(gen, lambda r: ks_test(r, max=1), n=2**18, r=100, gen_name="excel18", test_name="KS"))

#LCG TESTING
#gen = LCG(13, 1, 5, 1)
#print(test_gen(gen, lambda r: freq_monobit_test(r, max=4, type='int'), n=2**(18-2), r=100, gen_name="LCG", test_name="MONO")) #
#print(test_gen(gen, lambda r: freq_block_test(r, M=2**12, max=4, type='int'), n=2**(18-2), r=100, gen_name="LCG", test_name="BLOCK"))
#print(test_gen(gen, lambda r: chi_test(r, max=13), n=2**18, r=100, gen_name="LCG", test_name="CHI2"))
#print(test_gen(gen, lambda r: ks_test(r, max=13), n=2**18, r=100, gen_name="LCG", test_name="KS"))

#LCG, GLCG TESTING
#gen = LCG(2**10, 3, 7, 0) #(6*613+7)%2**10 == 613
#gen = GLCG()
#print(test_gen(gen, lambda r: freq_monobit_test(r, max=10, type='int'), n=int(np.floor((2**18)/10)), r=100, gen_name="GLCG", test_name="MONO")) #
#print(test_gen(gen, lambda r: freq_block_test(r, M=2**12, max=10, type='int'), n=int(np.floor((2**18)/10)), r=100, gen_name="GLCG", test_name="BLOCK"))
#print(test_gen(gen, lambda r: chi_test(r, max=2**10), n=2**18, r=100, gen_name="GLCG", test_name="CHI2"))
#print(test_gen(gen, lambda r: ks_test(r, max=2**10), n=2**18, r=100, gen_name="GLCG", test_name="KS"))

#RC4 TESTING
#gen = RC4(2**5, np.arange(2**5))
gen = RC4_keys(keys())
print(test_gen(gen, lambda r: freq_monobit_test(r, max=5, type='int'), n=int(np.floor((2**18)/5)), r=100, gen_name="RC4_keys", test_name="MONO")) #
print(test_gen(gen, lambda r: freq_block_test(r, M=2**12, max=5, type='int'), n=int(np.floor((2**18)/5)), r=100, gen_name="RC4_keys", test_name="BLOCK"))
print(test_gen(gen, lambda r: chi_test(r, max=2**5), n=2**18, r=100, gen_name="RC4_keys", test_name="CHI2"))
print(test_gen(gen, lambda r: ks_test(r, max=2**5), n=2**18, r=100, gen_name="RC4_keys", test_name="KS"))

#CONSTANT TESTING
#const = str_to_array(read_const_to_str("pi.txt"))
#const = str_to_array(read_const_to_str("e.txt"))
#const = str_to_array(read_const_to_str("sqrt2.txt"))
#gen = irrational_gen(const)
#print(test_gen(gen, lambda r: freq_monobit_test(r, max=32, type='bin'), n=int(np.floor(len(const)/100)), r=100, gen_name="sqrt2", test_name="MONO")) #
#print(test_gen(gen, lambda r: freq_block_test(r, M=2**7, max=32, type='bin'), n=int(np.floor(len(const)/100)), r=100, gen_name="sqrt2", test_name="BLOCK"))
