# **University Projects**

The repository consists of projects made by me (or in collaboration with someone mentioned in project). The code is exclusively written by me (with one exeption that is also mentioned). The repository holds only a few of my many projects and it serves a purpose of presenting my skills in fields of programming and mathematics. At the moment, the projects contain my bachelor thesis and codes in Python, R, SQL (more details below).

### **Probability**
* [Motif finding in DNA sequences](https://github.com/Homeomorphistic/University-projects/tree/master/Motif%20fiding%20in%20DNA%20sequences%20(EM%20algorithm)),
* [Pseudorandom number generators](https://github.com/Homeomorphistic/University-projects/tree/master/Pseudorandom%20number%20generators),
* [Recommender system](https://github.com/Homeomorphistic/University-projects/tree/master/Recommender%20system%20(PCA%20and%20SVD)),
* [Monte Carlo methods and simulations](https://github.com/Homeomorphistic/University-projects/tree/master/Monte%20Carlo%20methods%20and%20simulations).

### **Deterministic models**
* [Epidemic modeling](https://github.com/Homeomorphistic/University-projects/tree/master/Epidemic%20modeling%20(SEIR%20model)),
* [Heat equation](https://github.com/Homeomorphistic/University-projects/tree/master/Diffusion%20equation%20(heat%20modeling)),
* [Spiral waves](https://github.com/Homeomorphistic/University-projects/tree/master/Reaction-diffusion%20equations%20(Turing%20instability)).

### **Statistics**
* [A/B testing](https://github.com/Homeomorphistic/University-projects/tree/master/AB%20testing),
* [Statistical simulations](https://github.com/Homeomorphistic/University-projects/tree/master/Statistical%20simulations),
* [Valuation of portfolios](https://github.com/Homeomorphistic/University-projects/tree/master/Valuation%20of%20portfolios%20(random%20forests)),
* [Bootstrapping](https://github.com/Homeomorphistic/University-projects/tree/master/Resampling%20methods%20(bootstrap)).

### **Databases**
* [Ski jumping database](https://github.com/Homeomorphistic/University-projects/tree/master/Ski%20jumping%20database),
* [Major league soccer database](https://github.com/Homeomorphistic/University-projects/tree/master/Major%20league%20soccer%20database).

### **Theses**
* [Statistical analysis of codon frequency in genetic code](https://github.com/Homeomorphistic/University-projects/tree/master/Statistical%20analysis%20of%20codon%20frequency%20in%20genetic%20code).
* [Locally-informed proposals in Metropolis-Hastings algorithm with applications](https://github.com/Homeomorphistic/University-projects/tree/master/Locally-informed%20proposals%20in%20Metropolis-Hastings%20algorithm%20with%20applications)


# **Probability**

## **Motif finding in DNA sequences [Python/Numpy]**

Motif is a sequance of nucleotide. Given an array of motifs can one recover the probabilities of occurence of nucleotides? What if there are 2 different probability distributions? This project seeks to find answers to those questions using _**Expectation-Maximization algorithm (EM)**_. It is testing it performance on different types of data, which is entirely simulated.

## **Pseudorandom number generators [Python/Numpy]**

Randomness cannot be created on deterministic machines like computers. So, can we at least get close to real randomness? This project is exploring topic of pseudorandomness by constructing many _**pseudorandom number generators (PRNG)**_ and testing them with statistical methods.

## **Recommender system [Python/Numpy]**

Recommender system is a subclass of information filtering systems that seek to predict the "rating" or "preference" of a user. This project is based on ratings of movies. Given their past, how will their future ratings look? Answers to this question are provided by testing _**non-negative matrix factorization (NMF)**_ and _**singural-value decomposition (SVD)**_.

## **Monte Carlo methods and simulations [R]**

This project is about simulating stochastic processes and using basic Monte Carlo methods to answer real problems of assembly line production. The projects shows strength of using simulations, instead of long-term gathering of data. The fundamental _**Crude Monte Carlo estimator (CMC)**_ is used extensively to answer questions about minimizing the cost of downtime of machines or the length of a queue in asembly line.


# **Deterministic models**

## **Epidemic modeling [Python]**

In 2019 coronavirus has given rise to global pandemic. Now the disease is a worldwide problem, so governments and scientists are looking for solutions to stop the pandemic. One tool for understanding the problem are _**deterministic models**_ that use _**differential equations**_. The model employed to predict the pandemic outbreak is an _**infectious disease dynamics model (SEIR model)**_.

## **Heat equation [Python]**

The heat equation is one of many _**partial differential equations**_, yet it is one of the most studied among physicists and mathematicians. It is important in many fields of applied mathematics including probability theory and finacial mathematics. This project focuses on solving this equation on a _**rod (1 dimension)**_ and a _**plate (2 dimensions)**_ and presenting those solutions in _**3D animations**_.

## **Spiral waves [Python]**

The _**reaction-diffusion systems**_ are mathematical models used in many scientific fields (chemistry, physics, biology etc.) to model real world phenomena using _**partial differential equations**_ with _**diffusion**_ (heat). This project examines special case of _**FitzHugh-Nagumo model**_, which is used to model electricity in neurons and tries to introduce instabilities (including _**Turing instability**_) to the system that will create special patterns like _**spiral waves**_.


# **Statistics**

## **A/B testing [Python/Numpy/Pandas]**

A/B testing is a user experience reseach methodology. Given 2 ways of presenting data, which one is better? _**Statistical hypothesis testing**_ is a helpful tool in a problem of different hints that are shown to browser users.

## **Statistical simulations [R]**

This project is more of a series of labs, that are testing many statistical models by simulations. This project includes testing of: _**information criteria**_ (model selection), regularization methods (_**SLOPE, LASSO, ridge regression**_ and their modifications), _**PCA**_ with _**PESEL**_ procedure.

## **Valuation of portfolios [R]**

Porfolio evaluation is establishing the value of each asset owned by the investment fund or entity. In case of this project the payment of liabilities was evaluated. It was done by constructing _**binary trees, pruned trees**_ and finaly _**random forrests**_. The project included practical side of statistics: _**imputation of missing data**_, rating models with _**Gini index**_ or _**logloss**_ and inferring on various plots like _**partial plot**_.

## **Bootstrapping [R]**

Bootstrapping is method of resampling the data to _**estimate the properties of estimator**_. It is helpful when the distribution of statistic is not known. It can be used in a variety of statistical methods: hypothesis testing, statistical modeling, _**computing confidence intervals**_. This series of projects uses publicly available data about car crashes and wages to test out _**bootstrap methods**_.


# **Databases**

## **Ski jumping database [PostgreSQL]**

The data on ski jumping competitions provided by FIS proves to be really inaccessible and needs to be reorganized. The aim of this project is to construct basic database to clear up the data and test some database functionalities (_**views, functions, triggers**_).

## **Major league soccer database [MySQL]**

Major league soccer (MLS) is the highest football league in USA. This database was intended to help with gathering and processing the information, while working on a project with other scientists. It includes the extraction of the data from spreadsheets, reforming them into proper RDB, correcting errors and creating special saves for other users.


# **Theses**

## **Statistical analysis of codon frequency in genetic code [R]**

The genome is a medium of information in living organisms, the whole growth process and functionality of an organism depends on it. It is build by a triplets of nucleotides, called a _codons_. Are those randomly distributed in a genome? Are there any relations between them? Are we able to use classic statistical methods such as _**Chi squared test**_ or _**Maximum likelihood estimators**_ in a problem with tens of thousands of observations? My bachelor thesis seeks to answer those questions and analyze the tools used in those problems.


## **Locally-informed proposals in Metropolis-Hastings algorithm with applications [Python]**

The Markov Chain Monte Carlo methods (MCMC) are a family of algorithms used for approximating sampling from a given probability distribution. They prove very effective when the state space is large. This fact can be used to solve many hard deterministic problems -- one of them being _**traveling salesmen problem**_. It will be used in this thesis to test a new approach of _**locally-informed proposals**_ as a modification of well known _**Metropolis-Hastings algorithm**_.





