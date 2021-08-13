#Zad 1
generate_design <- function(n=1000, p=950, beta=3, nonzero=5) {
  X = matrix(rnorm(n*p, 0, 1/sqrt(n)), n, p)
  betas = c(rep(beta,nonzero), rep(0,p-nonzero))
  colnames(X) = betas
  return(X)
}
  
generate_response <- function(X, rdist=rnorm) {
  betas = as.numeric(colnames(X))
  X%*%betas + rdist(nrow(X))
}

PE_LS <- function(X, betas_hat) {
  betas = as.numeric(colnames(X))
  mean(sum( (X%*%(betas-betas_hat)+rnorm(nrow(X)))^2 ))
}

PE_RSS <- function(RSS, sigma=NULL, n, p) {
  if(is.null(sigma))
    RSS + 2*(RSS/(n-p))*p
  else
    RSS + 2*sigma^2*p
}

PE_CV <- function(X, Y, Y_hat) {
  H = X%*%solve(t(X)%*%X)%*%t(X)
  sum( ((Y-Y_hat) / (1-diag(H)) )^2);
}

AIC <- function(RSS, n, p) n*log(RSS) +2*p

PE_estimators <- function(n=1000, P = c(5,10,20,100,500,950), beta=3, nonzero=5) {
  data = data.frame()
  X = generate_design(n,max(P),beta,nonzero)
  Y = generate_response(X)
  
  for (p in P) {
    model = lm(Y~X[,1:p]-1)
    RSS = sum(model$residuals^2)
    data = rbind(data, c(PE_LS(X[,1:p], model$coefficients), PE_RSS(RSS,1,n,p), PE_RSS(RSS,NULL,n,p), #1/sqrt(n)?
                         PE_CV(X[,1:p],Y,predict(model)), AIC(RSS,n,p) ) )
  }
  colnames(data) = c( "PE_LS", "PE_sigma", "PE_RSS", "PE_CV", "AIC")
  rownames(data) = P
  #printTable(data, "Estimators of PE.", "SL_lab1_ex1_PE_est")
  return(data)
}

select_model <- function(PE_data) {
  best_known = as.numeric(rownames(PE_data)[which.min(PE_data$`PE_sigma`)])
  best_unknown = as.numeric(rownames(PE_data)[which.min(PE_data$`AIC`)])
  data.frame(PE_sigma=best_known, AIC=best_unknown)
}

PE_comparison <- function(n=1000, P = c(5,10,20,100,500,950), beta=3, nonzero=5, rep=100) {
  data = matrix(0, length(P), 5)
  SIG = data.frame(); RSS = data.frame(); CV = data.frame(); best_models=data.frame();
  for (i in 1:rep) {
    est = PE_estimators(n,P,beta,nonzero)
    data = data + est
    SIG = rbind(SIG, est$`PE_sigma`-est$`PE_LS`)
    RSS = rbind(RSS, est$`PE_RSS`-est$`PE_LS`)
    CV = rbind(CV, est$`PE_CV`-est$`PE_LS`)
    #best_models = rbind(best_models, select_model(est[-6,]))
  }
  data = data/rep
  colnames(SIG) = paste(P)
  colnames(RSS) = paste(P)
  colnames(CV) = paste(P)

  #FP = data.frame(PE_sigma=best_models$`PE_sigma`-nonzero, AIC=best_models$`AIC`-nonzero)
  #FN = data.frame(PE_sigma=best_models$`PE_sigma`-nonzero, AIC=best_models$`AIC`-nonzero)
  
  par(mfrow=c(2,2))
  boxplot( SIG, main="PE_sigma" )
  boxplot( RSS, main="PE_RSS" )
  boxplot( CV, main="PE_CV" )
  #par(mfrow=c(2,2))
  #hist(FP$`PE_sigma`, main="FP PE_sigma", xlim = c(0,1000))
  #hist(FP$`AIC`, main="FP AIC", xlim = c(0,1000))
  #hist(FN$`PE_sigma`, main="FN PE_sigma", xlim = c(0,1000))
  #hist(FN$`AIC`, main="FN AIC", xlim = c(0,1000))
  par(mfrow=c(1,1))
  
  save(list=c("data"), file=paste("SL_lab1_ex1_PE_comp",".RData", sep=""))  
  #printTable(data, "Mean of estimators of PE.", "SL_lab1_ex1_PE_comp")
  return(data)
}

####################################################################################################################################
#Zad 2
library(bigstep)
ric <- function(loglik, k, n, p) -2*loglik + 2*k*log(p)

covariates_identification <- function(n=1000, P=c(20,100,500,950), beta=3, nonzero=5, methods = c("bic", "mbic", "mbic2", "aic", "ric"),  rdist=rnorm, type="linear") {
  X = generate_design(n,max(P),beta,nonzero)
  Y = generate_response(X, rdist)
  methods_stats = list()
  for (crit_met in methods) {
    MSE=NULL; FD=NULL; TD=NULL; data=data.frame()
    
    for (p in P) {
      if(crit_met=="ric")
        FF_obj =  fast_forward(prepare_data(Y,unname(X[,1:p]), type=type), crit=ric, maxf = p)
      else if(crit_met=="rbic")
        FF_obj =  fast_forward(prepare_data(rank(Y),unname(X[,1:p]), type=type), crit="mbic", maxf = p)
      else if(crit_met=="rbic2")
        FF_obj =  fast_forward(prepare_data(rank(Y),unname(X[,1:p]), type=type), crit="mbic2", maxf = p)
      else 
        FF_obj =  fast_forward(prepare_data(Y,unname(X[,1:p]), type=type), crit=crit_met, maxf = p)

      if(is.null(FF_obj$model)){
        FD = 0
        TD = 0
        MSE = sum(lm(Y~1)$residuals^2)/n
      }else{
        FD = sum(!(as.numeric(FF_obj$model) %in% 1:nonzero))
        TD = sum(as.numeric(FF_obj$model) %in% 1:nonzero)
        MSE = FF_obj$metric_v
      }
      data = rbind(data, c(MSE, FD, TD, TD+FD, FD/max(TD+FD,1), TD/nonzero))
    }
    rownames(data) = paste(P)
    colnames(data) = c("MSE", "FD", "TD", "D", "FDR", "power")
    methods_stats[[crit_met]] = data
  }
  
  return(methods_stats)
}

estimate_cov_identification <- function(n=1000, P=c(20,100,500,950), beta=3, nonzero=5, methods = c("bic", "mbic", "mbic2", "aic","ric"), rep=100,  rdist=rnorm, type="linear") {
  methods_stats = covariates_identification(n, P, beta, nonzero, methods, rdist, type)
  for (i in 1:(rep-1)) {
    tmp = covariates_identification(n, P, beta, nonzero, methods, rdist, type)
    for (j in 1:length(methods)) 
      methods_stats[[j]] =  methods_stats[[j]] + tmp[[j]]
  }
  
  for (i in 1:length(methods_stats)) {
    methods_stats[[i]] = methods_stats[[i]]/rep
  }

  save(list=c("methods_stats"), file=paste("SL_lab1_ex5_est_covariates",".RData", sep=""))  
  for (i in 1:length(methods_stats)) {
    printTable(methods_stats[[i]], paste("Statistics for", methods[i]), paste("SL_lab1_ex5_est_covariates_",methods[i],sep=""))
  }#SUBTABLES AND 2 DIGITS!!!!
  
  return(methods_stats) 
}

####################################################################################################################################
#Zad 3
estimate_cov_identification(P=c(950), nonzero = 50, methods = c("mbic", "mbic2", "ric"))

####################################################################################################################################
#Zad 4
estimate_cov_identification(P=c(950), beta = 10, nonzero = 30, methods = c("mbic", "mbic2", "rbic", "rbic2"), rdist=rexp, rep=100)
estimate_cov_identification(P=c(950), beta = 10, nonzero = 30, methods = c("mbic", "mbic2", "rbic", "rbic2"), rdist=rcauchy, rep=100)

X = generate_design(1000,950,10,30)
Y = generate_response(X, rcauchy)
FF_obj =  fast_forward(prepare_data(rank(Y),unname(X)), crit="mbic2", maxf = 950)
model = as.numeric(FF_obj$model)
library(MASS)
LS = lm(Y~unname(X[,model])-1)$coefficients
hub = rlm(X[,model],Y,psi=psi.huber)$coefficients
bisq = rlm(X[,model],Y,psi=psi.bisquare)$coefficients
mean((LS- 10)^2)

####################################################################################################################################
#Zad 5
generate_response <- function(X, rdist=rnorm) {
  betas = as.numeric(colnames(X))
  logit = X%*%betas
  p = exp(logit)/(1+exp(logit))
  rbinom(nrow(X), 1, p)
}
estimate_cov_identification(P=c(950), beta = 10, nonzero = 30, methods = c("bic", "mbic", "mbic2"), rep=100, type = "logistic")

estimate_coefficients <- function(n=1000, P=c(950), beta = 10, nonzero = 30, methods = c("bic", "mbic", "mbic2"), rep=100, type = "logistic") {
  MSE = matrix(0, length(P), length(methods))
  for (r in 1:rep) {
    X = generate_design(n,max(P),beta,nonzero)
    Y = generate_response(X)
    mse = matrix(0, length(P), length(methods))
    for (i in 1:length(methods)) {
      crit_met = methods[i]
      for (j in 1:length(P)) {
        p = P[j]
        FF_obj =  fast_forward(prepare_data(Y,unname(X[,1:p]), type=type), crit=crit_met, maxf = p)
        model = as.numeric(FF_obj$model)
        ls = lm(Y~X[,model]-1)$coefficients
        mse[j,i] = mean((ls - beta)^2)
      }
    }
    MSE = MSE + mse
  }
  
  MSE = data.frame(MSE/rep)
  colnames(MSE) = methods
  rownames(MSE) = P
  
  printTable(MSE, "MSE of estimates of coefficients.", "SL_lab1_ex5_mse")
  return(MSE)
}
estimate_coefficients(rep=10)

X = matrix(rnorm(1000*950, 0, 1/sqrt(1000)), 1000, 950)
logit = X %*% c(rep(20,30), rep(0,920))
pr=exp(logit)/(1+exp(logit));
Y<-rbinom(1000,1,pr);
d = prepare_data(Y,X, type="logistic")
FF_obj = fast_forward(d, crit="aic", maxf = 950)







####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
library(xtable)
printTable <- function(data, cap, path) {
  print(xtable(data, auto = TRUE, caption = cap, digits=2), 
        file=paste(path, ".tex",sep=""), table.placement = "H", 
        floating.environment = "subtable",
        include.rowname = TRUE)
}
