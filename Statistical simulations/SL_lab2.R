#Zad 1
library(pracma)
library(glmnet)
library(bigstep)
library(MASS)

generate_design <- function(n=1000, p=950) {
  X = matrix(rnorm(n*p, 0, 1/sqrt(n)), n, p)
  return(X)
}

generate_response <- function(X, beta=3.5, nonzero=20, rdist=rnorm) {
  betas = c(rep(beta,nonzero), rep(0,ncol(X)-nonzero))
  X%*%betas + rdist(nrow(X))
}

#i)
norm2 <- function(X) sum(X^2)
gamma_min <- function(beta) length(beta)/norm2(beta)

#ii)
bias_thr <- function(beta, gamma) -gamma/(1+gamma) * beta
bias_norm_thr <- function(beta, gamma) (gamma/(1+gamma))^2 * norm2(beta)
bias_norm_est <- function(beta, beta_hat) norm2(beta_hat - beta)

cov_beta_thr <- function(beta, gamma) 1/(1+gamma)^2 * diag(length(beta))
cov_trace_thr <- function(beta, gamma) length(beta)/(1+gamma)^2
cov_trace_est <- function(beta_hat) sum(diag(cov(beta_hat,beta_hat)))

mse_thr <- function(beta, gamma) bias_norm_thr(beta, gamma) + cov_trace_thr(beta, gamma)
se <- function(beta, beta_hat) sum((beta_hat - beta)^2)

#iii)
power_thr <- function(beta1, p=950, alpha=.1) {
  c = qnorm(1-alpha/(2*p))
  pnorm(-c - beta1) + 1 - pnorm(c - beta1)
}
power_est <- function(beta, beta_hat, gamma, p=950, alpha=.1) {
  c = qnorm(1-alpha/(2*p)) / (1+gamma)
  test = abs(beta_hat[beta!=0]) > c
  sum(test) / sum(beta!=0)
}

fwer_est <- function(beta, beta_hat, gamma, p=950, alpha=.1) {
  c = qnorm(1-alpha/(2*p)) / (1+gamma)
  test = abs(beta_hat[beta==0]) > c
  sum(test)>0
}

#iv)
ex1 <- function(n=1000, p=950, beta=3.5, nonzero=c(20,100,200), rep=200) {
  beta_hat=matrix(0,rep,p); OLS=matrix(0,rep,p); ridge=matrix(0,rep,p);
  beta_hat_results=data.frame(); OLS_results=data.frame(); ridge_results=data.frame();
  results_list = list(beta_hat_results, OLS_results, ridge_results)
  names(results_list) = c("OLS", "ridge", "beta_hat")
  
  X = randortho(n)[,1:p]#X = generate_design(n,p,beta,max(nonzero))
  for (k in nonzero) {
    betas = c(rep(beta,k), rep(0,p-k))
    gamma = gamma_min(betas)
    for (r in 1:rep) {
      Y = X%*%betas + rnorm(n)
      OLS[r,] = coef(lm(Y~X-1))
      ridge[r,] = coef(glmnet(X, Y, lambda=gamma/n, alpha = 0, intercept=FALSE, standardize = FALSE))[-1,1]
      beta_hat[r,] = 1/(1+gamma) * t(X) %*% Y
      cat("k= ",k,"r= ",r,"\n")
    }
    
    est_list = list(OLS, ridge, beta_hat)
    for (i in 1:length(est_list)) {
      est = est_list[[i]]
      bias = bias_norm_est(betas, apply(est, MARGIN=2, mean))
      tr_cov = sum(apply(est, MARGIN=2, var))
      mse = mean(apply(est, MARGIN=1, function(b)se(betas, b)))
      
      power = mean(apply(est, MARGIN=1, function(b)power_est(betas,b, ifelse(i==1, 0, gamma))))
      fwer = mean(apply(est, MARGIN=1, function(b)fwer_est(betas,b, ifelse(i==1, 0, gamma))))
      
      results_list[[i]] = rbind(results_list[[i]], c(bias, tr_cov, mse, power, fwer))
    }
    
  }
  
  save(list=c("results_list"), file=paste0("SL_lab2_ex1",".RData"))  
  for (i in 1:length(results_list)) {
    name = names(results_list)[i]
    colnames(results_list[[i]]) = c("bias", "tr_cov", "mse", "power", "fwer")
    rownames(results_list[[i]]) = paste(nonzero)
    printTable(results_list[[i]], paste("Results for", name), paste0("SL_lab2_ex1_",name))
  }
  
  return(results_list)
}

ex1_thr <- function(n=1000, p=950, beta=3.5, nonzero=c(20,100,200)) {
  results_thr = data.frame()
  for (k in nonzero) {
    betas = c(rep(beta,k), rep(0,p-k))
    gamma = gamma_min(betas)
    bias = bias_norm_thr(betas,gamma)
    tr_cov = cov_trace_thr(betas,gamma)
    mse = mse_thr(betas,gamma)
    power = power_thr(beta, p)
    results_thr = rbind(results_thr, c(bias, tr_cov, mse, power))
  }
  
  colnames(results_thr) = c("bias", "tr_cov", "mse", "power")
  rownames(results_thr) = paste(nonzero)
  save(list=c("results_thr"), file=paste("SL_lab2_ex1_thr",".RData", sep="")) 
  printTable(results_thr, "Theoretical results", "SL_lab2_ex1_thr")
  return(results_thr)
}

####################################################################################################################################
#Ex.2 
SE <- function(Yhat, Y) norm2(Y - Yhat)

PE_est <- function(RSS, eigen, gamma, sigma=1) {
  RSS + 2*sigma^2 * sum( eigen / (eigen + gamma))
}

ridge_SURE_min2 <- function(X,Y) {
  beta = c(rep(3.5,20), rep(0,930))
  lambdaseq = seq(from=0.0001,to=0.005, length.out = 200)
  values = eigen(t(X)%*%X, only.values = TRUE)$values
  n = nrow(X)
  obj = glmnet(X,Y,alpha=0,intercept=FALSE, standardize=FALSE, lambda=lambdaseq)
  betaridge = obj$beta
  lambdas = obj$lambda
  
  sureridge = NULL
  for (u in 1:ncol(betaridge))
    sureridge[u] = PE_est(sum((X%*%betaridge[,u]-Y)^2), values, lambdas[u]*n)

  ind = which.min(sureridge)
  return(betaridge[,ind])
}

ols_mbic2 <- function(X,Y) {
  model = as.numeric(fast_forward(prepare_data(Y,X),crit=mbic2)$model)
  beta =  rep(0, ncol(X))
  if(length(model)!=0)
    beta[model] = coef(lm(Y~X[,model]-1))
  return(beta)
}

ex2 <- function(n=1000, p=950, beta=3.5, nonzero=c(20,100,200), rep=100, generate_x=generate_design) {
  SURE = matrix(0, rep, p); CV = matrix(0, rep, p); OLS = matrix(0, rep, p); MBIC2 = matrix(0, rep, p);
  results_list = list(data.frame(), data.frame(), data.frame(), data.frame())
  names(results_list) = c("SURE", "CV", "OLS", "MBIC2")
  
  X = generate_x(n,p)
  for (k in nonzero) {
    betas = c(rep(beta,k), rep(0,p-k))
    for (r in 1:rep) {
      Y = X%*%betas + rnorm(n)
      SURE[r,] = ridge_SURE_min2(X,Y)
      CV[r,] = coef(cv.glmnet(X,Y,alpha=0,intercept=FALSE, standardize=FALSE))[-1,1]
      OLS[r,] = coef(lm(Y~X-1))
      MBIC2[r,] = ols_mbic2(X,Y)
      cat("k= ",k,"r= ",r,"\n")
    }
    
    est_list = list(SURE, CV, OLS, MBIC2)
    for (i in 1:length(est_list)) {
      est = est_list[[i]]
      mse = mean(apply(est, 1, function(b)norm2(b-betas)))
      mu = mean(apply(est, 1, function(b)norm2(X%*%(b-betas))))
      results_list[[i]] = rbind(results_list[[i]], c(mse,mu))
    }
    
  }
  
  save(list=c("results_list"), file=paste0("SL_lab2_ex4_beta5",".RData"))  
  for (i in 1:length(results_list)) {
    name = names(results_list)[i]
    colnames(results_list[[i]]) = c("MSE(beta)", "MSE(mu)")
    rownames(results_list[[i]]) = paste(nonzero)
    printTable(results_list[[i]], paste("Results for", name), paste0("SL_lab2_ex4_",name,"_beta5"))
  }
  
  return(results_list)
}

####################################################################################################################################
#Ex.3
ex2(beta = 5) #REMBEMER FILE NAME

####################################################################################################################################
#Ex.4
generate_mvrnorm <- function(n=1000, p=950, ro=0.5) {
  Sigma = matrix(ro, p, p)
  diag(Sigma) = 1
  X = mvrnorm(n, rep(0,p), Sigma)/n
}

ex2(generate_x = generate_mvrnorm) #REMEMBER FILE NAME, FAST_FORWARD!
ex2(beta = 5, generate_x = generate_mvrnorm)

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
library(xtable)
printTable <- function(data, cap, path) {
  print(xtable(data, auto = TRUE, caption = cap, digits=3), 
        file=paste(path, ".tex",sep=""), table.placement = "H", 
        floating.environment = "subtable",
        include.rowname = TRUE)
}
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


ridge <- function(X, Y, gamma) {
  solve( t(X)%*%X + gamma*diag(ncol(X))  ) %*% t(X) %*% Y
}

ridge_SURE_min <- function(X, Y, interval_len=10, eps=.01, min_it=20) {
  eigen_values = eigen(t(X)%*%X, only.values = TRUE)$values
  gamma_int = seq(0, 10, length.out = interval_len)
  pe_tmp = 10^5; pe_min = 10^6;
  while (abs(pe_min - pe_tmp)>eps | pe_min==pe_tmp) {
    pe_tmp = pe_min
    
    pe = NULL
    for (i in 1:interval_len) {
      gamma = gamma_int[i]
      glm_mod = glmnet(X, Y, lambda=gamma/nrow(X), alpha = 0, intercept=FALSE, standardize = FALSE)#beta_hat = ridge(X, Y, gamma)
      rss = sum((predict(glm_mod, X)- Y)^2)#rss = RSS(X%*%beta_hat, Y)
      pe[i] = PE_est(rss, eigen_values, gamma)
      cat("gamma=",gamma, "pe=",pe[i],"\n")
    }
    ind_min = which.min(pe)
    pe_min = pe[ind_min]
    ind_left = ifelse(ind_min>1, ind_min-1, 1)
    ind_right = ifelse(ind_min<interval_len, ind_min+1, interval_len)
    gamma_int = seq(gamma_int[ind_left], gamma_int[ind_right], length.out=interval_len)
    print(gamma_int)
  }
  return(gamma_int)
}
