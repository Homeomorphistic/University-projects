library(glmnet)
library(SLOPE)
library(MASS)
#Zad 1
norm2 <- function(X,Y) sum((X-Y)^2)

generate_design <- function(n=500, p=450, correlated=FALSE, rho=.5) {
  if(!correlated)
    X = matrix(rnorm(n*p, 0, 1/sqrt(n)), n, p)
  else{
    Sigma = matrix(rho,p,p)
    diag(Sigma) = 1
    X = matrix(mvrnorm(n, rep(0,p), Sigma/n), n, p) #sqrt n?
  }
  return(X)
}
generate_response <- function(X, beta=10, nonzero=5, rdist=rnorm) {
  betas = c(rep(beta,nonzero), rep(0,ncol(X)-nonzero))
  X%*%betas + 2*rdist(nrow(X))
}

fdp <- function(tests, k, p) {
  d = sum(tests!=0)
  fd = sum(tests[(k+1):p]!=0)
  return(fd/max(d,1))
}

ifFalseDiscovery <- function(tests, k, p) {
  fd = sum(tests[(k+1):p]!=0)
  return(fd>0)
}

W <- function(beta_hat, p=length(beta_hat)/2) abs(beta_hat[1:p])-abs(beta_hat[(p+1):(2*p)]) #LASSO coefficient difference stat

threshold_W <- function(w, q=.2) { 
  ord = order(abs(w),decreasing=TRUE)
  fd = cumsum(w[ord]<0)
  nd = cumsum(w[ord]>0)
  fdr = (fd+1)/nd
  t_min = Inf 
  if(sum(fdr<=q)>0){
    t_ind = ord[max(which(fdr<=q))]
    t_min = abs(w[t_ind])
  }
  return(t_min)
}

knockoff_select <- function(beta_hat, q=.2) {
  p = length(beta_hat)/2
  w = W(beta_hat)
  t_min = threshold_W(w, q)
  if(t_min!=Inf){
    sel_id = which(w > t_min) #>=?
    beta_hat[-sel_id] = 0
  }
  else
    beta_hat[1:p] = rep(0,p)
  return(beta_hat[1:p])
}

adaptive_lasso1_select <- function(X, Y, beta_hat) {
  nonzero_id = which(beta_hat!=0)
  X_nonzero = X[,nonzero_id]
  beta_nonzero =  beta_hat[nonzero_id]
  
  W = 1/abs(beta_nonzero)
  X_nonzero = sweep(X_nonzero,2,W,'/') #*
  
  ad_lasso = cv.glmnet(X_nonzero, Y, intercept=FALSE, standardize=FALSE)
  beta_nonzero = rep(0, ncol(X))
  beta_nonzero[nonzero_id] = coef(ad_lasso, s='lambda.min')[-1,1] / W #*
  
  return(beta_nonzero)
}

adaptive_lasso2_select <- function(X, Y, beta_hat, q=.2) {
  n = nrow(X); p = ncol(X); lambda_alasso = qnorm(1-q/2/p) #CONSTANTS
  
  RSS = sum((Y- X%*%beta_hat)^2)
  nonzero_id = which(beta_hat!=0)
  X_nonzero = X[,nonzero_id]
  beta_nonzero =  beta_hat[nonzero_id]
  l = length(nonzero_id)
  sigma_lassoCV = sqrt(RSS/(n-l))
  
  W = sigma_lassoCV/abs(beta_nonzero)
  X_nonzero = sweep(X_nonzero,2,W,'/')
  
  ad_lasso = glmnet(X_nonzero, Y, intercept=FALSE, standardize=FALSE,lambda=sigma_lassoCV*lambda_alasso/n)
  beta_nonzero = rep(0,p)
  beta_nonzero[nonzero_id] = coef(ad_lasso)[-1,1] / W 
  return(beta_nonzero)
}

adaptive_slope_select <- function(X, Y, beta_hat, q=.2) {
  n = nrow(X); p = ncol(X); #CONSTANTS
  
  RSS = sum((Y- X%*%beta_hat)^2)
  nonzero_id = which(beta_hat!=0)
  l = length(nonzero_id)
  sigma_lassoCV = sqrt(RSS/(n-l))
  
  W = sigma_lassoCV/abs(beta_hat+0.000001) # beta_nonzero
  X = sweep(X,2,W,'/')
  
  ad_slope = SLOPE(X,Y,q=q, alpha=1/n*sigma_lassoCV, lambda='bh', solver='admm',max_passes=100, scale='none', intercept = FALSE)
  return(coef(ad_slope) / W)
}

ex1 <- function(n=500, p=450, beta=10, nonzero=c(5,20,50), rep=100, q=.2, correlated=FALSE, rho=.5) {
  OLS = matrix(0,rep,p); ridge = matrix(0,rep,p); LASSO = matrix(0,rep,p); ridge_kf = matrix(0,rep,p); 
  LASSO_kf = matrix(0,rep,p); ALASSO1 = matrix(0,rep,p); ALASSO2 = matrix(0,rep,p); ASLOPE = matrix(0,rep,p);
  results_list = list()
  X = generate_design(n,p,correlated,rho)
  
  if(!correlated)
    X_aug = cbind(X, generate_design(n,p,correlated,rho))
  else{
    Sigma = matrix(rho,p,p)
    diag(Sigma) = 1
    s = min(eigen(Sigma)$values)
    s = min(2*s,1)
    sseq = rep(s,p)
    V = 2*diag(sseq)-diag(sseq)%*%solve(Sigma)%*%diag(sseq)
    mu = X-X%*%solve(Sigma)%*%diag(sseq)
    X_aug = cbind(X, mu+mvrnorm(n,rep(0,p),V)/sqrt(n))
  }
  
  for (j in 1:length(nonzero)) {
    k = nonzero[j]
    betas = c(rep(beta,k), rep(0,p-k))
    for (r in 1:rep) {
      Y = generate_response(X,beta,k)
      OLS[r,] = coef(lm(Y~X-1))
      ridge[r,] = coef(cv.glmnet(X,Y,alpha=0,intercept=FALSE, standardize=FALSE), s='lambda.min')[-1,1]
      LASSO[r,] = coef(cv.glmnet(X,Y,intercept=FALSE, standardize=FALSE), s='lambda.min')[-1,1]
      
      ridge_aug = coef(cv.glmnet(X_aug,Y,alpha=0,intercept=FALSE, standardize=FALSE), s='lambda.min')[-1,1]
      LASSO_aug = coef(cv.glmnet(X_aug,Y,intercept=FALSE, standardize=FALSE), s='lambda.min')[-1,1]
      ridge_kf[r,]  = knockoff_select(ridge_aug,q)
      LASSO_kf[r,]  = knockoff_select(LASSO_aug,q)
      
      ALASSO1[r,] = adaptive_lasso1_select(X,Y,LASSO[r,])
      ALASSO2[r,] = adaptive_lasso2_select(X,Y,LASSO[r,],q)
      ASLOPE[r,] = adaptive_slope_select(X,Y,LASSO[r,],q)
      cat("k= ",k,"rep= ", r, "\n")
    }
    
    methods_coef = list(OLS, ridge, LASSO, ridge_kf, LASSO_kf, ALASSO1, ALASSO2, ASLOPE)
    methods_names = c("OLS", "ridge", "LASSO", "ridge_kf", "LASSO_kf", "ALASSO1", "ALASSO2", "ASLOPE")
    methods_stats = matrix(-1, length(methods_coef), 5)
    colnames(methods_stats) = c("MSE_beta", "MSE_mu", "FWER", "FDR", "power")
    rownames(methods_stats) = methods_names
    
    for (i in 1:length(methods_coef)) {
      coeff = methods_coef[[i]]
      methods_stats[i,1] = mean( apply(coeff, 1, function(beta_hat)norm2(beta_hat, betas)) ) #mse_beta
      methods_stats[i,2] = mean( apply(coeff, 1, function(beta_hat)norm2(X%*%beta_hat,X%*%betas)) ) #mse_mu
      
      if(!(methods_names[i] %in% c("OLS", "ridge") )){
        methods_stats[i,3] = mean( apply(coeff, 1, function(beta_hat)ifFalseDiscovery(beta_hat, k, p) )  ) #fwer
        methods_stats[i,4] = mean( apply(coeff, 1, function(beta_hat)fdp(beta_hat, k, p) )  ) #fdr
        methods_stats[i,5] = mean( apply(coeff, 1, function(beta_hat)(sum(beta_hat[1:k]!=0)/k)  ) ) #power
      }
    }

    results_list[[j]] = methods_stats
    save(list=c("methods_stats"), file=paste0("SL_lab3_ex1_nonzero",k,".RData"))
    printTable(methods_stats, paste("Results for k=",k), paste0("SL_lab3_ex1_nonzero",k))
  }
  
  names(results_list) = paste(nonzero)
  return(results_list)
}

#Zad 2
ex1(correlated = TRUE, rep=10)




####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
library(xtable)
printTable <- function(data, cap, path) {
  print(xtable(data, auto = TRUE, caption = cap, digits=3), 
        file=paste(path, ".tex",sep=""), table.placement = "H", 
        floating.environment = "table",
        include.rowname = TRUE)
}
