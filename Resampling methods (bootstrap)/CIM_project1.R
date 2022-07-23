#Project 1
set.seed(297759)
library("DAAG")
data(nassCDS)
names(nassCDS)
nassCDS = na.omit(nassCDS)
attach(na.omit(nassCDS))

#Converting factor to logical
levels(dead) = c(FALSE,TRUE)
dead = as.logical(dead)
levels(airbag) = c(FALSE,TRUE)
airbag = as.logical(airbag)

###########################################################################
#########################BOOTSTRAP#########################################

#BOOTSTRAP ALGORITHM FOR A VECTOR
resample_vector_nonparam = function(X, n=length(X)) sample(X, size = n, replace = TRUE)

resample_vector_param = function(X, rdist, n=length(X)) rdist(n)

bootstrap_vector = function(B=1000, X, theta_est, param=FALSE, rdist){
  if(param)
    X_boot = sapply(1:B, function(n)resample_vector_param(X, rdist, length(X)))
  else
    X_boot = sapply(1:B, function(n)resample_vector_nonparam(X, length(X)))
  
  theta_hat = apply(X_boot, 2, theta_est)
  return(theta_hat)
}

#BOOTSTRAP ALGORITHM FOR A DATAFRAME
resample_dataframe_nonparam = function(data, n=nrow(data)){
  id_boot = sample(1:n, size = n, replace = TRUE)
  return(data[id_boot,])
}

resample_dataframe_param = function(data, rdist_list, cols=1, n=nrow(data)){
  for (col in cols)
    data[,col] = rdist_list[[col]](n)
  
  return(data)
}

bootstrap_dataframe = function(B=1000, data, theta_est, param=FALSE, rdist_list, cols=1){
  if(param)
    data_boot = lapply(1:B, function(n)resample_dataframe_param(data, rdist_list, cols, nrow(data)))
  else
    data_boot = lapply(1:B, function(n)resample_dataframe_nonparam(data, nrow(data)))
  
  theta_hat = sapply(data_boot, theta_est)
  return(theta_hat)
}

#MONTE CARLO P-VALUE
p_value_boot = function(theta_hat, theta_obs) (sum(abs(theta_hat) >= abs(theta_obs)) + 1 ) / (length(theta_hat) + 1)

###########################################################################
#########################PROJECT 1#########################################

#Question 1
#Q1.1
#Estimating the model using the classical GLM approach
fit_binom = glm(dead~ageOFocc, family = "binomial")
fit_binom$coefficients
plot(ageOFocc, predict.glm(fit_binom, type = "response"))

#Q1.2
#Contingency table
cont = round(prop.table(table(dead, ageOFocc), margin = 2), 2) 
#Table sorted ascending by probability of death
cont[, order(cont[2,])]
#Random variable X_10 as an inverse of link function
logit = function(p) log(p/(1-p))
X_10 = function(beta_0, beta_1) (logit(0.1)-beta_0)/beta_1
#Estimator of X_10
X_10_est = function(data){
  beta = glm(data$dead~data$ageOFocc, family = "binomial")$coefficients
  return(X_10(beta[1], beta[2]))
}
#Bootstrap
X_10_hat = bootstrap_dataframe(B=1000, data.frame(dead, ageOFocc), X_10_est)
hist(X_10_hat, freq = FALSE)
abline(v = quantile(X_10_hat, c(0.025, 0.975)), col="red", lwd=2)
save(X_10_hat, file="CIM_project1_1_2_X_10.RData")

#Q1.3
#Fit the null model
fit_binom_0 = glm(dead~1, family = "binomial")
#Resampling under null hypothesis: beta_1 = 0 (fixing ageOFocc, sampling from dead)
sigmoid = function(x) exp(x) / (1 + exp(x))
#Sampling from binomial distribution with pi_j
rdist = function(n) rbinom(n=n, size=1, prob = sigmoid(fit_binom_0$coefficients))
#Estimator of betas
beta_est = function(data)glm(data[,1]~data[,2], family = "binomial")$coefficients
#bootstrap algorithm
beta = bootstrap_dataframe(B=1000, data = data.frame(dead, ageOFocc), 
                           theta_est = beta_est, param=TRUE, cols=1, rdist_list=list(rdist))


hist(beta[1,], freq = FALSE, main = "Histogram of beta_0", xlab = "beta_0")
hist(beta[2,], freq = FALSE, main = "Histogram of beta_1", xlab = "beta_1")
save(beta, file="CIM_project1_1_3_beta.RData")
#Monte Carlo p-value
p_value_boot(beta[2,], fit_binom$coefficients[2])
#low p-value, so we reject null hypothesis
mean(beta[2,])
sd(beta[2,])


#Question 2
table(airbag, dead)

#Q.2.2
oddsratio = function(X, Y){
  cont = table(X, Y)
  return( (cont[1,1]*cont[2,2]) / (cont[1,2]*cont[2,1]) )
}
oddsratio(airbag, dead)
#Standard error of logOR
SE = sqrt(1/11058 + 1/669 + 1/13825 + 1/511)
#Confidence intervals
exp(log(oddsratio(airbag, dead)) + SE * 1.96)
exp(log(oddsratio(airbag, dead)) - SE * 1.96)

#Q2.3 (lecture 1c, page 15)
#logOR_hat is asymptoticly N(logOR, SE)
log_theta_hat = rnorm(1000, mean = log(oddsratio(airbag, dead)), sd = SE)
#Bootstrap CI 
quantile(exp(log_theta_hat), probs = c(0.025, 0.975))

#Q2.4
#Nonparametric bootstrap
chi_test_est = function(data) chisq.test(data[,1], data[,2])$statistic
chi_test_hat = NULL

#resampling independetly
for (b in 1:1000) {
  dead_boot = resample_vector_nonparam(dead)
  airbag_boot = resample_vector_nonparam(airbag)
  chi_test_hat[b] = chi_test_est(data.frame(dead_boot, airbag_boot))
}

hist(chi_test_hat, freq = FALSE, breaks = 20)
x = seq(min(chi_test_hat), max(chi_test_hat), by=0.01)
lines(x, dchisq(x, df=1))

save(chi_test_hat, file="CIM_project1_2_4_chi_test_hat.RData")

#testing independence
p_value_boot(chi_test_hat, chisq.test(dead, airbag)$statistic)

#Question 3
#Q3.1
MSE = function(theta_boot, theta_obs) var(theta_boot) + (mean(theta_boot)-theta_obs)^2
#MSE of mean
mean_boot = bootstrap_vector(B=1000, X=weight, theta_est = mean)
MSE(mean_boot, mean(weight))
#MSE of median
median_boot = bootstrap_vector(B=1000, X=weight, theta_est = median)
MSE(median_boot, median(weight))
#MSE of trimmed mean
t_mean = function(x) mean(x, trim = 0.1)
t_mean_boot = bootstrap_vector(B=1000, X=weight, theta_est = t_mean)
MSE(t_mean_boot, t_mean(weight))
#MSE of mid range
midrange = function(x) (min(x) + max(x))/2
midrange_boot = bootstrap_vector(B=1000, X=weight, theta_est = midrange)
MSE(midrange_boot, midrange(weight))

#Q3.2
hist(median_boot, freq = FALSE, breaks=30)
abline(v=quantile(median_boot, probs = c(0.025, 0.975)), col="red", lwd=2)

#Q3.3
jackknife = function(X, theta_est){
  theta_hat = NULL
  for (i in 1:length(X)) {
    theta_hat[i] = theta_est(X[-i])
  }
  return(theta_hat)
}
#MSE needs to include inflation factor
MSE_jackknife = function(theta_boot, theta_obs){
  n = length(theta_boot)
  (n-1)^2 / n * var(theta_boot) +((n-1)*(mean(theta_boot)-theta_obs))^2
}
#MSE of mean
mean_jack = jackknife(weight, mean)
MSE_jackknife(mean_jack, mean(weight))
#MSE of median
median_jack = jackknife(weight, median)
MSE_jackknife(median_jack, median(weight))
#MSE of trimmed mean
t_mean_jack = jackknife(weight, t_mean)
MSE_jackknife(t_mean_jack, t_mean(weight))
#MSE of midrange
midrange_jack = jackknife(weight, midrange)
MSE_jackknife(midrange_jack, midrange(weight))


