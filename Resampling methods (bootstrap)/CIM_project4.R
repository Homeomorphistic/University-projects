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
resample_dataframe_nonparam = function(data, cols=1:ncol(data), n=nrow(data), replacement=TRUE){
  id_boot = sample(1:n, size = n, replace = replacement)
  data[,cols] = data[id_boot, cols]
  return(data)
}

resample_dataframe_param = function(data, rdist_list, cols=1, n=nrow(data)){
  for (col in cols)
    data[,col] = rdist_list[[col]](n)
  
  return(data)
}

bootstrap_dataframe = function(B=1000, data, theta_est, cols=1:ncol(data), param=FALSE, rdist_list, replacement=TRUE){
  if(param)
    data_boot = lapply(1:B, function(n)resample_dataframe_param(data, rdist_list, cols, nrow(data)))
  else
    data_boot = lapply(1:B, function(n)resample_dataframe_nonparam(data, cols, nrow(data), replacement))
  
  theta_hat = sapply(data_boot, theta_est)
  return(theta_hat)
}

#MONTE CARLO P-VALUE
p_value_boot = function(theta_hat, theta_obs) 
  (sum( abs(theta_hat) >= abs(theta_obs)) + 1) / (length(theta_hat) + 1)

###########################################################################
################BOOTSTRAP CONFIDENCE INTERVALS#############################
improved_normal_CI = function(theta_boot, theta_obs, alpha=.05){
  bias = mean(theta_boot) - theta_obs
  se = sd(theta_boot)
  return( theta_obs - bias + se * qnorm(c(alpha/2, 1-alpha/2)) )
}

basic_bootstrap_CI = function(theta_boot, theta_obs, alpha=.05){
  return( 2*theta_obs - quantile(theta_boot, probs = c(1-alpha/2, alpha/2)) )
}

percentile_CI = function(theta_boot, alpha=.05){
  return( quantile(theta_boot, probs = c(alpha/2, 1-alpha/2)) )
}

studentized_CI = function(theta_boot, theta_obs, se_theta_obs, alpha=.05){ #WRONG
  t_boot = (theta_boot - theta_obs) / se_theta_obs #not this se, needed from each bootstrap replicate
  return( theta_obs + se_theta_obs *  quantile(t_boot, probs = c(alpha/2, 1-alpha/2)) )
}

###########################################################################
#########################PROJECT 1#########################################
#Preparing data
set.seed(297759)
library(stevedata)
data(gss_wages)
names(gss_wages)
gss_wages = na.omit(gss_wages)
attach(na.omit(gss_wages))

#Question 1
#Q1.1
fit = lm(realrinc~age+childs+gender+maritalcat)
confint(fit)
#Q1.2
#nonparametric
beta_est = function(data){ 
  lm(data$realrinc~data$age+data$childs+data$gender+data$maritalcat)$coefficients
}
beta_hat_nonparam = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                               theta_est = beta_est)

save(beta_hat_nonparam, file="CIM_project4_1_2_beta_hat_nonparam.RData")

#Distribution
beta_1_hat = beta_hat_nonparam[2,]
sd(beta_1_hat)
hist(beta_1_hat, freq = FALSE, main = "Histogram of beta_1, nonparametric", xlab = "beta_1", breaks=20)
#CI's
beta_1_obs = fit$coefficients[2]
se_beta_1_obs = sqrt(vcov(fit)[2,2])
improved_normal_CI(beta_1_hat, beta_1_obs) #red
basic_bootstrap_CI(beta_1_hat, beta_1_obs) #blue
percentile_CI(beta_1_hat) #green
studentized_CI(beta_1_hat, beta_1_obs, se_beta_1_obs) #purple == green?
legend("topright", legend=c("Normal", "Basic", "Precentile"), col = c("red", "blue", "green"), lty=1)

#Parametric
realrinc_dist = function(n) rnorm(n, predict(fit), summary(fit)$sigma)
beta_hat_param = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                     theta_est = beta_est, param = TRUE, cols = 1, rdist_list = list(realrinc_dist))
save(beta_hat_param, file="CIM_project4_1_2_beta_hat_param.RData")
#Distribution
beta_1_hat = beta_hat_param[2,]
sd(beta_1_hat)
hist(beta_1_hat, freq = FALSE, main = "Histogram of beta_1, parametric", xlab = "beta_1", breaks=20)
#CI's
improved_normal_CI(beta_1_hat, beta_1_obs) #red
basic_bootstrap_CI(beta_1_hat, beta_1_obs) #blue
percentile_CI(beta_1_hat) #green
studentized_CI(beta_1_hat, beta_1_obs, se_beta_1_obs) #purple == green?
legend("topright", legend=c("Normal", "Basic", "Precentile"), col = c("red", "blue", "green"), lty=1)

#Q1.3
loglik_test = function(data){
  fit_saturated = lm(data$realrinc~data$age+data$childs+data$gender+data$maritalcat)
  fit_reduced = lm(data$realrinc~data$age+data$maritalcat)
  return( -2 * as.numeric(logLik(fit_reduced) - logLik(fit_saturated)) )
}

#Nonparametric
loglik_hat_nonparam = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                          theta_est = loglik_test, cols = c(1,2,5))
save(loglik_hat_nonparam, file="CIM_project4_1_3_loglik_hat_nonparam.RData")
#Monte Carlo p-value
fit_reduced = lm(realrinc~age+maritalcat)
loglik_obs = -2 * as.numeric(logLik(fit_reduced) - logLik(fit))
p_value_boot(loglik_hat_nonparam, loglik_obs) #reject H_0

#Parametric
#to obtain beta_0 + beta_1 X_1i + beta_4 X_4i, without creating matrix X with many dummy variables
means_reduced = predict(fit, newdata = data.frame(realrinc, age, childs=0, gender="Female", maritalcat))
realrinc_reduced_dist = function(n) rnorm(n, means_reduced, summary(fit)$sigma)
loglik_hat_param = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                       theta_est = loglik_test, param = TRUE, cols = 1, rdist_list = list(realrinc_reduced_dist))
save(loglik_hat_nonparam, file="CIM_project4_1_3_loglik_hat_param.RData")

#Monte Carlo p-value
p_value_boot(loglik_hat_param, loglik_obs) #reject H_0

#Semi-parametric
loglik_hat_semiparam_est = function(e){
  y = e + predict(fit, newdata = data.frame(realrinc, age, childs=0, gender="Female", maritalcat)) 
  return( loglik_test(data.frame(realrinc=y, age, childs, gender, maritalcat)) )
}
loglik_hat_semiparam = bootstrap_vector(B=1000, X = fit$residuals, theta_est = loglik_hat_semiparam_est)
save(loglik_hat_semiparam, file="CIM_project4_1_3_loglik_hat_semiparam.RData")

#Monte Carlo p-value
p_value_boot(loglik_hat_semiparam, loglik_obs) #reject H_0

#Q1.4
loglik_hat_nonparam_perm = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                          theta_est = loglik_test, cols = c(1,2,5), replacement = FALSE)
save(loglik_hat_nonparam_perm, file="CIM_project4_1_4_loglik_hat_nonparam_perm.RData")

#Monte Carlo p-value
p_value_boot(loglik_hat_nonparam_perm, loglik_obs) #reject H_0

#Question 2
#Q2.1
fit = lm(realrinc~age+childs+gender+maritalcat)
new_subject = data.frame(age=40, childs=2, gender="Female", maritalcat="Divorced")
predict(fit, newdata = new_subject, interval = "confidence")

predict_new_subject_est = function(data){
  new_fit = lm(realrinc~age+childs+gender+maritalcat, data = data)
  return(predict(new_fit, newdata = new_subject))
}

#Non-parametric
predict_new_subject_hat_nonparam = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                              theta_est = predict_new_subject_est)
save(predict_new_subject_hat_nonparam, file="CIM_project4_2_1_predict_new_subject_hat_nonparam.RData")
percentile_CI(predict_new_subject_hat_nonparam)

#Parametric 
predict_new_subject_hat_param = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                                       theta_est = predict_new_subject_est,
                                                    param = TRUE, rdist_list = list(realrinc_dist), cols = 1)
save(predict_new_subject_hat_param, file="CIM_project4_2_1_predict_new_subject_hat_param.RData")
percentile_CI(predict_new_subject_hat_param)

#Q2.2
predict_new_subject_male_est = function(data){
  new_fit = lm(realrinc~age+childs+gender+maritalcat, data = data)
  return(predict(new_fit, newdata = data.frame(age=46, childs=2, gender="Male", maritalcat="Divorced")))
}
#Non-parametric
predict_new_subject_male_hat_nonparam = bootstrap_dataframe(B=1000, data = data.frame(realrinc, age, childs, gender, maritalcat),
                                                       theta_est = predict_new_subject_male_est)
save(predict_new_subject_male_hat_nonparam, file="CIM_project4_2_2_predict_new_subject_male_hat_nonparam.RData")
percentile_CI(predict_new_subject_male_hat_nonparam)

hist(predict_new_subject_hat_nonparam, freq = FALSE)
hist(predict_new_subject_male_hat_nonparam, freq = FALSE)

#Q2.3
predict_female_semiparam_est = function(e){
  y = e + predict(fit) 
  return( predict_new_subject_est(data.frame(realrinc=y, age, childs, gender, maritalcat)) )
}
predict_male_semiparam_est = function(e){
  y = e + predict(fit) 
  return( predict_new_subject_male_est(data.frame(realrinc=y, age, childs, gender, maritalcat)) )
}

predict_new_subject_hat_semiparam = bootstrap_vector(B=1000, X = fit$residuals, theta_est = predict_female_semiparam_est)
predict_new_subject_male_hat_semiparam = bootstrap_vector(B=1000, X = fit$residuals, theta_est = predict_male_semiparam_est)

save(predict_new_subject_hat_semiparam , file="CIM_project4_2_3_predict_new_subject_hat_semiparam.RData")
save(predict_new_subject_male_hat_semiparam , file="CIM_project4_2_3_predict_new_subject_male_hat_semiparam.RData")

sd(predict_new_subject_hat_semiparam)
sd(predict_new_subject_male_hat_semiparam)

#Question 3
table(childs, gender) #with na.omit less than in .pdf
gss_wages$childs3 = (childs < 3)
attach(gss_wages)

#Q3.1
cont_tab = table(gender, childs3)
n = sum(cont_tab[1,])
m = sum(cont_tab[2,])
pi_obs = prop.table(cont_tab, margin = 1)[,2]
pi_tot_obs = (13534 + 13852) / sum(cont_tab)
pi_obs[1] - pi_obs[2] + qnorm(c(0.025, 0.975)) * sqrt( pi_tot_obs * (1-pi_tot_obs) * (1/n + 1/m )  )

#Q3.2
#Parametric
t = NULL
for (i in 1:1000) {
  pi_hat1 = sum(rbinom(n, size = 1, prob = pi_obs[1])) / n
  pi_hat2 = sum(rbinom(m, size = 1, prob = pi_obs[2])) / m
  pi_tot = (n*pi_hat1 + m*pi_hat2) / (n+m)
  t[i] = (pi_hat1 - pi_hat2) #/ sqrt( pi_tot * (1-pi_tot) * (1/n + 1/m ) )
}
percentile_CI(t)

#Q3.3
#Non-parametric under null
z = c(rep(1, 13534), rep(0, n-13534), rep(1, 13852), rep(0, m-13852))
t = NULL
for (i in 1:1000) {
  z_boot = sample(z, size = n+m, replace = TRUE)
  pi_hat1 = sum(z_boot[1:n]) / n
  pi_hat2 = sum(z_boot[(n+1):(n+m)]) / m
  pi_tot = (n*pi_hat1 + m*pi_hat2) / (n+m)
  t[i] = (pi_hat1 - pi_hat2) / sqrt( pi_tot * (1-pi_tot) * (1/n + 1/m ) )
}
p_value_boot(t, (pi_obs[1] - pi_obs[2]) / sqrt( pi_tot_obs * (1-pi_tot_obs) * (1/n + 1/m )) )
#high p-value, accept H_0

#Q3.4
hist(t, freq = FALSE)
x = seq(-3,3,by=0.01)
lines(x, dnorm(x))


#Question 4
#Q4.1
t.test(prestg10)
n = length(prestg10)
(n-1)*var(prestg10) / qchisq(0.975, n-1)
var(prestg10)
(n-1)*var(prestg10) / qchisq(0.025, n-1)

#Q4.2
prestg10_dist = function(n) rnorm(n, mean(prestg10), sd(prestg10))
mean_and_var_est = function(X) return( c(mean(X), var(X))  )
#Parametric
mean_and_var_hat_param = bootstrap_vector(B=1000, prestg10, theta_est = mean_and_var_est, param = TRUE, rdist = prestg10_dist)
save(mean_and_var_hat_param , file="CIM_project4_4_2_mean_and_var_hat_param.RData")
percentile_CI(mean_and_var_hat_param[1,])
percentile_CI(mean_and_var_hat_param[2,])

#Q4.3
mu_0 = 43.14
t.test(prestg10, alternative = "less", mu = mu_0)
t_obs = t.test(prestg10, alternative = "less", mu = mu_0)$statistic

#Q4.4
t_est = function(X) t.test(X, alternative = "less", mu = mu_0)$statistic
#Non-parametric
t_hat_nonparam = bootstrap_vector(B=1000, prestg10 - mean(prestg10) + mu_0, theta_est = t_est)
save(t_hat_nonparam , file="CIM_project4_4_4_t_hat_nonparam.RData")
(1 + sum(t_hat_nonparam < t_obs)) / (1001)
