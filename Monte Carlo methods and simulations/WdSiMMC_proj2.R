library(iterators)
#Machines (as iterators)
M1 = iter(function() sum(rexp(2,10)))
M2 = iter(function() ifelse(rbinom(1,1,.8), rexp(1,9), rexp(1,3)))

enqueue <- function(queue, element) c(queue, element)
dequeue <- function(queue) queue[-1]

#Simulatinig production
production <- function(work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  time = 0; i=1; product = 1; on_m1 = 1; on_m2 = 0;
  buffer = NULL; time_1 = NULL; time_2 = NULL;
  history = data.frame()
  
  t1 = nextElem(machine1); t2 = 0;
  history = rbind(history, c(round(time,3), round(t1,3), round(t2,3), on_m1, paste(buffer, collapse = ","), on_m2))
  
  while (time < work_time & on_m2 <= n) {
    waiting = length(buffer)
    
    if(waiting == 0){ #IS BUFFER EMPTY?
      time = time + t1
      time_1[i] = t1 #producing product
      buffer = enqueue(buffer, on_m1) #enqueuing product
      on_m1 = on_m1 + 1 #next product on machine
      
      if(t2 - t1 < 0){ #2ND MACHINE WORKING PARALLEL
        time_2[i] = t2 - t1 #done and waiting for product
        on_m2 = buffer[1]
        buffer = dequeue(buffer) #removing from buffer right after enqueuing
        t2 = nextElem(machine2)
      }
      else{
        t2 = t2 - t1 #working on product on 2nd machine
        time_2[i] = t1
      }
      
      t1 = nextElem(machine1)
    }
      else if(waiting == k){ #IS BUFFER FULL?
        time = time + t2
        time_2[i] = t2 #working on product
        on_m2 = buffer[1]
        buffer = dequeue(buffer)
        
        if(t1 - t2 < 0){ #1ST MACHINE WORKING PARALLEL
          time_1[i] = t1 - t2 #done and waiting for space
          buffer = enqueue(buffer, on_m1) #enqueuing right after removing
          on_m1 = on_m1 + 1 #next product on machine
          t1 = nextElem(machine1)
        } 
        else{
          t1 = t1 - t2 #producing the product on 1st machine
          time_1[i] = t2
        }
        
        t2 = nextElem(machine2)
      }
        else if(t1<t2){ #WHO'S FASTER?
          time = time + t1
          t2 = t2 - t1
          time_1[i] = t1
          time_2[i] = t1
          buffer = enqueue(buffer, on_m1)
          on_m1 = on_m1 + 1
          t1 = nextElem(machine1)
        }
        else{
          time = time + t2
          t1 = t1 - t2
          time_1[i] = t2
          time_2[i] = t2
          on_m2 = buffer[1]
          buffer = dequeue(buffer)
          t2 = nextElem(machine2)
        }
    
    
    history = rbind(history, c(round(time,3), round(t1,3), round(t2,3), on_m1, paste(buffer, collapse = ","), on_m2))
    i = i+1
  }
  colnames(history) = c("time", "t1", "t2", "on_m1", "buffer", "on_m2")
  return(list(history=history, times=cbind(time_1, time_2)))
}

#Fraction of waiting time
f_1 <- function(rep=100, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  f = NULL
  for (i in 1:rep) {
    time1 = production(work_time, n, k, machine1, machine2)$times[,1]
    f[i] = sum(abs(time1[time1<0])) / work_time#/ length(time1)
  }
  return(f)
}

#Estimating variance of f_1
sigma_f_1 <- function(rep=1000, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  return(var(f_1(rep, work_time, n, k, machine1, machine2)))
}

#How many replications are needed? k=1,5,10 -> R=400,100,20
R <- function(b=.01, alpha=.1, rep=10000, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  return(ceiling(sigma_f_1(rep, work_time, n, k, machine1, machine2) * qnorm(1-alpha/2)^2 / b^2))
}

reps = NULL
for(k in 19:20){
  reps[k] = R(k=k, b=.00001)
  cat("k= ",k, "rep= ",reps[k],"\n")
}

#1st task 
#Cost function
cost <- function(C, k, Ef_1){ 
  return(C * Ef_1 + k)
}

#Minimal k
k_min <- function(C, Ef_1, K=1:20) {
  costs = cost(C, K, Ef_1)
  return(which.min(costs))
}

#Analysing k(C) 
reps_trim = ifelse(reps>10^4, 10^4, reps)
Ef_1 = NULL
for (k in 1:20) {
  Ef_1[k] = mean(f_1(rep = reps_trim[k], k=k))
  cat("k= ", k, "rep=",rep[k],"Ef_1=", Ef_1[k],"\n")
}
analyze_k_min <- function(costs=c(10^(0:2), c(2,5,7)*10^2, 10^(3:6)), Ef_1) {
  k_mins = NULL;
  for (i in 1:length(costs)) {
    C = costs[i]
    k_mins[i] = k_min(C,Ef_1)
    cat("C= ", C, "k_min= ", k_mins[i], "\n")
  }
  plot(costs, k_mins, main = "k(C)", log = "x", type = "s", xlab = "C", ylab = "k_min")
  return(cbind(costs, k_mins))
}
test = analyze_k_min(Ef_1 = Ef_1)
#2nd task
#Estimating variance of f_1 == 0
sigma_f_1_p0 <- function(rep=1000, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  return(var(f_1(rep, work_time, n, k, machine1, machine2) == 0))
}

#How many replications are needed? k=1,5,10 -> R=400,100,20
R_p0 <- function(b=.01, alpha=.1, rep=10000, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  return(ceiling(sigma_f_1(rep, work_time, n, k, machine1, machine2) * qnorm(1-alpha/2)^2 / b^2))
}

Ef_1_p0 = NULL
for (k in 1:20) {
  Ef_1_p0[k] = estimate_f_1_p0(rep=1000, k=k)
  cat("k= ", k,"Ef_1_p0=", Ef_1_p0[k],"\n")
}

reps_p0 = NULL
for(k in 1:20){
  reps_p0[k] = R_p0(k=k)
  cat("k= ",k, "rep= ",reps_p0[k],"\n")
}

#Probability  that  f_1 == 0
estimate_f_1_p0 <- function(rep=100, work_time=8, n=120, k=5, machine1=M1, machine2=M2) {
  return(mean(f_1(rep, work_time, n, k, machine1, machine2) == 0))
}

#Finding minimal k so that P(f_1==0) >= 0.9
analyze_k_min_p0 <- function(reps=reps_p0, work_time=8, n=120, machine1=M1, machine2=M2) {
  est = NULL
  for (k in 1:20) {
    est[k] = estimate_f_1_p0(reps[k], work_time=8, n=120, k=k, machine1=M1, machine2=M2)
    cat("est= ", est[k], "k= ", k, "\n")
  }
  #return(min(which(est>=0.9)))
  plot(1:20, est, type = "l", main = "P(f_1(k)==0)", xlab="k", ylab = "EB")
  abline(h=0.9, lty="dashed")
  return(est)
}
test_p0 = analyze_k_min_p0()


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
Ef_1 = c(0.158947 , 0.1098468 , 0.07419468, 0.05180968, 0.03848406 , 0.02744068, 0.02007242 , 0.0141119 , 0.009893351,
         0.006670777, 0.004925221, 0.003847083, 0.001528942 , 0.001759757, 0.00121926, 0.0008284395, 0.0004495798, 0.0003239812,
         0.0003376858, 0.000188241)

library(xtable)
printTable <- function(data, path, cap=NULL, lab=NULL) {
  print(xtable(data, auto = TRUE, caption = cap, label=lab, digits=0), 
        file=paste(path, ".tex",sep=""), table.placement = "!htb", 
        floating.environment = "table",
        include.rowname = TRUE)
}
printTable(test$history, "history_tab", "Historia 8 godzin pracy.", "subtab:history")
printTable(test$times, "times_tab", "Czasy pracy i czekania maszyn.", "subtab:times")

b = c(rep(.001,3), rep(.0001,5), rep(.00001,2))
rbind(11:20, b, reps[11:20])
printTable(rbind(11:20, reps_p0[11:20]), "R_B_tab2", "Liczba replikacji zm. $B$ dla $k=11, \\ldots, 220$.",  "tab:R_B_tab2")
