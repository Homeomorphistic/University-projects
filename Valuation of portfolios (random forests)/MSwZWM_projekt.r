load("D:/Uniwersytet Wrocławski/Semestr VII/Modelowanie statystyczne w zarządzaniu wierzytelnościami masowymi/Laboratorium/KrukUWr2020.RData")
library(data.table)
library(tree)
library(rpart)
library(randomForest)
library(ModelMetrics)
Cases = data.table(cases)
####################################################################################################################################
#PRZYGOTOOWYWANIE DANYCH
set.seed(2)

#Modyfikacja danych na dummy variable
Cases[, Product:=ifelse(Product=="Cash loan", 1, 0)]
Cases[, Gender:=ifelse(Gender=="MALE", 1, 0)]

#IMPUTACJA (kod z laboratoriów)
variables = c(         "LoanAmount",
                       "TOA",
                       "Principal",
                       "Interest",
                       "Other",
                       "Product",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "ExternalAgency",
                       "Bailiff",
                       "ClosedExecution",
                       "Land",
                       "PopulationInCity",
                       "Age",
                       "Gender",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate",
                       "GDPPerCapita",
                       "MeanSalary"
)

#Imputacja ekspercka
#Uzupełnienie Other przez TOA, Principal i Interest
Cases[is.na(Other), Other:=TOA-Principal-Interest,]

#Imputacja medianą
nullCounts <- lapply(Cases[,.SD,.SDcols=variables], function(x) sum(is.na(x)))
variables <- c(        "LoanAmount",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "PopulationInCity",
                       "Age",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate"#,
                       #"GDPPerCapita",
                       #"MeanSalary"
)

for (variable in variables) {      
  if (eval(parse(text=paste("nullCounts$",variable,sep=""))) > 0) {
    med <- eval(parse(text=paste("median(Cases[,",variable,"],na.rm=TRUE)",sep="")))
    eval(parse(text=paste("Cases[is.na(",variable,"), ",variable,":=med]",sep="")))
  }           
}

#Imputacja danych 0/1
Cases[is.na(Gender),Gender:= ifelse(runif(nullCounts$Gender,0,1)<Cases[,mean(Gender,na.rm=TRUE)],1L,0L)]
Cases[is.na(Bailiff),Bailiff:= ifelse(runif(nullCounts$Bailiff,0,1)<Cases[,mean(Bailiff,na.rm=TRUE)],1L,0L)]
Cases[is.na(ExternalAgency),ExternalAgency:= ifelse(runif(nullCounts$ExternalAgency,0,1)<Cases[,mean(ExternalAgency,na.rm=TRUE)],1L,0L)]

Cases[is.na(ClosedExecution) & Bailiff==0, ClosedExecution:= 0L]
Cases[is.na(ClosedExecution), ClosedExecution:= ifelse(runif(dim(Cases[is.na(ClosedExecution),])[1],0,1)<Cases[,mean(ClosedExecution,na.rm=TRUE)],1L,0L)]

#Imputacja rozkładem dla Land oraz uzupełnianie GDPPerCapita i MeanSalary
prob = unlist((Cases[!is.na(Land), .N, by=Land][order(Land)] / Cases[!is.na(Land),.N])[,2])
Cases[is.na(Land), Land:=sample(1:37, Cases[is.na(Land),.N], prob, replace=TRUE)]
land_stats = Cases[!is.na(MeanSalary), .N, by=.(Land,GDP=GDPPerCapita, MS=MeanSalary)]
Cases = Cases[land_stats, on=.(Land)]
Cases[is.na(MeanSalary), ':='(MeanSalary=MS, GDPPerCapita=GDP),]
Cases[, ':='(MS=NULL, GDP=NULL)]

#OUTLIERS
#Usuwanie wartości odstających
Proportion = 0.001
Cases <- Cases[LoanAmount<quantile(Cases[,LoanAmount], probs=1-Proportion, na.rm=TRUE),]
Cases <- Cases[DPD<quantile(Cases[,DPD], probs=1-Proportion, na.rm=TRUE),]
Cases <- Cases[LastPaymentAmount<quantile(Cases[,LastPaymentAmount], probs=1-Proportion, na.rm=TRUE),]

#DODATKOWE ZMIENNE I PODZIAŁ NA ZBIORY TRENINGOWE, TESTOWE
#Dodanie zmiennej związanej z płatnością wykonaną w ciągu 6 miesięcy
tmp = events[Month<=6, .(Payment6M=sum(PaymentAmount, na.rm = TRUE)), by=CaseId]
Cases = Cases[tmp, on=.(CaseId), nomatch=0]
Cases[, IfPayment6M := ifelse(Payment6M>0, 1, 0),]
Cases[, IfPayment6M := as.factor(IfPayment6M),]

#Stworzenie sztucznych portfeli i SR
##Cases[, Portfel:=sample(1:500, .N, replace = TRUE),]
##Cases[, SR := Payment6M/TOA]
##Cases[, SR_port:=sum(Payment6M)/sum(TOA), by=Portfel]

#Podział na zbiory treningowe, walidacyjne oraz testowe
Cases[, Set:=sample(1:3, .N, replace = TRUE, prob = c(.5,.25,.25)),]
CasesTrn = Cases[Set==1,,]
CasesVal = Cases[Set==2,,]
CasesTst = Cases[Set==3,,]
CasesTrn[, SR := Payment6M/TOA]##
CasesTrn <- CasesTrn[SR<quantile(CasesTrn[,SR], probs=1-Proportion),]##
CasesVal[, Portfel:=sample(1:200, .N, replace = TRUE)]##
CasesVal[,.N,by=Portfel]##


####################################################################################################################################
#FUNKCJE POMOCNICZE
library(ROCR)
rocplot <- function(pred, truth, name) {
  predob <- prediction(pred, truth)
  perf <- performance(predob, "tpr", "fpr")
  gini <- 2*attributes(performance(predob, "auc"))$y.values[[1]] - 1
  
  plot(perf, main=paste0(name,", Gini: ", round(gini, 2)))
  
  invisible(round(gini,4))
}

accuracy <- function(pred, truth) {
  confMat <- table(truth, pred)
  sum(diag(confMat))/sum(confMat)
}
####################################################################################################################################
#DRZEWA BINARNE
variables = c(         "TOA",
                       "Product",
                       "D_ContractDateToImportDate",
                       "DPD",
                       "ExternalAgency",
                       "Bailiff",
                       "ClosedExecution",
                       "PopulationInCity",
                       "Age",
                       "Gender",
                       "LastPaymentAmount",
                       "M_LastPaymentToImportDate",
                       "MeanSalary"
)
formula = as.formula(paste("IfPayment6M",paste(variables, collapse = "+"), sep = "~"))
####################################################################################################################################
#DRZEWO
#tree
tTree <- tree(formula, data=CasesTrn, 
              split="gini", mincut=500, minsize=1000, mindev=1e-12)
accuracy(CasesTst$IfPayment6M, predict(tTree, CasesTst, type="class"))

#rpart
rTree <- rpart(formula, data=CasesTrn, 
               method="class", minsplit=1000, minbucket=500, cp=1e-12,
               parms=list(split="gini"))

plot(rTree)
text(rTree, pretty=0, cex=0.7)

g_rTree <- rocplot(predict(rTree, CasesTst, type="prob")[, 2], CasesTst$IfPayment6M, "rTree")
accuracy(CasesTst$IfPayment6M, predict(rTree, CasesTst, type="class"))

#Przycinanie drzewa
cp <- printcp(rTree)
cp <- data.table(cp)
plotcp(rTree)

rTreePruned <- prune(rTree, cp=cp[which.min(cp$xerror), ]$CP)

plot(rTreePruned)
text(rTreePruned, pretty=0, cex=0.7)

g_rTreePruned <- rocplot(predict(rTreePruned, CasesTst, type="prob")[, 2], CasesTst$IfPayment6M)
accuracy(CasesTst$IfPayment6M, predict(rTreePruned, CasesTst, type="class"))

####################################################################################################################################
#LASY
ct=CasesTrn[, .N/CasesTrn[, .N], by=IfPayment6M][order(IfPayment6M)]$V1[2]

create_forests <- function(N=c(500,1000), M=c(13,10,5,3), n_node=c(1,5,20,100,200), cutoff=c(.5, ct)) {
  results = data.frame()
  for (n in N) {
    for (m in M) {
      for (nn in n_node) {
        for (ct in cutoff) {
          time=system.time(
            forest <- randomForest(formula, data=CasesTrn,
                                      mtry=m, ntree=n, nodesize=nn,
                                      cutoff=c(ct, 1-ct),
                                      importance=TRUE, keep.forest=TRUE)
          )
          acc = round(accuracy(CasesVal$IfPayment6M, predict(forest, CasesVal, type="class")), digits=4)
          gini = rocplot(predict(forest, CasesVal, type="prob")[, 2], CasesVal$IfPayment6M,
                         paste("n=",n,"m=",m,"nn=",nn,"ct=",round(100*ct),"%") )
          mean_size = round(mean(treesize(forest, terminal=TRUE)), digits=3)
          imp = importance(forest)
          mda_ind = which.max(imp[,3])
          mdg_ind = which.max(imp[,4])
          var_names = rownames(imp)[c(mda_ind, mdg_ind)]
          cat(n,m,nn,ct,acc,gini,time[1],"\n")
          results = rbind(results, c(n,m,nn,round(ct,2),mean_size,acc,gini,var_names,round(time[1],digits=1)))
          save(list=c("forest"), file=paste0("forest_n=",n,"m=",m,"nn=",nn,"ct=",round(100*ct),"%",".RData"))
        }
      }
    }
  }
  colnames(results) = c("ntree", "mtry", "nodesize", "cutoff", "m. tr. size", 
                        "accuracy", "gini ind",  "mda var", "mdg var", "time")
  save(list=c("results"), file="results.RData")
  return(results)
}

load(file="results.RData")
results$cutoff = round(as.numeric(results$cutoff),2)
results[order(results$accuracy),]
results[order(results$`gini ind`),]

val_forests <- function(N=c(500,1000), M=c(13,10,5,3), n_node=c(1,5,20,100,200), cutoff=c(.5, ct)) {
  validation = data.frame()
  for (n in N) {
    for (m in M) {
      for (nn in n_node) {
        for (ct in cutoff) {
          load(paste0("forest_n=",n,"m=",m,"nn=",nn,"ct=",round(100*ct),"%",".RData"))
          CasesTrn[, pred:= predict(forest, CasesTrn, type="prob")[, 2]]
          CasesVal[, pred:= predict(forest, CasesVal, type="prob")[, 2]]
          breaks_trn = quantile(CasesTrn$pred, probs=seq(0, 1, length.out=11))
          logl = logLoss(CasesVal$IfPayment6M, CasesVal$pred)
          
          CasesTrn[, Band:= cut(pred, breaks=breaks_trn, labels=1:10,include.lowest=TRUE)]
          ##CasesTrn[, mean_SR:=mean(SR), by=Band]
          ##tmp = CasesTrn[, .N, by=.(Band, mean_SR)]
          
          CasesVal[, Band:= cut(pred, breaks=breaks_trn,labels=1:10,include.lowest=TRUE)]
          ##CasesVal = CasesVal[tmp, on=.(Band), nomatch=0]
          ##CasesVal[, mean_SR_port:=mean(mean_SR), by=Portfel]
          
          ##CasesVal[,ABS1:=abs(mean_SR_port-SR_port)/SR_port,by=.(Portfel,mean_SR_port,SR_port)]
          ##CasesVal[,ABS2:=abs(mean_SR_port-SR_port)/mean_SR_port,by=.(Portfel,mean_SR_port,SR_port)]
          ##CasesVal[, `:=`(mean_SR=NULL,N=NULL)]
          
          Payments_Group <- tapply(CasesTrn[,Payment6M],CasesTrn[,Band],sum)##
          TOA_Group <- tapply(CasesTrn[,TOA],CasesTrn[,Band],sum)##
          SR_Group <- Payments_Group/TOA_Group##
          Forecast <- data.table(cbind(Band=row.names(SR_Group),SR=SR_Group*1.0))##
          CasesValP <- CasesVal[Forecast, on = "Band"]##
          CasesValP[, Forecast := as.numeric(i.SR)*as.numeric(TOA)]##
          dev_w <- mean(na.omit(CasesValP[,.(dev_w=(abs(sum(Payment6M)-sum(Forecast)))/sum(Payment6M)),by=Portfel][,dev_w]))##
          dev_p <- mean(na.omit(CasesValP[,.(dev_p=(abs(sum(Payment6M)-sum(Forecast)))/sum(Forecast)),by=Portfel][,dev_p]))##
          
          ##cat(c(n,m,nn,round(ct,2),logl,mean(CasesVal$ABS1),mean(CasesVal$ABS2)), "\n")
          cat(c(n,m,nn,round(ct,2),logl,dev_w,dev_p), "\n")
          ##validation = rbind(validation, c(n,m,nn,round(ct,2),logl,mean(CasesVal$ABS1),mean(CasesVal$ABS2)))
          validation = rbind(validation, c(n,m,nn,round(ct,2),logl,dev_w,dev_p))
          
        }
      }
    }
  }
  colnames(validation) = c("ntree", "mtry", "nodesize", "cutoff", "logloss","|w-p|/w","|w-p|/p")
  save(list=c("validation"), file="validation.RData")
  return(validation)
}

load("validation.Rdata")
load("forest_n=1000m=3nn=200ct=50%.RData")
head(cbind(results[,1:7], validation[,5:7])[order(validation$`|w-p|/p`),])
head(cbind(results[,1:7], validation[,5:7])[order(validation$logloss),])

importance(bagForest)
varImpPlot(bagForest)

# rozmiar drzew
boxplot(treesize(bagForest, terminal=TRUE))

# pojedyncze drzewo
getTree(bagForest, k=6, labelVar=TRUE)

head(bagForest$votes)

# partial plot (pokazuje kierunek działania zmiennej)
# weryfikuje nasze intuicje/reguły biznesowe
partialPlot(bagForest, CasesTst, DPD, "0")
partialPlot(bagForest, CasesTst, DPD, "1")



####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#NORMALIZACJA DANYCH
variables = c("DPD", 
              "Land" , 
              "LastPaymentAmount", 
              "ExternalAgency", 
              "PopulationInCity", 
              "M_LastPaymentToImportDate",
              "TOA",
              "Product",#
              "Bailiff",#
              "Age",
              "GDPPerCapita",
              "D_ContractDateToImportDate",
              "ClosedExecution",#
              "Gender",#
              "MeanSalary")

for (v in variables) { 
  expr <- paste0("cases[, ", v, ":=(", v, " - min(cases$", v, 
                 "))/(max(cases$", v, ") - min(cases$", v, "))]")
  eval(parse(text=expr))
} 
####################################################################################################################################
#SIECI NEURONOWE neuralnet
frmCl <- as.formula(paste("IfPayment6M ~ ", paste(variables, collapse = "+")))

grid = as.matrix(expand.grid(1:4, 1:3, 1:2))
neurons = unname(grid[grid[,1]>=grid[,2] & grid[,2] >= grid[,3], ])
results = data.frame()

for (i in 1:nrow(neurons)) {
  nn = neurons[i,]
  t = system.time(
    nnetFitCl <- neuralnet(frmCl, data=casesTrn, hidden=nn, stepmax=1e+5, threshold = 0.1, rep=3)
  )
  nnetFitClPred <- compute(nnetFitCl, casesTst[, .SD, .SDcols=variables])
  
  confMat <- table(truth=casesTst$IfPayment6M,
                   predict=round(nnetFitClPred$net.result[, 1], 0))
  
  results[1,i] <- sum(diag(confMat))/sum(confMat)
  results[2,i] <- t[1]
  colnames(results)[i] <- paste(nn, collapse="")
  
  cat("neurons=", nn, "time=", t[1], "\n")
  
  save(list=c("nnetFitCl"), file=paste("nnet",paste(nn, collapse=""),".RData", sep=""))
}
rownames(results) = c("acc", "time")
save(list=c("results"), file=paste("results",".RData", sep=""))

plot(nnetFitCl)

load("nnet6.RData")
                         