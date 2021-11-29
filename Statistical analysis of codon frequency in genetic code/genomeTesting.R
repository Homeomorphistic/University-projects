#Biblioteka Biostrings
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()

BiocManager::install("Biostrings")
library(Biostrings)

BiocManager::install("biomaRt")
library(biomaRt)
library(ape)
##############################################
#Kod genetyczny
genomeEcoli = readDNAStringSet("GCF_000008865.2_ASM886v2_cds_from_genomic.fna", format = "fasta")
genomeDrosChrX = readDNAStringSet("Drosophila melanogaster chrX cds.fna", format = "fasta")
genomeDrosChrX = deleteSeq(genomeDrosChrX)


#Podzielenie DNA na 3 czesci: poczatek, srodek i koniec
divideDNA3 = function(DNA){
  width = width(DNA)
  beg = DNAStringSet()
  mid = DNAStringSet()
  end = DNAStringSet()
  
  for (i in 1:length(DNA)) {
    n = width[i]/3
    beg = c(beg, subseq(DNA[i], 1, n))
    mid = c(mid, subseq(DNA[i], n+1, 2*n))
    end = c(end, subseq(DNA[i], 2*n+1, 3*n))
  }
  
  return(list(beg, mid, end))
}
#Szukanie kodonow odpowiadajacych aminokwasowi
whichCodon = function(AA){
  names(GENETIC_CODE[which(GENETIC_CODE == AA)])
}
#Czestosc kodonow kodujaca jeden aminokwas
codonFreq = function(DNA, AA){
  freq = trinucleotideFrequency(DNA, step=3)
  indicies = which( colnames(freq) %in% whichCodon(AA) )
  return(freq[,indicies])
}
#Usuwanie niedostepnych sekwencji
deleteSeq <- function(DNA) {
  DNA[width(DNA)>9]
}
#Testowanie niezaleznosci kodonu od polozenia
testCodonInd <- function(DNA, AA) {
  contingency = data.frame()
  dividedDNA = divideDNA3(DNA)
  
  for (DNA in dividedDNA) {
    sumFreq = apply(codonFreq(DNA, AA), MARGIN=2, sum)
    contingency = rbind(contingency, sumFreq)
  }
  colnames(contingency) = whichCodon(AA)
  rownames(contingency) = c("beg", "mid", "end")
  contingency = t(contingency)
  test = chisq.test(contingency)
  pval = test$p.val
  stat = unname(test$statistic)
  c = discrepancyCoeff(stat, sum(contingency))
  nr = nrow(contingency)
  contingency = cbind(contingency, c(stat,rep(NA, nr-1)), c(c,rep(NA, nr-1)), c(pval,rep(NA, nr-1)))
  nc = ncol(contingency)
  colnames(contingency)[(nc-2):nc] = c("Stat", "C", "Pval")
  return(contingency)
}
#Testowanie zgodnosci AA
testCodonFreq <- function(DNA, AA) {
  contingency = apply(codonFreq(DNA, AA), MARGIN=2, sum)
  expected = sum(contingency)/length(contingency)
  test = chisq.test(contingency)
  stat = test$statistic
  pval = test$p.value
  c = discrepancyCoeff(stat, sum(contingency))
  data = t(c(contingency, L.oczekiwanych=expected, Stat=unname(stat), C=c, Pval=pval))
  return(as.data.frame(data))
}
#Wystepowanie aminokwasow
AAPos <- function(dna, AA) {
  greg = gregexpr(AA,translate(dna))[[1]]
  3*greg[1:length(greg)]-2
}
#Wystepoiwanie kodonow w sekwencji kodujacej
codonPos <- function(dna, AA) {
  cod = whichCodon(AA)
  listPos = list()
  pos = codons(dna)

  for (c in cod) {
    listPos[[c]] = start(pos[as.data.frame(pos)[,1]==c,])
  }
  return(listPos)
}
#Zamiana pozycji na przerwy
breaks <- function(X) {
  n = length(X)
  return( X - c(0,X[1:n-1]) )
}
#Przerwy w wystepowaniu aminokwasow w sekwencji kodujacej
AABreaks <- function(dna, AA)
  breaks(AAPos(dna, AA))
#Przerwy w wystepowaniu kodonow w sekwencji kodujacej
codonBreaks <- function(dna, AA)
  lapply(codonPos(dna, AA), breaks)
#Przerwy w wystepowaniu aminokwasow w calym genomie
genomeAABreaks <- function(DNA, AA) {
  breaks = c()
  for (i in 1:length(DNA)) {
    breaks = c(breaks, AABreaks(DNA[[i]],AA))
  }
  return(breaks)
}
#Przerwy w wystepowaniu kodonow w calym genomie
genomeCodonBreaks <- function(DNA, AA) {
  codons = whichCodon(AA)
  listPos = list()
  for (i in 1:length(DNA)) {
    listBreaks = codonBreaks(DNA[[i]], AA)
    for (c in codons) {
      listPos[[c]] = c(listPos[[c]], listBreaks[[c]])
    }
  }
  return(listPos)
}
#Przedzialy do rownego rozdzielenia rozkladu 
quantIntervals <- function(qdist,theta,n) {
  p = 1/n
  rev(qdist(1:n*p, theta, lower.tail = FALSE))
}
#Testowanie rozkladu geometrycznego
testCodonDist <- function(DNA, AA, n=5, qdist=qgeom, MLE=function(X)1/mean(X)) {
  codonBr = genomeCodonBreaks(DNA,AA)
  AABr = list(genomeAABreaks(DNA, AA))
  names(AABr) = AA
  data = data.frame()
  
  for (cod in c(whichCodon(AA), AA)) {
    breaks = append(codonBr, AABr)[[cod]]
    mle = MLE(breaks)
    int = quantIntervals(qdist, mle, n)

    count = c()
    for (i in 1:(length(int)-1)) 
      count[i] = sum( int[i]<=breaks & breaks<int[i+1])
    count[n] = sum(int[n]<=breaks)
    
    prob = c(pgeom(int[-1]-1,mle) - pgeom(int[-n]-1,mle),pgeom(int[n]-1,mle, lower.tail = FALSE))
    test = chisq.test(count,p=prob)
    pval= test$p.value
    stat = unname(test$statistic)
    c = discrepancyCoeff(stat, length(breaks))
    
    dataCod = data.frame(c(count,NA,NA,NA), c(prob*length(breaks), stat, c, pval))
    colnames(dataCod) = c(paste("Liczba", cod), paste("L.Oczek.", cod))
    data = rbind(data, t(dataCod))
  }
  colnames(data) = c(paste(1:(ncol(data)-3)), "Stat", "C", "Pval")
  return(data)
}
#Tablica przejsc z kodonow
transitionMatrix <- function(dna, AA) {
  listPos = codonPos(dna, AA)
  k = length(whichCodon(AA))
  transMat = matrix(0,k,k)
  
  posVec = NULL
  for (i in 1:k)
    posVec[listPos[[i]]] = i
  posVec = posVec[!is.na(posVec)]

  for (i in 1:k) {
    nextCod = c(posVec[which(posVec==i)+1], 1:k)
    row = table(nextCod)-1
    transMat[i,] = row
  }
  rownames(transMat) = whichCodon(AA)
  colnames(transMat) = whichCodon(AA)
  return(transMat)
}
#Tablica przejsc z kodonow dla genomu
transitionGenome <- function(DNA, AA) {
  k = length(whichCodon(AA))
  transMat = matrix(0,k,k)
  for (i in 1:length(DNA)) {
    transMat = transMat + transitionMatrix(DNA[[i]],AA)
  }
  transMat = diag(1/rowSums(transMat)) %*% transMat
  rownames(transMat) =whichCodon(AA)
  return(transMat)
}
#Wspolczynnik rozbieznosci
discrepancyCoeff <- function(chi, N) unname(chi/N)
##############################################

dividedList = divideDNA3(genome)
letterFrequency(translate(genome[[19]]), AA_ALPHABET)

codonFreq(genomeDrosChrX, "S")
cf = apply(codonFreq(genomeDrosChrX, "S"), MARGIN=2, sum)
chisq.test(codonFreq(genomeDrosChrX, "S")[1:200,])

trinucleotideFrequency(genomeDrosChrX, step=3)[1,]

##############################################
mart <- useMart("ENSEMBL_MART_ENSEMBL", dataset="hsapiens_gene_ensembl")#, dataset="hsapiens_gene_ensembl")
listFilterValues(mart, filter = "chromosome_name")

seq = getSequence(id = "BRCA1", 
                  type = "hgnc_symbol",
                  seqType = "coding", 
                  mart = mart)
genomeDrosGFF =  read.gff("genomic.gff")
genomeDrosGFF = genomeDrosGFF[genomeDrosGFF[,3] =="CDS", ]
chrX = readDNAStringSet("chrX.fna", format = "fasta")
chrX = DNAString(chrX[[1]])
chrX[genomeDrosGFF[5,]$start:genomeDrosGFF[5,]$end]

##############################################TABLICE W LATEX##############################################
printTable <- function(data, cap, path, inclRow) {
  print(xtable(data, caption = cap, display=c(rep("d",ncol(data)-1), "E","E")),
        #xtable(data, caption = cap, digits = 3),
        file=paste(path, ".tex",sep=""), #table.placement = "H", 
        floating.environment = "subtable",
        include.rowname = inclRow,
        math.style.exponents = TRUE)
}

tabFreq <- function(DNA, name) {
  for (aa in AA_ALPHABET[-c(13,18,21:30)]) {
    printTable(testCodonFreq(DNA,aa), paste("Test dla minokwasu",aa), paste("tabFreq",name,aa,sep=""), FALSE)
  }
}
histDens <- function(DNA, AA, ddist=dgeom, MLE=function(X)1/mean(X)) {
  aaBr = genomeAABreaks(DNA, AA)
  l = length(whichCodon(AA))
  nrows = (l+1)%/%2 + (l+1)%%2
  par(mfrow = c(nrows, 2))
  
  codBr = genomeCodonBreaks(DNA, AA)
  for (cod in whichCodon(AA)) {
    m = quantile(codBr[[cod]], 0.99)
    hist(codBr[[cod]], breaks = 400, freq = FALSE, xlim = c(0,m), main = paste("Histogram",cod), xlab = cod)
    lines(1:m, ddist(1:m, MLE(codBr[[cod]])), col="red")
    lines(1:m, dgeom(1:m, 1/mean(codBr[[cod]])), col="blue") # do usuniêcia
  }
  m = quantile(aaBr, 0.99)
  hist(aaBr, breaks = 400, freq = FALSE, xlim = c(0,m), main = paste("Histogram",AA), xlab = AA)
  lines(1:m, ddist(1:m, MLE(aaBr)), col="red")
  lines(1:m, dgeom(1:m, 1/mean(aaBr)), col="blue") # do usuniêcia
  par(mfrow = c(1,1))
}
tabDist <- function(DNA, name) {
  for (aa in AA_ALPHABET[-c(21:30)]) {
    printTable(testCodonDist(DNA, aa, n=3), paste("Test dla aminokwasu", aa),
               paste("tabDist",name,aa,sep=""), TRUE)
  }
}
qqPlots <- function(DNA, AA, qdist=qgeom, MLE=function(X)1/mean(X)) {
  codonBr = genomeCodonBreaks(DNA,AA)
  AABr = list(genomeAABreaks(DNA, AA))
  names(AABr) = AA
  l = length(whichCodon(AA))
  nrows = (l+1)%/%2 + (l+1)%%2
  par(mfrow = c(nrows, 2))
  
  for (cod in c(whichCodon(AA), AA)) {
    breaks = append(codonBr, AABr)[[cod]]
    mle = MLE(breaks)
    qqplot(breaks, qdist(ppoints(breaks), mle), main = paste("Wykres K-K",cod), 
           xlab = paste("Próbkowe kwantyle", cod), ylab = "Teoretyczne kwantyle")
    abline(0,1)
  }
  par(mfrow = c(1,1))
}
tabInd <- function(DNA, name) {
  for (aa in AA_ALPHABET[-c(13,18,21:30)]) {
    printTable(testCodonInd(DNA, aa), paste("Test dla aminokwasu", aa),
               paste("tabInd",name,aa,sep=""), TRUE)
  }
}
tabTrans <- function(DNA, name) {
  for (aa in AA_ALPHABET[-c(13,18,21:30)]) {
    printTable(transitionGenome(DNA, aa), paste("Macierz dla aminokwasu", aa),
               paste("tabTrans",name,aa,sep=""), TRUE)
  }
}
fi <- function(X,i) sum(X==i)
f <- function(X){
  sum1 = 0
  sum2 = 0
  for (j in 1:max(X)) 
    sum2 = sum2 + j^2 * fi(X,j)
  for (j in 1:max(X)) {
    sum1 = sum1 + (j*fi(X,j))/sum2
  }
  return(1-sum1)
}
