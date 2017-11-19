
library('bnlearn')
#source('lteData.R')

DATASETS = c(#'anneal-U', 
             'australian', 
             #'auto', 
             #'balance-scale', 
             'breast', 
             #'breast-cancer', 
             'cars', 
             'cleve', 
             'crx', 
             'diabetes', 
             #'german', 
             'german-org', 
             'glass', 
             'glass2', 
             'heart', 
             'iris')

eval.bn <- function(bn, data){
  n.instances <- dim(data)[1]
  ll <- logLik(bn, data)
  avg.ll <- ll / n.instances
  return(avg.ll)
}

learn.bn.and.eval <- function(data.name, data.dir='/home/molina/git/TF_SPN/TFSPN/src/mlutils/datasets/MLC/proc-db/proc/', learn.method='mmhc'){
  
  data.path <- paste(data.dir, data.name, sep='')
  cat('\n\n\n*** Processing dataset', data.path, '***\n\n')
  
  ## loading dataset splits
  data <- load.dataset.splits(data.path)
  
  cat('\ttrain split loaded, containing', dim(data$train), 'entries\n')
  cat('\tvalid split loaded, containing', dim(data$valid), 'entries\n')
  cat('\ttest split loaded, containing', dim(data$test), 'entries\n')
  
  # cat('Train Column types', str(data$train), '\n')
  # cat('Valid Column types', str(data$valid), '\n')
  # cat('Test Column types', str(data$test), '\n')
  
  data$train = as.data.frame(data$train[,1])
  data$valid = as.data.frame(data$valid[,1])
  data$test = as.data.frame(data$test[,1])
  
  ## learning the network structure
  if (learn.method == 'mmhc')
  {
    net <- mmhc(data$train)
  }
  else if (learn.method == 'hc')
  {
    net <- hc(data$train)
  }
  else if (learn.method == 'tabu')
  {
    net <- tabu(data$train)
  }
  else if (learn.method == 'gs')
  {
    net <- gs(data$train)
  }
  else if (learn.method == 'iamb')
  {
    net <- iamb(data$train)
  }
  else if (learn.method == 'mmpc')
  {
    net <- mmpc(data$train)
  }
  else if (learn.method == 'si.hiton.pc')
  {
    net <- si.hiton.pc(data$train)
  }
  else if (learn.method == 'rsmax2')
  {
    net <- rsmax2(data$train)
  }
  else if (learn.method == 'nb')
  {
    net <- naive.bayes(data$train, training = 'class')
  }
  
  
  plot(net)
  
  ## learning the network weights
  net.fit <- bn.fit(net, data$train, keep.fitted = TRUE)
  
  ## evaluating
  train.ll <- eval.bn(net.fit, data$train)
  valid.ll <- eval.bn(net.fit, data$valid)
  test.ll <- eval.bn(net.fit, data$test)
  
  cat('\n\ndataset', data.name, '\n')
  cat('\ttrain ll', train.ll, '\n')
  cat('\tvalid ll', valid.ll, '\n')
  cat('\ttest ll', test.ll, '\n')
}

learn.bn.and.eval("australian", learn.method='rsmax2')  
#for (d in DATASETS){
#  learn.bn.and.eval(d, learn.method='rsmax2')  
#}


