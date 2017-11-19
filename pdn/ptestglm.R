library(iterators)
library(grid)
library(partykit)
library(Formula)
library(foreach)
library(doMC)
library(parallel)
library(dplyr)
library(gtools)
library(igraph)
library(digest)
library(energy)
library(dummies)
registerDoMC(detectCores()-1)

options(warn = -1)
#data<- read.csv("~/Dropbox/pspn/spyn/bin/experiments/graphlets/wl/3enzymes.build_wl_corpus.csv", comment.char="#")

#data<-read.csv2("/Users/alejomc/Dropbox/pspn/spyn/experiments/graphclassification/wl/1mutag.build_wl_corpus.csv", header = FALSE, sep = ",",  quote="\"", skip=1)

#data<-read.csv2("/home/molina/Dropbox/Papers/pspn/spyn/experiments/graphclassification/wl/1mutag.build_wl_corpus.csv", header = FALSE, sep = ",",  quote="\"", skip=1)

#data<-read.csv2("/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/2ptc.build_wl_corpus.csv" , header = FALSE,  sep = ",",  quote="\"", skip=1)

#fmla = fmla
#tree = glmtree(fmla, data, family="poisson", verbose = FALSE, maxdepth=2)


rdc <- function(x,y,s=1/6,f=sin, linear=FALSE) {
  if(var(x) == 0|| var(y) == 0) {
    return(0)
  }
  x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)ecdf(u)(u)),1)
  k = max(ncol(x), ncol(y))
  
  set.seed(42)
  x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
  set.seed(43)
  y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))
  
  
  if(linear){
	xy <- cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
	yx <- cancor(cbind(f(y),1),cbind(f(x),1))$cor[1]
  }else{
	xy <- dcor(cbind(f(x),1),cbind(f(y),1))
	yx <- dcor(cbind(f(y),1),cbind(f(x),1))
  }
  
  return(max(xy,yx))
}

testRDC <- function(data, ohe, featureTypes, linear) {
  #write.csv(data, file="/tmp/data.txt")
  
  adjm <- matrix(0, ncol=ncol(data), nrow = ncol(data))
  
  inputpos <- t(combn(ncol(data),2))
  
  columnnames <- names(data)
    
  rdccoef <- foreach(i = 1:nrow(inputpos), .combine=rbind) %dopar% {
  	c1 <- inputpos[i,1]
  	c2 <- inputpos[i,2]
  	
  	d1 <- data[,c1]
  	d2 <- data[,c2]
  	
  	if(ohe){
  		if(featureTypes[[c1]] == "categorical"){
  			d1 <- dummy(columnnames[c1], data, sep="_")
  		}
  		
  		if(featureTypes[[c2]] == "categorical"){
  			d2 <- dummy(columnnames[c2], data, sep="_")
  		}
  	}
  	
  	rdcv1 <- rdc(d1,d2, linear=linear)
  	
  	#cat(rdcv1)
  	#cat(rdcv2)
  	
    return(rdcv1)
  }
  for (i in 1:nrow(inputpos)){
    adjm[inputpos[i,1], inputpos[i,2]] = rdccoef[i]
  }
  diag(adjm) <- 1
  #write.table(adjm,"/tmp/adj.txt",sep=",",row.names=FALSE)
  return(adjm) 
}

subclustersRDC <- function(tests, threshold) {
  tests[tests<threshold] = 0
  tests[tests > 0] = 1
  
  g = graph.adjacency(as.matrix(tests), mode="max", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  res = c$membership
  names(res)<-NULL
  
  #print(paste("subclusters:", length(unique(c$membership))  ))
  return(res)
}

getIndependentGroupsRDC <- function(data, threshold, ohe, featureTypes, linear) {
  return(subclustersRDC(testRDC(data, ohe, featureTypes, linear), threshold))
}




ptestpdnglm <- function(data, families) {
  start.time <- Sys.time()
  data = as.data.frame(data)
  cols = ncol(data)
  vard = apply(data, 2, var)
  n = mixedsort(names(data))
  #print(data)
  
  #for (i in 1:cols) {
  adjc <- foreach(i = 1:cols, .combine=rbind) %do% {
      
      fmla = as.formula(paste(n[i], " ~ . ", sep = ''))
      fam = families[[i]]
      
      
      tree = glmtree(fmla, data, family=fam, verbose = FALSE, maxdepth=2)
      #tree = glmtree(fmla, data, family="poisson", verbose = FALSE, maxit = 25, maxdepth=2)
      #tree = glmtree(fmla, data, family="poisson", verbose = FALSE, minsize=5)
      
      treeresults = sctest.modelparty(tree,node=1)
      ptest = as.data.frame(matrix(1, ncol=cols,nrow=1))
      colnames(ptest)<-n
      
      
      #next
      
      if(!is.null(ptest)){
        ptest[1,colnames(treeresults)] = treeresults[2,colnames(treeresults)]

        return(ptest)
      }
            
      return(matrix(1,ncol=cols))
    
  }
  
  
  rownames(adjc)<-n
  
  
  #print(adjc)
  
  pvals = pmin(adjc[upper.tri(adjc)],t(adjc)[upper.tri(adjc)])
  
  

  adjc[upper.tri(adjc)] = pvals
  adjc[lower.tri(adjc)] = 0

  adjc = adjc + t(adjc)
  diag(adjc)<-1
  
  
  
  #print(paste("dims:", dim(data)[1], "x", dim(data)[2],
  #            "families: ", families,
  #            "ptest time: ", format(difftime(Sys.time(), start.time)),
  #            "ptest min: ", min(pvals),
  #            "ptest max: ", max(pvals),
  #            "ptest mean: ", mean(pvals)
  #))
  
  return(adjc) 
}



findpval <- function(data){
  
  ptests = ptestglmblock(data)
  
  return(median(ptests[ptests<0.05]))
}

ptestglmblock <- function(data, family) {
  
  
  if(dim(data)[2] < 5){
    return(ptestpdnglm(data))  
  }
  
  start.time <- Sys.time()
  
  n = mixedsort(names(data))
  
  nblocks = ceiling(dim(data)[2] / 100)
  if(nblocks < 1){
    nblocks = 1
  }
  
  blocks = split(n, rank(n) %% nblocks)
  
  
  ptests = foreach(ni = n, .combine=rbind) %do% {
  #for(ni in n){
    #print(paste("v:", ni))
    ptestscols = foreach(bk = blocks, .combine=c) %do% {
    #for(bk in blocks){
      othervars = bk[bk != ni]
      fmla = as.formula(paste(ni, " ~ ", paste(othervars, collapse = ' + '), sep=' '))
      
      tree = glmtree(fmla, data, family=family, verbose = FALSE, maxdepth=2)
      
      ptest = sctest.modelparty(tree,node=1)[2,]
      
      if(is.null(ptest)){
        ptest = array(1, dim=length(othervars))
        #ptest = matrix(1,ncol=length(othervars))
        names(ptest) <- othervars
      }
      return(ptest)
    }
    ptestscols[ni] = 1
    ptestscols = ptestscols[mixedsort(names(ptestscols))]
    #ptestscols[]
    return(ptestscols)
  }
  rownames(ptests) = n
  #if there was a problem computing the pvalues, default to 1
  ptests[is.na(ptests)] <- 1
  
  
  print(paste("dims:", dim(data)[1], "x", dim(data)[2], 
                "ptest time: ", format(difftime(Sys.time(), start.time)),
                "ptest min: ", min(ptests),
                "ptest max: ", max(ptests),
                "ptest mean: ", mean(ptests)
                ))
  
  return(ptests)
}




subconnected2 <- function(ptests) {
  library(igraph)
  
  start.time <- Sys.time()
  
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  maxval = max(wptests)
  repeat{
    #get clusters
    g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
    c = clusters(g)
    
    #variable assignment to subsets
    vas[,]=0
    
    for(w in order(c$csize, decreasing = TRUE)){
      j = which.min(apply(vas, 2, sum))
      vas[c$membership == w,j] = 1      
    }
    binsizes = apply(vas, 2, sum)
    r = min(binsizes)/sum(vas)
    
    ptestcut = median(wptests[wptests>0])
    
    if(ptestcut == maxval){
      #we couldn't really partition this
      vas[,1]=1
      vas[,2]=0
      break
    }
    
    if(r >= 0.3){
      break
    }
    
    wptests[wptests <= ptestcut] = 0
  }
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
  
  #plot.igraph(g,vertex.label=V(g)$name,layout=layout.fruchterman.reingold, edge.color="black",edge.width=E(g)$weight)
                         
  #min_cut(g, capacity = E(g)$weight, value.only = FALSE)
  #cluster_spinglass(g, spins = 2)
}

subconnected <- function(ptests, alpha) {

  #alpha = 1-alpha
  #wptests = 1-ptests
  #wptests[wptests <= alpha] = 0
  
  ptests[ptests>alpha] = 0
  
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  
  g = graph.adjacency(ptests, mode="min", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  print(paste("Number of connected components", c$no,":",toString(table(c$membership))))
  
  for(w in order(c$csize, decreasing = TRUE)){
    j = which.min(apply(vas, 2, sum))
    vas[c$membership == w,j] = 1      
  }
  
  return(vas)
}

subclusters <- function(ptests, alpha) {
  ptests[ptests>alpha] = 0
  ptests[ptests > 0] = 1
  
  g = graph.adjacency(as.matrix(ptests), mode="min", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  res = c$membership
  names(res)<-NULL
  
  #print(paste("subclusters:", length(unique(c$membership))  ))
  
  return(res) 
}

getIndependentGroupsAlpha <- function(data, alpha, families) {
  #print(digest(data, algo="sha1"))
  return(subclusters(ptestpdnglm(data, families), alpha))
}

getIndependentGroupsAlpha3 <- function(data, alpha) {
  return(subconnected(ptestglmblock(data), alpha))
}

getIndependentGroupsAlpha2 <- function(data, alpha) {
  return(subconnectedCombined(ptestglmblock(data)))
}


subconnectedCombined <- function(ptests){
  library(igraph)
  
  start.time <- Sys.time()
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  
  if(c$no ==  1){
    s = cluster_spinglass(g, spins = 2)
    
    vas[s[[1]],1] = 1
    
    if(length(s) > 1){
      vas[s[[2]],2] = 1
    }
    
  }else{
    for(w in order(c$csize, decreasing = TRUE)){
      j = which.min(apply(vas, 2, sum))
      vas[c$membership == w,j] = 1      
    }
  }
  
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
   
}


subconnectedCD <- function(ptests) {
  library(igraph)

  start.time <- Sys.time()
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
  
  s = cluster_spinglass(g, spins = 2)
  
  vas[s[[1]],1] = 1
  
  if(length(s) > 1){
    vas[s[[2]],2] = 1
  }
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
}

#dim = 8
#nval = 100*dim
#data = as.data.frame(ceiling(matrix(runif(nval,min=1,max=10),ncol=dim)))


#start.time <- Sys.time()
#pt = ptestpdnglm(data[,c(117,15,114,590,668,sample(1:dim(data)[2],10))])
#ptests = getIndependentGroupsAlpha(data, 0.001, "poisson")
#end.time <- Sys.time()
#time.taken <- end.time - start.time
#print(time.taken)
#print(round(pt,2))
#diag(pt) = 1
#print(any(round(pt,2) < 0.05))

