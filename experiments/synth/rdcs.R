library(energy)


rdccancor <- function(x,y,k=5,s=1/6,f=sin) {
  set.seed(42)
  x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)ecdf(u)(u)),1)
  x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
  y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))
  cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
}

rdcdcor <- function(x,y,k=5,s=1/6,f=sin) {
  set.seed(42)
  x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)ecdf(u)(u)),1)
  x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
  y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))
  dcor(cbind(f(x),1),cbind(f(y),1))
}
