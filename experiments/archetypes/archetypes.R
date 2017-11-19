# TODO: Add comment
# 
# Author: molina
###############################################################################


library("archetypes")
library("sirt")


getArchetypes <- function(data, numArchetypes, seed=1) {
	set.seed(seed)
	a <- archetypes(data, numArchetypes, verbose = FALSE)
	arcs <- parameters(a)
	mixt <- coef(a)
	return(list("archetypes" = arcs, "mixture" = mixt))
}

getDirichlet <- function(data, seed=1){
	set.seed(seed)
	alphas <- dirichlet.mle(data)$alpha
	return(alphas)
}