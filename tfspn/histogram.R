# TODO: Add comment
# 
# Author: molina
###############################################################################


library("histogram")


getHistogram <- function(data, seed=1) {
	set.seed(seed)
	hh <- histogram(data, verbose = FALSE, plot = FALSE, penalty="penA")
	return(hh)
	breaks <- hh$breaks
	return(list("breaks" = breaks))
}