# return a set of features to use for supervised learning
feature_mRMR <- function(dataset) {

    # number of variables
    n <- ncol(dataset)-1

    correlation <- abs(cor(dataset[,-n], dataset[,n]))
    selected <- c()
    candidates <- 1:n

    for(j in 1:n) {

        redudancy <- numeric(length(candidates))
        if(length(selected) > 0) {
            correl <- cor(dataset[,selected,drop=F], dataset[,candidates,drop=F])
            redudancy <- apply(correl, 2, mean)
        }

        score <- correlation[candidates] - redudancy

        best <- candidates[which.max(score)]
        selected <- c(selected, best)
        candidates <- setdiff(candidates, best)

    }



}
