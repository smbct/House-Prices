# return a set of features to use for supervised learning
feature_mRMR <- function(dataset) {

    # number of variables
    n <- ncol(dataset) - 1

    #correlation between the variables and the output
    correlation <- abs(cor(dataset[, -(n+1)], dataset[, (n+1)]))

    # selected variables
    selected <- c()

    #candidates variables
    candidates <- 1:n

    for (j in 1:n) {

        # compute the mean of the correlations between the candidates and the selected variables
        redudancy <- numeric(length(candidates))
        if (length(selected) > 0) {
            correl <- cor(dataset[, selected, drop = F],
                dataset[, candidates, drop = F])
            redudancy <- apply(correl, 2, mean)
        }

        score <- correlation[candidates] - redudancy

        best <- candidates[which.max(score)]
        selected <- c(selected, best)
        candidates <- setdiff(candidates, best)

    }

    print(selected)

    for (nb_features in 1:n) {

        DS <- cbind(dataset[,selected[1:nb_features], drop=F], prices=dataset[,(n+1)])
        model <- rpart(prices ~ ., DS)
        Y.hat <- predict(model, dataset[,selected[1:nb_features], drop=F])

        error <- sqrt(mean((Y.hat-dataset[,(n+1)])^2))
        print(nb_features)
        print(error)
    }
}

feature_corr <- function(dataset) {

    # number of variables
    n <- ncol(dataset) - 1

    #correlation between the variables and the output
    correlation <- abs(cor(dataset[, -(n+1)], dataset[, (n+1)]))
    ranking <- sort(correlation, dec=T, index.return=T)$ix

    for (nb_features in 1:n) {

        DS <- cbind(dataset[,ranking[1:nb_features], drop=F], prices=dataset[,(n+1)])
        model <- lm(prices ~ ., DS)

        Y.hat <- predict(model, dataset[,ranking[1:nb_features], drop=F])

        Y.hat2 <- log(Y.hat)
        Y2 <- log(dataset[,(n+1), drop=T])

        error <- mean( ( Y.hat2 - Y2)^2, na.rm=T )
        print(nb_features)
        print(error)
    }

}

feature_PCA <- function(dataset) {



}
