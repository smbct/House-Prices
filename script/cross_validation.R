cross_validation <- function(dataset, nfold) {

    # create X and Y
    X <- dataset[, setdiff(colnames(dataset), "SalePrice")]
    Y <- dataset[, "SalePrice"]

    N <- nrow(X)

    size.CV <- floor(N / nfold)

    CV.err <- numeric(nfold)

    for (i in 1:nfold) {

        # test set for this fold
        i.ts <- ((( i - 1)  * size.CV + 1):(i * size.CV))
        X.ts <- X[i.ts, ]
        Y.ts <- Y[i.ts]

        # training set, the remaining lines
        i.tr <- setdiff(1:N, i.ts)

        X.tr <- X[i.tr, ]
        Y.tr <- Y[i.tr]

        DS <- cbind(X.tr, Y.tr)

        # model <- lm(Y.tr ~ ., DS)
        # model <- rpart(Y.tr ~ ., DS, method="poisson", parms=list(0.5)) #linearRidge
        # model <- nnet(Y.tr ~ ., DS, size=10, linout=T)
        model <- lazy(Y.tr ~ ., DS)
        # model <- svm(Y.tr ~ ., DS, type="nu-regression") # support vector machine (e1071)

        # mean square error
        Y.hat.ts <- predict(model, X.ts)

        # for lazy model
        Y.hat.ts <- unlist(Y.hat.ts)

        CV.err[i] <- sqrt( mean(( log(Y.hat.ts) - log(Y.ts) ) ^ 2, na.rm=T) )
    }

    round(mean(CV.err), digits = 4)
}
