cross_validation <- function(dataset, nfold) {

    # create X and Y
    X <- dataset[,setdiff(colnames(dataset), "SalePrice")]
    Y <- dataset[,"SalePrice"]

    N <- nrow(X)

    size.CV <- floor(N / nfold)

    CV.err <- numeric(nfold)

    for(i in 1:nfold) {

        # test set for this fold
        i.ts <- (((i-1)*size.CV+1):(i*size.CV))
        X.ts <- X[i.ts,]
        Y.ts <- Y[i.ts]

        # training set, the remaining lines
        i.tr <- setdiff(1:N, i.ts)

        X.tr <- X[i.tr,]
        Y.tr <- Y[i.tr]

        DS <- cbind(X.tr, Y.tr)

        model <- rpart(Y.tr ~ ., DS)

        # mean square error
        Y.hat.ts <- predict(model, X.ts)
        CV.err[i] <- mean((Y.hat.ts-Y.ts)^2)
    }

    round(mean(CV.err), digits=4)
}
