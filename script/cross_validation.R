cross_validation <- function(X, Y, nfold) {

    size.CV <- floor(nrow(X) / nfold)

    CV.err <- numeric(10)

    for(i in 1:nfold) {

        # test set for this fold
        i.ts <- (((i-1)*size.CV+1):(i*size.CV))
        X.ts <- X[i.ts,]
        Y.ts <- Y[i.ts]

        # training set, the remaining lines
        i.tr <- setdiff(1:nrow(X), i.ts)
        X.tr <- X[i.tr,]
        Y.tr <- Y[i.tr]

        DS <- cbind(X.tr, Y.tr)

        model <- lm(Y.tr ~ ., DS)

        # mean square error
        Y.hat.ts <- predict(model, X.ts)
        CV.err[i] <- mean((Y.hat.ts-Y.ts)^2)
    }

    round(mean(CV.err))
}
