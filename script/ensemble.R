ensemble_models <- function(dataset, nfold, nmodel) {

    N <- nrow(dataset)
    n <- ncol(dataset)-1

    size_CV <- floor(N/nfold)

    CV_err <- numeric(nfold)

    for(i in 1:nfold) {

        test_set_ind <- ((i-1)*size_CV+1):(i*size_CV)
        X_test_set <- dataset[test_set_ind,-(n+1)]
        Y_test_set <- dataset[test_set_ind, (n+1)]

        train_set_ind <- setdiff(1:N, test_set_ind)
        Y_hat <- matrix(0, nrow=nrow(X_test_set), ncol=nmodel)

        for(r in 1:nmodel) {

            # observations used to train this model
            resample_ind <- sample(train_set_ind, rep=T)

            X_train_set <- dataset[resample_ind, -(n+1)]
            Y_train_set <- dataset[resample_ind, (n+1)]

            DS <- cbind(X_train_set, Y_train_set)

            # build the model
            model <- svm(Y_train_set ~ ., DS)

            Y_hat[,r] <- predict(model, X_test_set)
        }

        # build the final prediction (average of all the predictions)
        Y_hat_mean <- apply(Y_hat, 1, mean)

        error <-  mean(sqrt(( log(Y_hat)-log(Y_test_set) )^2), na.rm=T)

        CV_err[i] <- error
        print(i)
    }

    print(paste("CV error=",round(mean(CV_err),digits=4), " ; std dev=",round(sd(CV_err),digits=4)))

}
