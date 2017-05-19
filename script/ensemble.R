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

            # create a random subset of features
            sub_features_ind <- sample(1:n, size=(0.6*n), replace=FALSE)
            # print(sub_features_ind)

            # observations used to train this model
            resample_ind <- sample(train_set_ind, rep=T)

            # X_train_set <- dataset[resample_ind, -(n+1)]
            X_train_set <- dataset[resample_ind, sub_features_ind]
            Y_train_set <- dataset[resample_ind, (n+1)]

            DS <- cbind(X_train_set, Y_train_set)

            # build the model
            # model <- rpart(Y_train_set ~ ., DS)
            model <- lazy(Y_train_set ~ ., DS)

            # Y_hat[,r] <- predict(model, X_test_set)
            Y_hat[,r] <- unlist(predict(model, X_test_set[, sub_features_ind]))
            #print("test")
            #Y_hat[,r] <- unlist(Y_hat[,r]) # lazy model
        }

        # build the final prediction (average of all the predictions)
        Y_hat_mean <- apply(Y_hat, 1, mean)

        error <-  sqrt( mean( (log(Y_hat_mean)-log(Y_test_set) )^2, na.rm=T))

        CV_err[i] <- error
        print(i)
    }

    print(paste("CV error=",round(mean(CV_err),digits=4), " ; std dev=",round(sd(CV_err),digits=4)))

}
