
feature_mRMR <- function(dataset, nfold) {

    # number of variables
    n <- ncol(dataset)-1

    # number of examples
    N <- nrow(dataset)

    # size of a test set
    CV_size <- floor(N/nfold)

    CV_err<-matrix(0,nrow=n,ncol=nfold)

    for (i in 1:nfold) {

        test_set_ind <- ((i-1)*CV_size+1):(i*CV_size)
        X_test_set <- dataset[test_set_ind, -(n+1)]
        Y_test_set <- dataset[test_set_ind, (n+1)]

        train_set_ind <- setdiff(1:N, test_set_ind)
        X_train_set <- dataset[train_set_ind, -(n+1)]
        Y_train_set <- dataset[train_set_ind, (n+1)]

        # correlation between the variables and the output
        correlation <- abs(cor(X_train_set, Y_train_set))

        # selected variables
        selected <- c()

        # candidates variables
        candidates <- 1:n

        for (j in 1:n) {
            # compute the mean of the correlations between the candidates and the selected variables
            redudancy <- numeric(length(candidates))
            if (length(selected) > 0) {
                correl <- cor(X_train_set[, selected, drop = F],
                    X_train_set[, candidates, drop = F])
                redudancy <- apply(correl, 2, mean)
            }

            score <- correlation[candidates] - redudancy

            best <- candidates[which.max(score)]
            selected <- c(selected, best)
            candidates <- setdiff(candidates, best)
        }

        # print(selected)

        for (nb_features in 1:n) {

            DS <- cbind(X_train_set[,selected[1:nb_features], drop=F],Y_train_set)
            model <- svm(Y_train_set ~ ., DS)
            Y_hat <- predict(model, X_test_set[,selected[1:nb_features], drop=F])

            error <- sqrt(mean((log(Y_hat)-log(Y_test_set))^2, na.rm=T))
            # print(nb_features)
            # print(error)

            CV_err[nb_features, i] <- error
        }

        print(i)
        print(selected)
    }

    print(paste("#Features: ",c(1:n)," ; CV error=",round(apply(CV_err,1,mean),digits=4), " ; std dev=",round(apply(CV_err,1,sd),digits=4)))
}

feature_corr <- function(dataset, nfold) {

    # number of variables
    n <- ncol(dataset) - 1

    # number of examples
    N <- nrow(dataset)

    # size of a test set
    CV_size <- floor(N/nfold)

    CV_err<-matrix(0,nrow=n,ncol=nfold)

    # cross validation loop
    for (i in 1:nfold) {

        test_set_ind <- ((i-1)*CV_size+1):(i*CV_size)
        X_test_set <- dataset[test_set_ind, -(n+1)]
        Y_test_set <- dataset[test_set_ind, (n+1)]

        train_set_ind <- setdiff(1:N, test_set_ind)
        X_train_set <- dataset[train_set_ind, -(n+1)]
        Y_train_set <- dataset[train_set_ind, (n+1)]

        # correlation between the variables and the output
        correlation <- abs(cor(X_train_set, Y_train_set))
        ranking <- sort(correlation, dec=T, index.return=T)$ix

        for (nb_features in 1:n) {

            DS <- cbind(X_train_set[,ranking[1:nb_features], drop=F], Y_train_set)
            model <- svm(Y_train_set ~ ., DS)

            Y.hat <- predict(model, X_test_set[,ranking[1:nb_features], drop=F])

            error <- sqrt(mean( ( log(Y.hat) - log(Y_test_set) )^2, na.rm=T ))
            print(nb_features)
            print(error)

            CV_err[nb_features, i] <- error
        }

    }

    print(paste("#Features: ",c(1:n)," ; CV error=",round(apply(CV_err,1,mean),digits=4), " ; std dev=",round(apply(CV_err,1,sd),digits=4)))

}

feature_PCA <- function(dataset, nfold) {

    # number of variables
    n <- ncol(dataset) - 1

    # number of examples
    N <- nrow(dataset)

    # size of a test set
    CV_size <- floor(N/nfold)

    X_pca <- data.frame(prcomp(dataset[, -(n+1)], retx=T)$x)


    CV_err<-matrix(0,nrow=n,ncol=nfold)

    # cross validation loop
    for (i in 1:nfold) {

        test_set_ind <- ((i-1)*CV_size+1):(i*CV_size)
        X_test_set <- X_pca[test_set_ind,-(n+1)]
        Y_test_set <- dataset[test_set_ind, (n+1)]

        train_set_ind <- setdiff(1:N, test_set_ind)
        X_train_set <- X_pca[train_set_ind,-(n+1)]
        Y_train_set <- dataset[train_set_ind, (n+1)]

        for(nb_features in 1:n) {

            # select only few columns
            DS<-cbind(X_train_set[, 1:nb_features, drop=F], Y_train_set)

            model <- svm(Y_train_set ~ ., DS)

            Y_hat <- predict(model, X_train_set[,1:nb_features, drop=F])

            error <- sqrt(mean( (log(Y_hat) - log(Y_test_set))^2, na.rm=T ))
            # print(nb_features)
            # print(error)

            CV_err[nb_features, i] <- error

        }
    }

    print(paste("#Features: ",c(1:n)," ; CV error=",round(apply(CV_err,1,mean),digits=4), " ; std dev=",round(apply(CV_err,1,sd),digits=4)))
}

feature_wrap <- function(dataset, nfold) {

    # number of variables
    n <- ncol(dataset) - 1

    # number of examples
    N <- nrow(dataset)

    # size of a test set
    CV_size <- floor(N/nfold)

    selected <- NULL

    for (round in 1:n) {

        candidates <- setdiff(1:n, selected)

        CV_err <- matrix(0, nrow=length(candidates), ncol=nfold)

        for (j in 1:length(candidates)) {

            features_to_include <- c(selected, candidates[j])

            for (i in 1:nfold) {

                test_set_ind <- ((i-1)*CV_size+1):(i*CV_size)
                X_test_set <- dataset[test_set_ind, features_to_include]
                Y_test_set <- dataset[test_set_ind, (n+1)]

                train_set_ind <- setdiff(1:N, test_set_ind)
                X_train_set <- dataset[train_set_ind, features_to_include]
                Y_train_set <- dataset[train_set_ind, (n+1)]

                DS <- as.data.frame(cbind(X_train_set, Y_train_set))

                model <- rpart(Y_train_set ~ ., DS)

                Y_hat <- predict(model, X_test_set)

                error <- sqrt( mean( (log(Y_hat) - log(Y_test_set))^2, na.rm=T ))

                CV_err[j, i] <- error

                #print(i)
                #print(error)
            }
        }

        CV_err_mean <- apply(CV_err, 1, mean)
        CV_err_sd <- apply(CV_err, 1, sd)

        selected_current<-which.min(CV_err_mean)
        selected<-c(selected,candidates[selected_current])

        print(paste("Round ",round," ; Selected feature: ",candidates[selected_current]," ; CV error=",round(CV_err_mean[selected_current],digits=4), " ; std dev=",round(CV_err_sd[selected_current],digits=4)))
    }
}
