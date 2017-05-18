# model used: support vector machine
library(e1071)

# features selected from wrapper method

# ensemble of 10 models


# readin the training set
data_train <- read.csv("data/train.csv")
data_test <- read.csv("data/test.csv")

replace_na <- function(vec) {
    mean_vec <- mean(vec, na.rm = T)
    vec[is.na(vec)] <- mean_vec
    vec
}


# remove all classes
factor_variables <- which(sapply(data_train[1, ], class) == "factor")
data_train_preprocessed <- data_train[, -factor_variables]
data_test_preprocessed <- data_test[, -factor_variables]

# remove the first col (id)
data_train_preprocessed <- data_train_preprocessed[, -1]
data_test_preprocessed <- data_test_preprocessed[, -1]

# replace nan values with the mean
data_train_preprocessed <- data.frame(apply(data_train_preprocessed, 2, replace_na))
data_test_preprocessed <- data.frame(apply(data_test_preprocessed, 2, replace_na))

# select a set of features
features_ind <- c(36, 4, 16, 1, 6, 5, 12, 11, 3, 24, 26, 21, 25, 13, 7, 9)
test_set <- data_test_preprocessed[,features_ind]
features_ind <- c(features_ind, ncol(data_train_preprocessed)) # the price column is added
training_set <- data_train_preprocessed[,features_ind]


# an ensemble of 10 models is used
nmodel <- 50

# list of predictions of all models
Y_hat_matrix <- matrix(0, nrow=nrow(test_set), ncol=nmodel)

X_training_set <- training_set[,-ncol(training_set)]
Y_training_set <- training_set[,ncol(training_set)]

# compute one prediction per model
for (i in 1:nmodel) {

    print(i)

    # resampling
    resample_ind <- sample(1:nrow(training_set), rep=T)

    # learning step
    DS <- cbind(X_training_set[resample_ind,], Price=Y_training_set[resample_ind])
    model <- svm(Price ~ ., DS)

    # prediction for this model
    Y_hat_matrix[,i] <- predict(model, test_set)
}

# final prediction: mean of the predictions
Y_hat_final <- apply(Y_hat_matrix, 1, mean)

res <- cbind(data_test[,1], Y_hat_final)
colnames(res) <- c("Id","SalePrice")
write.csv(res, "res.csv", quote=F, row.names=F)
