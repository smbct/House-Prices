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

X_training_set <- training_set[,-ncol(training_set)]
Y_training_set <- training_set[,ncol(training_set)]

# compute the prediction

# learning step
DS <- cbind(X_training_set, Price=Y_training_set)
model <- svm(Price ~ ., DS)

# prediction for this model
Y_hat <- predict(model, test_set)

res <- cbind(data_test[,1], Y_hat)
colnames(res) <- c("Id","SalePrice")
write.csv(res, "res.csv", quote=F, row.names=F)
