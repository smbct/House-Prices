preprocessing <- function(data) {

    # remove all classes
    factor_variables <- which(sapply(data[1, ], class) == "factor")
    data_preprocessed <- data[, -factor_variables]

    # remove the first col (id)
    data_preprocessed <- data_preprocessed[, -1]

    # replace nan values with the mean
    data_preprocessed <- data.frame(apply(data_preprocessed, 2, replace_na))

    # factor_sel <- c("MSZoning", "Utilities", "Neighborhood", "Condition1", "Condition2", "HouseStyle", "Exterior2nd", "ExterQual", "BsmtCond", "HeatingQC", "KitchenQual", "Functional", "GarageQual", "PoolQC", "Fence", "SaleType", "SaleCondition")
}

replace_na <- function(vec) {
    mean_vec <- mean(vec, na.rm = T)
    vec[is.na(vec)] <- mean_vec
    vec
}


scale_input <- function(dataset) {
    Y <- dataset[, ncol(dataset)]
    cbind(as.data.frame(scale(dataset[, -ncol(dataset)])), Y)
}
