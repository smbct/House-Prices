preprocessing <- function(data) {

    # remove all classes
    factor_variables <- which(sapply(data[1,], class) == "factor")
    data_preprocessed <- data[, -factor_variables]


    # replace nan values with the mean
    data_preprocessed <- data.frame(apply(data_preprocessed,2,replace_na))

}

replace_na <- function(vec) {
    mean_vec <- mean(vec, na.rm=T)
    vec[is.na(vec)] <- mean_vec
    vec
}
