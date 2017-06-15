library(nnet)

X <- data_preprocessed[, -ncol(data_preprocessed)]
Y <- data_preprocessed[, ncol(data_preprocessed)]

Y_mu <- mean(Y)
Y_sigma <- sd(Y)

X <- scale(X)
Y <- scale(Y)

max_w <- 50

plot_mean <- numeric(max_w)
plot_sigma <- numeric(max_w)
plot_ab <- seq(1:max_w)

for (i in 1:max_w) {
  
  # model
  dataset <- cbind(X, Y)
  model <- nnet(X, Y, size=i, linout=T, trace=F, maxit=500, MaxNWts=90000)
  
  # predictions
  Y_hat <- predict(model, X)
  
  Y_hat <- Y_hat*Y_sigma + Y_mu
  
  # compute the error
  error_vec <- (log(Y_hat) - log(Y*Y_sigma + Y_mu))^2
  error_mean <- sqrt(mean(error_vec, na.rm=T))
  error_sigma <- sd(error_vec, na.rm=T)
  
  print(paste("mean: ", error_mean, " sigma: ", error_sigma))
  plot_mean[i] <- error_mean
  plot_sigma[i] <- error_sigma
}

plot(plot_ab, plot_mean)
points(plot_sigma)
