

#' @importFrom parallel detectCores
#' @importFrom stats predict
NULL

log_residuals <- function(truth, prediction) {
  # positive examples should always be larger and negative always smaller
  ifelse(truth, -log(prediction), log(1-prediction))
}

#' Adapt xgboost to a fit_predict() signature
#'
#' @param train_data data.frame training data.
#' @param vars character vector, explanatory variable names.
#' @param dep_var character, the name of dependent variable.
#' @param dep_target scalar, the value considered to be the target category of dep_var.
#' @param application_data data.frame application data
#' @param ... not used, force arguments to be bound by name
#' @param cl parallel cluster for processing
#' @return a matrix with the same number of rows as data representing the modeling residual trajectories.
#'
#' @examples
#'
#' data <- iris
#' vars <- c("Sepal.Length", "Sepal.Width",
#'           "Petal.Length", "Petal.Width")
#' dep_var <- "Species"
#' dep_target <- "versicolor"
#' xval <- vtreat::kWayStratifiedY(
#'   nrow(data),
#'   3,
#'   data,
#'   data[[dep_var]]==dep_target)
#' train_data <- data[xval[[1]]$train, , drop = FALSE]
#' application_data <- data[xval[[1]]$app, , drop = FALSE]
#' resids <- xgboost_fit_predict_c(
#'   train_data = train_data,
#'   vars = vars,
#'   dep_var = dep_var,
#'   dep_target = dep_target,
#'   application_data = application_data)
#'
#' @export
#'
xgboost_fit_predict_c <- function(train_data,
                                  vars,
                                  dep_var,
                                  dep_target,
                                  application_data,
                                  ...,
                                  cl = NULL) {
  wrapr::stop_if_dot_args(substitute(list(...)),
                          "xgboost_fit_predict_c")
  ncore <- parallel::detectCores()
  obs_points <- c(1, 2, 5, 10, 30, 100)
  nrounds <- max(obs_points)
  params <- list(max_depth = 5,
                 objective = "binary:logistic",
                 nthread = ncore)
  cfe <- vtreat::mkCrossFrameCExperiment(train_data,
                                         varlist = vars,
                                         outcomename = dep_var,
                                         outcometarget = dep_target,
                                         verbose = FALSE,
                                         parallelCluster = cl)
  sf <- cfe$treatments$scoreFrame
  selvars <- sf$varName[sf$sig<1/nrow(sf)]
  model <- xgboost::xgboost(data = as.matrix(cfe$crossFrame[, selvars, drop = FALSE]),
                   label = cfe$crossFrame[[dep_var]]==dep_target,
                   nrounds = nrounds,
                   params = params,
                   verbose = FALSE)
  app_matrix <-  as.matrix(
    vtreat::prepare(cfe$treatments,
                    application_data,
                    varRestriction = selvars)[, selvars, drop = FALSE])
  resids <- matrix(0, nrow = nrow(app_matrix), ncol = length(obs_points))
  for(i in seq_len(length(obs_points))) {
    preds <- predict(model, newdata = app_matrix, ntree=obs_points[[i]])
    resids[, i] <- log_residuals(application_data[[dep_var]]==dep_target, preds)
  }
  resids
}

