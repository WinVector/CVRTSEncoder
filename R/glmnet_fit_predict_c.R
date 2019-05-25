
#' @importFrom glmnet glmnet
NULL


#' Adapt glmnet to a fit_predict() signature
#'
#' @param train_data data.frame training data.
#' @param vars character vector, explanatory variable names.
#' @param dep_var character, the name of dependent variable.
#' @param dep_target scalar, the value considered to be the target category of dep_var.
#' @param application_data data.frame application data
#' @param ... not used, force arguments to be bound by name
#' @param cl parallel cluster for processing
#' @return a vector with the same number of rows as data representing the model predictions
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
#' preds <- glmnet_fit_predict_c(
#'   train_data = train_data,
#'   vars = vars,
#'   dep_var = dep_var,
#'   dep_target = dep_target,
#'   application_data = application_data)
#'
#' @export
#'
glmnet_fit_predict_c <- function(train_data,
                                 vars,
                                 dep_var,
                                 dep_target,
                                 application_data,
                                 ...,
                                 cl = NULL) {
  wrapr::stop_if_dot_args(substitute(list(...)),
                          "glmnet_fit_predict_c")
  cfe <- vtreat::mkCrossFrameCExperiment(train_data,
                                         varlist = vars,
                                         outcomename = dep_var,
                                         outcometarget = dep_target,
                                         verbose = FALSE,
                                         parallelCluster = cl)
  sf <- cfe$treatments$scoreFrame
  selvars <- sf$varName[sf$sig<1/nrow(sf)]
  alpha = 0.5
  if(length(selvars)<2) {
    # TODO: special case code for too few vars
    print("break")
  }
  cv <- glmnet::cv.glmnet(x = as.matrix(cfe$crossFrame[, selvars, drop = FALSE]),
                          y = cfe$crossFrame[[dep_var]]==dep_target,
                          family = "binomial",
                          alpha = alpha)
  app_matrix <-  as.matrix(
    vtreat::prepare(cfe$treatments,
                    application_data,
                    varRestriction = selvars)[, selvars, drop = FALSE])
  preds <- predict(cv, newx = app_matrix, s = "lambda.min", type = "response")
  as.numeric(preds)
}

