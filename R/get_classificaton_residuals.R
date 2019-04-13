

#' Build residual classification trajectory.
#'
#' Build a cross-validated residual trajectory for a model.
#'
#' @param data The data.frame of data to fit.
#' @param fit_predict A function with signature fit_predict(train_data, vars, dep_var, dep_target, application_data) that returns a matrix with one row of predictions per row of appication_data, and an ordered set of columns of predictions.
#' @param vars character vector, explanatory variable names.
#' @param dep_var character, the name of dependent variable.
#' @param ... not used, force arguments to be bound by name
#' @param dep_target scalar, the value considered to be the target category of dep_var.
#' @param cross_plan a vtreat-style cross validation plan for data rows (list of disjoint tran/app lists where app partitions the data rows).
#' @param cl parallel cluster for processing
#' @return a matrix with the same number of rows as data representing the cross-validated modeling residual trajectories.
#'
#' @examples
#'
#' data <- iris
#' vars <- c("Sepal.Length", "Petal.Length",
#'           "Sepal.Width", "Petal.Width")
#' dep_var <- "Species"
#' dep_target <- "versicolor"
#' resids <- calculate_residual_classification_trajectory(
#'   data = data,
#'   vars = vars,
#'   fit_predict = xgboost_fit_predict_c,
#'   dep_var = dep_var,
#'   dep_target = dep_target
#' )
#'
#' @export
#'
#' @keywords internal
#'
calculate_residual_classification_trajectory <- function(
  data,
  fit_predict,
  vars,
  ...,
  dep_var,
  dep_target = TRUE,
  cross_plan = vtreat::kWayStratifiedY(
    nrow(data),
    3,
    data,
    data[[dep_var]]==dep_target),
  cl = NULL) {
  wrapr::stop_if_dot_args(substitute(list(...)),
                          "calculate_residual_classification_trajectory")
  if(!is.data.frame(data)) {
    stop("calculate_residual_classification_trajectory: data should be a data.frame")
  }
  if(!is.character(vars)) {
    stop("calculate_residual_classification_trajectory: vars should be character")
  }
  if(!is.function(fit_predict)) {
    stop("calculate_residual_classification_trajectory: fit_predict should be a function")
  }
  resids <- NULL
  for(i in seq_len(length(cross_plan))) {
    train_data <- data[cross_plan[[i]]$train, , drop = FALSE]
    application_data <- data[cross_plan[[i]]$app, , drop = FALSE]
    residsi <- xgboost_fit_predict_c(
      train_data = train_data,
      vars = vars,
      dep_var = dep_var,
      dep_target = dep_target,
      application_data = application_data,
      cl = cl)
    if(i==1) {
      resids <- matrix(0, nrow = nrow(data), ncol = ncol(residsi))
    }
    resids[cross_plan[[i]]$app, ] <- residsi
  }
  resids
}
