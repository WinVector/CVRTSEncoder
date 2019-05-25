

to_link <- function(p) {
  eps <- 1.0e-6
  p <- pmax(eps, pmin(1-eps, p))
  log(p/(1-p))
}

to_resid <- function(p, target) {
  eps <- 1.0e-6
  p <- pmax(eps, pmin(1-eps, p))
  ifelse(target, -log(p), log(1-p))
}


#' Build residual classification trajectory.
#'
#' Build a cross-validated residual trajectory for a model.  The core idea is: other models factor the quantity to be explained into
#' an explainable versus residual portion (with respect to the given model).  Each of these components are possibly useful for modeling.
#'
#' @param data The data.frame of data to fit.
#' @param fit_predict A function with signature fit_predict(train_data, vars, dep_var, dep_target, application_data) that returns a matrix with one row of predictions per row of appication_data, and an ordered set of columns of predictions.
#' @param vars character vector, explanatory variable names.
#' @param dep_var character, the name of dependent variable.
#' @param ... not used, force arguments to be bound by name
#' @param dep_target scalar, the value considered to be the target category of dep_var.
#' @param cross_plan a vtreat-style cross validation plan for data rows (list of disjoint tran/app lists where app partitions the data rows).
#' @param fitter fit/predict signature function
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
#' augments <- calculate_residual_classification_trajectory(
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
  fitter = xgboost_fit_predict_c,
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
  nround = 5
  extra_cols <- data.frame(x = numeric(nrow(data)))
  extra_cols$x <- NULL
  target <- data[[dep_var]]==dep_target
  resid_cols <- data.frame(synthetic_resid_0 = to_resid(rep(mean(target), length(target)), target))
  for(r in seq_len(nround)) {
    tryCatch({
      preds <- numeric(nrow(data))
      for(i in seq_len(length(cross_plan))) {
        train_data <- data[cross_plan[[i]]$train, , drop = FALSE]
        application_data <- data[cross_plan[[i]]$app, , drop = FALSE]
        mvars <- vars
        if(ncol(extra_cols)>0) {
          train_data <- cbind(train_data, extra_cols[cross_plan[[i]]$train, , drop = FALSE])
          application_data <- cbind(application_data, extra_cols[cross_plan[[i]]$app, , drop = FALSE])
          mvars <- c(vars, colnames(extra_cols))
        }
        predsi <- fitter(
          train_data = train_data,
          vars = mvars,
          dep_var = dep_var,
          dep_target = dep_target,
          application_data = application_data,
          cl = cl)
        preds[cross_plan[[i]]$app] <- predsi
      }
      extra_cols[[paste0("synthetic_col_", ncol(extra_cols)+1)]] <- to_link(preds)
      resid_cols[[paste0("synthetic_resid_", ncol(resid_cols)+1)]] <- to_resid(preds, target)
    },
    error = function(e) { warning("CVRTSEncoder::calculate_residual_classification_trajectory caught", e) }
    )
  }
  cbind(extra_cols, resid_cols)
}
