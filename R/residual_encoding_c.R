

#' @importFrom vtreat prepare
#' @export
vtreat::prepare

stomp_cat_column <- function(v) {
  v <- as.character(v)
  v[is.na(v)] <- "____NA____"
  v
}

table_col <- function(v, y) {
  z <- tapply(y, v, FUN = mean) - mean(y)
  nms <- names(z)
  z <- as.numeric(z)
  names(z) <- nms
  z
}

map_col <- function(v, tab) {
  r <- tab[v]
  names(r) <- NULL
  r[is.na(r)] <- 0
  r
}

#' Build residual class classification trajectory.
#'
#' Build a cross-validated residual trajectory for a model.
#'
#' @param data The data.frame of data to fit.
#' @param ... not used, force arguments to be bound by name
#' @param fit_predict A function with signature fit_predict(train_data, vars, dep_var, dep_target, application_data) that returns a matrix with one row of predictions per row of appication_data, and an ordered set of columns of predictions.
#' @param evars character vector, categorical explanatory variable names to be encoded.
#' @param avars character vector, additional explanatory variable names.
#' @param dep_var character, the name of dependent variable.
#' @param dep_target scalar, the value considered to be the target category of dep_var.
#' @param cross_plan a vtreat-style cross validation plan for data rows (list of disjoint tran/app lists where app partitions the data rows).
#' @param n_comp number of components to generate
#' @param cl parallel cluster for processing
#' @return a matrix with the same number of rows as data representing the cross-validated modeling residual trajectories.
#'
#' @examples
#'
#' data <- iris
#' avars <- c("Sepal.Length", "Petal.Length")
#' evars <- c("Sepal.Width", "Petal.Width")
#' dep_var <- "Species"
#' dep_target <- "versicolor"
#' for(vi in evars) {
#'   data[[vi]] <- as.character(round(data[[vi]]))
#' }
#' cross_enc <- estimate_residual_encoding_c(
#'   data = data,
#'   avars = avars,
#'   evars = evars,
#'   fit_predict = xgboost_fit_predict_c,
#'   dep_var = dep_var,
#'   dep_target = dep_target,
#'   n_comp = 4
#' )
#' enc <- prepare(cross_enc$coder, data)
#' data <- cbind(data, enc)
#' newvars <- c(avars, colnames(enc))
#' f <- wrapr::mk_formula(dep_var, newvars, outcome_target = dep_target)
#' model <- glm(f, data = data, family = binomial)
#' data$pred <- predict(model, newdata = data, type = "response")
#' table(data$Species, data$pred>0.5)
#'
#' @export
#'
estimate_residual_encoding_c <- function(
  data,
  ...,
  fit_predict = xgboost_fit_predict_c,
  evars,
  avars,
  dep_var,
  dep_target = TRUE,
  cross_plan = vtreat::kWayStratifiedY(
    nrow(data),
    3,
    data,
    data[[dep_var]]==dep_target),
  n_comp = 20,
  cl = NULL) {
  wrapr::stop_if_dot_args(substitute(list(...)),
                          "estimate_residual_encoding_c")
  if(!is.data.frame(data)) {
    stop("estimate_residual_encoding_c: data should be a data.frame")
  }
  if(!is.character(avars)) {
    stop("estimate_residual_encoding_c: avars should be character")
  }
  if(!is.character(evars)) {
    stop("estimate_residual_encoding_c: evars should be character")
  }
  if(!is.function(fit_predict)) {
    stop("estimate_residual_encoding_c: fit_predict should be a function")
  }
  # get raw residual trajectory of from the model
  resids <- calculate_residual_classification_trajectory(
    data = data,
    fit_predict = fit_predict,
    vars = avars,
    dep_var = dep_var,
    dep_target = dep_target,
    cross_plan = cross_plan,
    cl = cl)
  # y-aware encode
  nresid <- ncol(resids)
  codes_by_var <- list()
  cross_frame <- data.frame(x = numeric(nrow(data)))
  cross_frame$x <- NULL
  for(vn in evars) {
    var <- stomp_cat_column(data[[vn]])
    codes <- list()
    for(j in seq_len(nresid)) {
      y <- resids[, j, drop = TRUE]
      code_name <- paste(vn, j, sep = "_")
      codes[[code_name]] <- table_col(var, y)
      enc <- numeric(length(y))
      for(j in seq_len(length(cross_plan))) {
        cpj <- cross_plan[[j]]
        code_tab_j <- table_col(var[cpj$train],
                                y[cpj$train])
        enc_j <- map_col(var[cpj$app],
                         code_tab_j)
        enc[cpj$app] <- enc_j
      }
      cross_frame[[code_name]] <- enc
    }
    codes_by_var[[vn]] <- codes
  }
  # get the projection that has most of the y-aware
  # variation
  s <- svd(cross_frame, nu = 0, nv = n_comp)
  v_mat <- s$v
  # project down
  cross_frame <- as.matrix(cross_frame) %*% v_mat
  colnames(cross_frame) <- sprintf("c_%03g", seq_len(ncol(cross_frame)))
  # return the info
  r <- list(codes_by_var = codes_by_var,
            v_mat = v_mat)
  class(r) <- "model_residual_trajectory_code"
  list(coder = r, cross_frame = cross_frame)
}


#' Residual trajectory encode categorical variables.
#'
#' Residual encode.
#'
#' @param treatmentplan a model_residual_trajectory_code coder
#' @param dframe data frame to be encoded.
#' @param ... not used, force arguments to be bound by name
#' @return data frame encoding the categorical columns specified in treatmentplan
#'
#' @export
#'
prepare.model_residual_trajectory_code <- function(treatmentplan,
                                                   dframe,
                                                   ...) {
  wrapr::stop_if_dot_args(substitute(list(...)),
                          "prepare.model_residual_trajectory_code")
  # get the raw codes
  enc_frame <- data.frame(x = numeric(nrow(dframe)))
  enc_frame$x <- NULL
  for(vn in names(treatmentplan$codes_by_var)) {
    codes <- treatmentplan$codes[[vn]]
    var <- stomp_cat_column(dframe[[vn]])
    for(code_name in names(codes)) {
      enc <- map_col(var, codes[[code_name]])
      enc_frame[[code_name]] <- enc
    }
  }
  # project down
  enc_frame <- as.matrix(enc_frame) %*% treatmentplan$v_mat
  colnames(enc_frame) <- sprintf("c_%03g", seq_len(ncol(enc_frame)))
  enc_frame
}
