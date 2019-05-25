
test_operators <- function() {

  data <- iris
  vars <- c("Sepal.Length", "Sepal.Width",
            "Petal.Length", "Petal.Width")
  dep_var <- "Species"
  dep_target <- "versicolor"
  xval <- vtreat::kWayStratifiedY(
    nrow(data),
    3,
    data,
    data[[dep_var]]==dep_target)
  train_data <- data[xval[[1]]$train, , drop = FALSE]
  application_data <- data[xval[[1]]$app, , drop = FALSE]
  preds <- xgboost_fit_predict_c(
    train_data = train_data,
    vars = vars,
    dep_var = dep_var,
    dep_target = dep_target,
    application_data = application_data)
  RUnit::checkTrue(is.numeric(preds))

  invisible(NULL)
}
