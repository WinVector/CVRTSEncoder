% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glmnet_fit_predict_r.R
\name{glmnet_fit_predict_r}
\alias{glmnet_fit_predict_r}
\title{Adapt glmnet to a fit_predict() signature}
\usage{
glmnet_fit_predict_r(train_data, vars, dep_var, application_data, ...,
  cl = NULL)
}
\arguments{
\item{train_data}{data.frame training data.}

\item{vars}{character vector, explanatory variable names.}

\item{dep_var}{character, the name of dependent variable.}

\item{application_data}{data.frame application data}

\item{...}{not used, force arguments to be bound by name}

\item{cl}{parallel cluster for processing}
}
\value{
a vector with the same number of rows as data representing the model predictions
}
\description{
Adapt glmnet to a fit_predict() signature
}
\examples{

data <- iris
vars <- c("Sepal.Length", "Sepal.Width",
          "Petal.Length", "Petal.Width")
data$tvalue <- as.numeric(data$Species=="versicolor")
dep_var <- "tvalue"
xval <- vtreat::kWayStratifiedY(
  nrow(data),
  3,
  data,
  data[[dep_var]])
train_data <- data[xval[[1]]$train, , drop = FALSE]
application_data <- data[xval[[1]]$app, , drop = FALSE]
preds <- glmnet_fit_predict_r(
  train_data = train_data,
  vars = vars,
  dep_var = dep_var,
  application_data = application_data)

}
