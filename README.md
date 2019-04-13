
<!-- README.md is generated from README.Rmd. Please edit that file -->

[`CVRTSEncoder`](https://github.com/WinVector/CVRTSEncoder) is a
categorical variable encoding for supervised learning.

Re-encode a set of categorical variables jointly as a spectral
projection of the trajectory of modeling residuals. This is intended as
a succinct numeric linear representation of a set of categorical
variables in a manner that is useful for supervised learning.

The concept is y-aware encoding the trajectory of non-linear model
residuals in terms of target categorical variables.

The idea is an extension of the
[`vtreat`](https://github.com/WinVector/vtreat) coding
[concepts](https://github.com/WinVector/vtreat/blob/master/extras/vtreat.pdf)
and of the y-aware scaling concepts of Nina Zumel and John Mount:

  - [Principal Components Regression, Pt.1: The Standard
    Method](http://www.win-vector.com/blog/2016/05/pcr_part1_xonly/)
  - [Principal Components Regression, Pt. 2: Y-Aware
    Methods](http://www.win-vector.com/blog/2016/05/pcr_part2_yaware/)
  - [Principal Components Regression, Pt. 3: Picking the Number of
    Components](http://www.win-vector.com/blog/2016/05/pcr_part3_pickk/)
  - [y-aware scaling in
    context](http://www.win-vector.com/blog/2016/06/y-aware-scaling-in-context/).

<!-- end list -->

``` r
library("CVRTSEncoder")

data <- iris
avars <- c("Sepal.Length", "Petal.Length")
evars <- c("Sepal.Width", "Petal.Width")
dep_var <- "Species"
dep_target <- "versicolor"
for(vi in evars) {
  data[[vi]] <- as.character(round(data[[vi]]))
}
cross_enc <- estimate_residual_encoding_c(
  data = data,
  avars = avars,
  evars = evars,
  fit_predict = xgboost_fit_predict_c,
  dep_var = dep_var,
  dep_target = dep_target,
  n_comp = 4
)
enc <- prepare(cross_enc$coder, data)
data <- cbind(data, enc)
newvars <- c(avars, colnames(enc))
f <- wrapr::mk_formula(dep_var, newvars, outcome_target = dep_target)
model <- glm(f, data = data, family = binomial)
 #  Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
summary(model)
 #  
 #  Call:
 #  glm(formula = f, family = binomial, data = data)
 #  
 #  Deviance Residuals: 
 #       Min        1Q    Median        3Q       Max  
 #  -1.44336  -0.01155   0.00000   0.00203   2.69526  
 #  
 #  Coefficients:
 #                 Estimate Std. Error z value Pr(>|z|)   
 #  (Intercept)  -3.336e+00  2.237e+06   0.000  1.00000   
 #  Sepal.Length  3.507e+00  1.665e+00   2.106  0.03517 * 
 #  Petal.Length -1.170e+01  3.770e+00  -3.103  0.00191 **
 #  c_001         8.219e+01  2.475e+06   0.000  0.99997   
 #  c_002         2.931e+02  3.002e+07   0.000  0.99999   
 #  c_003        -3.916e+00  5.280e+05   0.000  0.99999   
 #  c_004         5.871e+02  2.437e+06   0.000  0.99981   
 #  ---
 #  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
 #  
 #  (Dispersion parameter for binomial family taken to be 1)
 #  
 #      Null deviance: 190.954  on 149  degrees of freedom
 #  Residual deviance:  22.619  on 143  degrees of freedom
 #  AIC: 36.619
 #  
 #  Number of Fisher Scoring iterations: 21

data$pred <- predict(model, newdata = data, type = "response")
table(data$Species, data$pred>0.5)
 #              
 #               FALSE TRUE
 #    setosa        50    0
 #    versicolor     3   47
 #    virginica     49    1
```
