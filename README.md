
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
[concepts](https://github.com/WinVector/vtreat/blob/master/extras/vtreat.pdf),
the re-encoding concepts of
[JavaLogistic](https://github.com/WinVector/Logistic), and of the
y-aware scaling concepts of Nina Zumel and John Mount:

  - [Principal Components Regression, Pt.1: The Standard
    Method](http://www.win-vector.com/blog/2016/05/pcr_part1_xonly/)
  - [Principal Components Regression, Pt. 2: Y-Aware
    Methods](http://www.win-vector.com/blog/2016/05/pcr_part2_yaware/)
  - [Principal Components Regression, Pt. 3: Picking the Number of
    Components](http://www.win-vector.com/blog/2016/05/pcr_part3_pickk/)
  - [y-aware scaling in
    context](http://www.win-vector.com/blog/2016/06/y-aware-scaling-in-context/).

The core idea is: other models factor the quantity to be explained into
an explainable versus residual portion (with respect to the given
model). Each of these components are possibly useful for modeling.

``` r
library("CVRTSEncoder")
library("wrapr")

data <- iris
avars <- c("Sepal.Length", "Petal.Length")
evars <- c("Sepal.Width", "Petal.Width")
dep_var <- "Species"
dep_target <- "versicolor"
for(vi in evars) {
  data[[vi]] <- as.character(round(data[[vi]]))
}
str(data)
 #  'data.frame':   150 obs. of  5 variables:
 #   $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
 #   $ Sepal.Width : chr  "4" "3" "3" "3" ...
 #   $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
 #   $ Petal.Width : chr  "0" "0" "0" "0" ...
 #   $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...

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
data %.>%
  head(.) %.>% 
  knitr::kable(.)
```

| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |     c\_001 |      c\_002 |      c\_003 |      c\_004 |
| -----------: | :---------- | -----------: | :---------- | :------ | ---------: | ----------: | ----------: | ----------: |
|          5.1 | 4           |          1.4 | 0           | setosa  | \-1.728808 |   1.0609191 | \-0.0152318 | \-0.0750118 |
|          4.9 | 3           |          1.4 | 0           | setosa  | \-1.377114 | \-0.3953607 | \-0.1182516 | \-0.0072902 |
|          4.7 | 3           |          1.3 | 0           | setosa  | \-1.377114 | \-0.3953607 | \-0.1182516 | \-0.0072902 |
|          4.6 | 3           |          1.5 | 0           | setosa  | \-1.377114 | \-0.3953607 | \-0.1182516 | \-0.0072902 |
|          5.0 | 4           |          1.4 | 0           | setosa  | \-1.728808 |   1.0609191 | \-0.0152318 | \-0.0750118 |
|          5.4 | 4           |          1.7 | 0           | setosa  | \-1.728808 |   1.0609191 | \-0.0152318 | \-0.0750118 |

``` r

f0 <- wrapr::mk_formula(dep_var, avars, outcome_target = dep_target)
print(f0)
 #  (Species == "versicolor") ~ Sepal.Length + Petal.Length
 #  <environment: base>

model0 <- glm(f0, data = data, family = binomial)
summary(model0)
 #  
 #  Call:
 #  glm(formula = f0, family = binomial, data = data)
 #  
 #  Deviance Residuals: 
 #      Min       1Q   Median       3Q      Max  
 #  -1.5493  -0.9437  -0.6451   1.2645   1.7894  
 #  
 #  Coefficients:
 #               Estimate Std. Error z value Pr(>|z|)   
 #  (Intercept)    3.0440     1.9752   1.541  0.12328   
 #  Sepal.Length  -1.1262     0.4611  -2.443  0.01459 * 
 #  Petal.Length   0.7369     0.2282   3.229  0.00124 **
 #  ---
 #  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
 #  
 #  (Dispersion parameter for binomial family taken to be 1)
 #  
 #      Null deviance: 190.95  on 149  degrees of freedom
 #  Residual deviance: 178.32  on 147  degrees of freedom
 #  AIC: 184.32
 #  
 #  Number of Fisher Scoring iterations: 4

data$pred0 <- predict(model0, newdata = data, type = "response")
table(data$Species, data$pred0>0.5)
 #              
 #               FALSE TRUE
 #    setosa        50    0
 #    versicolor    45    5
 #    virginica     38   12

newvars <- c(avars, colnames(enc))
f <- wrapr::mk_formula(dep_var, newvars, outcome_target = dep_target)
print(f)
 #  (Species == "versicolor") ~ Sepal.Length + Petal.Length + c_001 + 
 #      c_002 + c_003 + c_004
 #  <environment: base>

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
 #  (Intercept)  -3.751e+00  2.283e+06   0.000  1.00000   
 #  Sepal.Length  3.507e+00  1.665e+00   2.106  0.03517 * 
 #  Petal.Length -1.170e+01  3.770e+00  -3.103  0.00191 **
 #  c_001         1.355e+01  2.027e+06   0.000  0.99999   
 #  c_002        -4.421e+01  2.159e+06   0.000  0.99998   
 #  c_003         8.422e+01  4.059e+07   0.000  1.00000   
 #  c_004         9.540e+02  7.949e+07   0.000  0.99999   
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
