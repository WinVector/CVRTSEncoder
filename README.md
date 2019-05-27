
<!-- README.md is generated from README.Rmd. Please edit that file -->

[`CVRTSEncoder`](https://github.com/WinVector/CVRTSEncoder) is a
categorical variable encoding for supervised learning.

This package is still in a research and development mode. Functionality
and interfaces may change.

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

| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |     c\_001 |    c\_002 |     c\_003 |      c\_004 |
| -----------: | :---------- | -----------: | :---------- | :------ | ---------: | --------: | ---------: | ----------: |
|          5.1 | 4           |          1.4 | 0           | setosa  | \-6.858432 | 1.8427407 | \-2.388919 |   0.0299307 |
|          4.9 | 3           |          1.4 | 0           | setosa  | \-5.278981 | 0.3747432 |   2.706826 | \-0.2840527 |
|          4.7 | 3           |          1.3 | 0           | setosa  | \-5.278981 | 0.3747432 |   2.706826 | \-0.2840527 |
|          4.6 | 3           |          1.5 | 0           | setosa  | \-5.278981 | 0.3747432 |   2.706826 | \-0.2840527 |
|          5.0 | 4           |          1.4 | 0           | setosa  | \-6.858432 | 1.8427407 | \-2.388919 |   0.0299307 |
|          5.4 | 4           |          1.7 | 0           | setosa  | \-6.858432 | 1.8427407 | \-2.388919 |   0.0299307 |

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

model <- glmnet::cv.glmnet(as.matrix(data[, newvars, drop = FALSE]), 
                           as.numeric(data[[dep_var]]==dep_target), 
                           family = "binomial")
coef(model, lambda = "lambda.min")
 #  7 x 1 sparse Matrix of class "dgCMatrix"
 #                        1
 #  (Intercept)   0.6076907
 #  Sepal.Length  .        
 #  Petal.Length -0.5206192
 #  c_001         0.5398427
 #  c_002        -0.9798759
 #  c_003         .        
 #  c_004         0.1553199
data$pred <- as.numeric(predict(model, newx = as.matrix(data[, newvars, drop = FALSE]), s = "lambda.min"))
table(data$Species, data$pred>0.5)
 #              
 #               FALSE TRUE
 #    setosa        50    0
 #    versicolor     4   46
 #    virginica     50    0
```
