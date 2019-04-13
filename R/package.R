#' \code{CVRTSEncoder}: Categorical Variable Residual Trajectory Specteral Encoder
#'
#' Re-encode a set of categorical variables jointly as a spectral projection of
#' the trajectory of modeling residuals.  This is intended as a succinct numeric
#' linear representation of a set of categorical variables in a manner that is useful
#' for supervised learning.
#' An extension of the y-aware scaling concepts of Nina Zumel and John Mount,
#' \url{http://www.win-vector.com/blog/2016/05/pcr_part1_xonly/},
#' \url{http://www.win-vector.com/blog/2016/05/pcr_part2_yaware/},
#' \url{http://www.win-vector.com/blog/2016/05/pcr_part3_pickk/},
#' \url{http://www.win-vector.com/blog/2016/06/y-aware-scaling-in-context/}.
#'
#' @docType package
#' @name CVRTSEncoder
NULL

# make sure dot doesn't look like an unbound ref
. <- NULL

#' @importFrom wrapr mk_tmp_name_source
NULL
