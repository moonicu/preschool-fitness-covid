# V-sit endurance centiles from a GAMLSS model fitted to the present 2017-2019 records
# (Supplementary Table S7, column "GAMLSS"), for comparison with the empirical percentiles
# of the same records and with the published GAMLSS values of Oh & Lee (2025).
#
# Input : output_v2/private/vse_pre_records.csv (sex, age, endu; written by analysis_v2.py;
#         record-level data, not distributed)
# Output: output_v2/supplementary/S_vse_gamlss_present.csv (centiles only)
# Usage : Rscript gamlss_compare.R      (requires the R package 'gamlss')
#
# Model per sex: mu and sigma as P-splines of age in months. A Box-Cox (BCCGo) family is tried
# first, within a time limit; if it fails to converge, the log-normal (LOGNO) fit is used. The
# family actually used is recorded in the output.
suppressMessages(library(gamlss))
d <- read.csv("output_v2/private/vse_pre_records.csv")
ages <- c(48, 60, 72, 84); cents <- c(10, 50, 90)

fit_sex <- function(ds) {
  m0 <- gamlss(endu ~ pb(age), sigma.fo = ~pb(age), family = LOGNO, data = ds, trace = FALSE, n.cyc = 200)
  setTimeLimit(elapsed = 300, transient = TRUE)
  m1 <- tryCatch(gamlss(endu ~ pb(age), sigma.fo = ~pb(age), nu.fo = ~1, family = BCCGo, data = ds,
                        trace = FALSE, n.cyc = 200, mu.start = fitted(m0, "mu"),
                        sigma.start = fitted(m0, "sigma"), method = mixed(20, 200)),
                 error = function(e) NULL, warning = function(w) NULL)
  setTimeLimit(elapsed = Inf)
  ok <- !is.null(m1) && isTRUE(m1$converged) && all(is.finite(fitted(m1, "mu")))
  if (ok) list(m = m1, fam = "BCCGo") else list(m = m0, fam = "LOGNO")
}

out <- data.frame()
for (s in c(1, 2)) {
  ds <- na.omit(subset(d, sex == s & endu > 0, select = c(age, endu)))
  set.seed(1); f <- fit_sex(ds); m <- f$m
  for (a in ages) {
    nd <- data.frame(age = a)
    mu <- predict(m, what = "mu", newdata = nd, type = "response", data = ds)
    sg <- predict(m, what = "sigma", newdata = nd, type = "response", data = ds)
    for (cc in cents) {
      q <- if (f$fam == "BCCGo") {
        nu <- predict(m, what = "nu", newdata = nd, type = "response", data = ds)
        qBCCGo(cc / 100, mu = mu, sigma = sg, nu = nu)
      } else qLOGNO(cc / 100, mu = mu, sigma = sg)
      out <- rbind(out, data.frame(sex = s, age = a, centile = cc, family = f$fam, gamlss = round(q, 1)))
    }
  }
  cat("sex", s, "records", nrow(ds), "family", f$fam, "GAIC", round(GAIC(m)), "\n")
}
write.csv(out, "output_v2/supplementary/S_vse_gamlss_present.csv", row.names = FALSE)
print(out, row.names = FALSE)
