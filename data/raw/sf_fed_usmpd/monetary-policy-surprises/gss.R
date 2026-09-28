## Calculate target and path factors (GSS 2005) and NS surprise from USMPD
## based on high-frequency rate surprises around FOMC statements
## Filename: gss.R
## Authors: Michael Bauer
## Date: 05/18/2026
## Reference: Guerkaynak, Sack, and Swanson (2005, IJCB)

library(readxl)
library(dplyr)
library(lubridate)
library(readr)

##################################################
### CONFIGURATION
USMPD_FILENAME <- "USMPD.xlsx"
END_DATE <- NULL
## END_DATE <- as.Date("2025-11-19")

FUTNAMES <- c("MP1", "MP2", "ED2", "ED3", "ED4")

##################################################

## Load statement surprises from USMPD
read_statements <- function() {
    d <- read_xlsx(USMPD_FILENAME, "Statements",
                   na = c("", "NA", "N/A", "#N/A")) %>%
        mutate(Date = as.Date(date_time)) %>%
        select(Date, MP1, MP2, ED2, ED3, ED4)
    if (!is.null(END_DATE))
        d <- filter(d, Date <= END_DATE)
    d
}

## One-year GSW yield for normalization
y1_filename <- "y1.csv"
gsw_filename <- "feds200628.csv"
create_y1_file <- function() {
    download.file(url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
                  destfile = gsw_filename)
    y1 <- read_csv(gsw_filename, skip = 9, col_names = TRUE, show_col_types = FALSE) %>%
        select(Date, SVENY01) %>%
        na.omit %>%
        filter(year(Date) >= 1994)
    write_csv(y1, file = y1_filename)
}
load_y1 <- function() {
    if (!file.exists(y1_filename)) {
        cat("\n Downloading GSW yields...\n")
        create_y1_file()
    }
    y1 <- read_csv(y1_filename, show_col_types = FALSE) %>%
        mutate(dy1 = SVENY01 - dplyr::lag(SVENY01)) %>%
        select(Date, dy1)
    return(y1)
}

## Statement surprise (= Nakamura-Steinsson surprise): rescaled first PC
calc_stmt <- function(dat) {
    pca_result <- prcomp(dat[FUTNAMES], scale = TRUE)
    dat$PC1 <- pca_result$x[, 1]

    y1 <- load_y1()
    if (max(y1$Date) < max(dat$Date)) {
        cat("\n Updating one-year GSW yield...\n")
        create_y1_file()
        y1 <- load_y1()
    }
    d <- dat %>%
        select(Date, PC1) %>%
        left_join(y1, by = "Date")
    model <- lm(dy1 ~ PC1, d)
    dat$STMT <- as.numeric(coef(model)["PC1"] * dat$PC1)
    dat %>% select(Date, STMT)
}

## GSS target and path factors
calc_gss <- function(dat) {
    fut <- as.matrix(dat[FUTNAMES])

    ## standardize to zero mean and unit SD
    fut <- sweep(fut, 2, colMeans(fut))
    fut <- t(t(fut) / apply(fut, 2, sd))

    ## first two PCs, rescaled to unit variance
    eig <- eigen(cov(fut))
    W <- eig$vectors[, 1:2]
    F_mat <- fut %*% W %*% diag(1 / sqrt(eig$values[1:2]))

    ## rotation Z = F*U where Z2 has no effect on MP1 (GSS appendix eq A8-A11)
    g <- lm(dat$MP1 ~ F_mat - 1)$coef
    ## sign convention: positive g[1] ensures target is positively correlated with MP1
    if (g[1] < 0) { F_mat <- -F_mat; g <- -g }
    alpha1 <- 1 / sqrt(1 + (g[2] / g[1])^2)
    alpha2 <- alpha1 * g[2] / g[1]
    ## beta orthogonal to alpha with unit length
    beta1 <- 1 / sqrt(1 + (alpha1 / alpha2)^2)
    beta2 <- -alpha1 / alpha2 * beta1
    U <- rbind(c(alpha1, beta1), c(alpha2, beta2))
    Z <- F_mat %*% U

    ## normalize target: moves MP1 one-for-one
    mod1 <- lm(dat$MP1 ~ Z[, 1])
    dat$target <- as.numeric(Z[, 1] * mod1$coef[2])

    ## normalize path: same effect on ED4 as target
    mod2 <- lm(dat$ED4 ~ dat$target + Z[, 2])
    dat$path <- as.numeric(Z[, 2] * mod2$coef[3] / mod2$coef[2])

    dat %>% select(Date, target, path)
}

##################################################
## Main

statements <- na.omit(read_statements())

## statement surprise and GSS target/path
stmt <- calc_stmt(statements)
gss <- calc_gss(statements)

## combine and compute average
results <- inner_join(stmt, gss, by = "Date") %>%
    mutate(avg = (target + path) / 2)

##################################################
## Report

cat(sprintf("\nN = %d observations, %s to %s\n",
            nrow(results), min(results$Date), max(results$Date)))
cat(sprintf("Standard deviations: STMT=%.4f, Target=%.4f, Path=%.4f\n",
            sd(results$STMT), sd(results$target), sd(results$path)))
cat("\nCorrelations:\n")
print(round(cor(results %>% select(STMT, target, path)), 3))
cat(sprintf("\nCorr(STMT, avg(target,path)) = %.3f\n", cor(results$STMT, results$avg)))

## demonstrate normalizations: target moves MP1 1-for-1, path has same ED4 effect as target
d <- statements %>% inner_join(gss, by = "Date")
m1 <- lm(MP1 ~ target + path, d)
m2 <- lm(ED4 ~ target + path, d)
cat("\nNormalization checks (regressions on target and path):\n")
cat(sprintf("           %10s %10s\n", "MP1", "ED4"))
cat(sprintf("  target   %10.3f %10.3f\n", coef(m1)["target"], coef(m2)["target"]))
cat(sprintf("           %10s %10s\n",
            sprintf("(%.3f)", summary(m1)$coef["target",2]),
            sprintf("(%.3f)", summary(m2)$coef["target",2])))
cat(sprintf("  path     %10.3f %10.3f\n", coef(m1)["path"], coef(m2)["path"]))
cat(sprintf("           %10s %10s\n",
            sprintf("(%.3f)", summary(m1)$coef["path",2]),
            sprintf("(%.3f)", summary(m2)$coef["path",2])))
cat(sprintf("  R2       %10.3f %10.3f\n", summary(m1)$r.squared, summary(m2)$r.squared))
cat(sprintf("  N        %10d %10d\n", nobs(m1), nobs(m2)))

## Export (optional)
## write_csv(results %>% select(Date, STMT, target, path), "gss.csv")
