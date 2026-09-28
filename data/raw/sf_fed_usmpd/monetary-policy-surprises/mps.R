## Calculate factors for monetary policy surprises from USMPD
## Filename: mps.R
## Authors: Michael Bauer and Caroline Foshee
## Date: 12/09/2025

library(readxl)
library(dplyr)
library(lubridate)
library(readr)

##################################################
### CONFIGURATION
## Path to USMPD file
USMPD_FILENAME <- "USMPD.xlsx"
## Optional: Sample end date for factor calculation (set to NULL for full sample)
END_DATE <- NULL
## END_DATE <- as.Date("2025-11-19")

##################################################

## One-year GSW yield for normalization of MPS
y1_filename <- "y1.csv"
gsw_filename <- "feds200628.csv"
create_y1_file <- function() {
    ## download updated GSW yields from Board website
    ## source: https://www.federalreserve.gov/data/yield-curve-tables/feds200628_1.html
    res <- download.file(url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
                         destfile = gsw_filename)
    y1 <- read_csv(gsw_filename, skip=9, col_names = TRUE) %>%
        select(Date, SVENY01) %>%
        na.omit %>%
        filter(year(Date) >= 1994)
    write_csv(y1, file=y1_filename)
}
load_y1 <- function() {
    ## load daily changes in one-year yield
    if (!file.exists(y1_filename)) {
        cat("\n Downloading GSW yields...\n")
        create_y1_file()
    }
    y1 <- read_csv(y1_filename, show_col_types = FALSE) %>%
        mutate(dy1 = SVENY01 - dplyr::lag(SVENY01)) %>%
        select(Date, dy1)
    return(y1)
}

## load high-frequency money market surprises from USMPD
read_sheet <- function(sheet) {
    d <- read_xlsx(USMPD_FILENAME, sheet,
                   na = c("", "NA", "N/A", "#N/A")) %>%
        mutate(Date = as.Date(date_time)) %>%
        select(c(Date, MP1, MP2, ED2, ED3, ED4))
    if (!is.null(END_DATE))
        d <- filter(d, Date <= END_DATE)
    return(d)
}

## Calculate factors
calc_mps <- function(data) {
    ## input: data.frame with Date column and intraday rate changes (usually, MP1, MP2, ED2-ED4)
    data <- na.omit(data)
    ## Calculate principal components
    pca_result <- prcomp(data %>% select(-Date),
                         scale = TRUE)  # normalize each variable
    ## Extract the first principal component
    data$PC1 <- pca_result$x[,1]

    ## load one-year GSW yield for normalization
    y1 <- load_y1()
    if (max(y1$Date) < max(data$Date)) {
        cat("\n Updating one-year GSW yield...\n")
        create_y1_file() # download GSW yields again
        y1 <- load_y1()
    }

    ## combine PC1 with svensson yield and normalize
    d <- data %>%
        select(Date, PC1) %>%
        left_join(y1, by = "Date")
    model <- lm(dy1 ~ PC1, d)
    d <- d %>%
        mutate(MPS = coef(model)["PC1"] * PC1)
    ## return data.frame with Date and MPS
    d %>% select(Date, MPS)
}

## Statements
statements <- read_sheet("Statements") %>%
    calc_mps %>%
    rename(STMT = MPS)

## Press Conferences
pressconferences <- read_sheet("Press Conferences") %>%
    calc_mps %>%
    rename(PC = MPS)

## Monetary Events
monetaryevents <- read_sheet("Monetary Events") %>%
    calc_mps %>%
    rename(ME = MPS)

# Combine
meetings <- statements %>%
    full_join(pressconferences, by = "Date") %>%
    full_join(monetaryevents, by = "Date")

## Minutes
minutes <- read_sheet("Minutes") %>%
    select(-MP1) %>%  # exclude MP1 because it's identically zero for minutes
    filter(year(Date) >= 2005) %>% # once minutes started being released 3 weeks after meeting
    calc_mps %>%
    rename(MIN = MPS)

## Export to CSV
## -> two separate files since event dates are different
write_csv(meetings, "mps.csv")
write_csv(minutes, "mps_minutes.csv")

