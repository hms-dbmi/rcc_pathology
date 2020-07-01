# survival_by_grade_figure.R
# Plot KIRC survival by grade.
#
# 2020.04.24 Eliana Marostica

library(tidyverse)
library(survival)
library(survminer)
library(viridis)

subtype <- "KIRC"
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# get histologic grade
grades <- read_delim("./data/nationwidechildrens.org_clinical_patient_kirc.txt", col_names=T, delim='\t')[c(-1,-2),] %>%
  select(bcr_patient_barcode, ajcc_pathologic_tumor_stage, tumor_grade) %>%
  filter(ajcc_pathologic_tumor_stage == "Stage I") %>%
  rename(Patient_ID = bcr_patient_barcode)

# get original survival data
og_surv_tbl <- read_delim("./data/Overall_Survival_KIRC.txt", col_names=T, delim='\t') %>%
  filter(Study_ID == paste(tolower(subtype),"_tcga",sep="")) %>%
  left_join(., grades, by="Patient_ID") %>%
  filter(!is.na(ajcc_pathologic_tumor_stage), !is.na(OS_MONTHS), !is.na(tumor_grade)) 

# create survival object
og_surv_tbl$SurvObj <- with(og_surv_tbl, Surv(OS_MONTHS, OS_STATUS == "DECEASED"))
head(og_surv_tbl)

# fit survival curve
km_by_hr <- survfit(SurvObj ~ tumor_grade, data=og_surv_tbl, conf.type="log-log")
summary(km_by_hr)
km_by_hr

# log-rank test
sdf <- survdiff(SurvObj ~ tumor_grade, data=og_surv_tbl)
pval <- 1 - pchisq(sdf$chisq, length(sdf$n) - 1)

# plot
gg <- ggsurvplot(km_by_hr, 
                 data=og_surv_tbl,
                 alpha=0.5,
                 risk.table=T,
                 palette=cbPalette,
                 title="Stage I ccRCC Survival by Tumor Grade",
                 conf.int = TRUE)
gg <- gg$plot + ggplot2::annotate("text", x = 0.1, y = 0.1, label = paste("p=",round(pval,3),sep=""), hjust = 0)
show(gg)
ggsave("./KIRCSurvivalHistologicGradeCI.png", gg, "png", width=9, height=6)
