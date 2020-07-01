# Analyze MTLR survival classification 10-fold cross-validation results.
#
# 2020.02.11 Eliana Marostica

library(tidyverse)
library(survival)
library(survminer)
library(purrrlyr)

subtype <- "KIRC"
model <- "Res50"
agg_test <- data.frame("ptid" = c(), "median_pred" = c())
true_filepath <- paste("./Overall_Survival_", subtype,".txt",sep="") #"./data/cptac_surv.txt" # 

for(foldI in 0:9){
  trainfilepath <- paste("./results/",subtype, "SurvivalMTLRStageI10FoldCV", model, "_F", foldI, "_trainPredictions.txt", sep="")
  testfilepath  <- paste("./results/",subtype, "SurvivalMTLRStageI10FoldCV", model, "_F", foldI, "_testPredictions.txt", sep="")
  fold_train   <- read_delim(trainfilepath,col_names=F, delim=" ", col_types = cols())
  fold_test    <- read_delim(testfilepath,col_names=F, delim=" ", col_types = cols())
  
  # Get patient ids for patch-level predictions of current fold
  traintest <- read_delim(paste("./survival_",subtype,"_stageI_trainTest10FoldCV.txt", sep=""), col_names=F, delim=" ", col_types = cols()) %>%
    separate(X1, into=c("ptid","path"), ifelse(subtype=="CPTAC", 9, 12)) %>% 
    select(-path)
  
  test_ptids <- traintest %>%
    filter(X4 == foldI+1)
  test_ptids <- test_ptids$ptid
  
  stopifnot(nrow(fold_train) + nrow(fold_test) == nrow(traintest), nrow(fold_test) == length(test_ptids))
  
  fold_test$ptid <- test_ptids
  fold_test <- fold_test[,c(3,1,2)]
  
  # Determine bin prediction at patch-level
  
  test_binpreds <- tibble("ptid" = fold_test[[1]], "binpred" = apply(fold_test[,2:3], 1, which.max))
  
  # Aggregate test predictions to patient-level
  
  agg_test_i <- test_binpreds %>%
    group_by(ptid) %>%
    summarize(median_pred = round(median(binpred)))
  
  print(paste("FOLD", foldI))
  print(table(agg_test_i$median_pred))
  
  agg_test <- rbind(agg_test, agg_test_i)
}

high_ptids <- agg_test$ptid[which(agg_test$median_pred == 2)]
low_ptids <- agg_test$ptid[which(agg_test$median_pred == 1)]

surv_tbl <- read_delim(true_filepath, col_names=T, delim='\t') %>%
  filter(Study_ID == paste(tolower(subtype),"_tcga",sep="")) %>%
  mutate(group = case_when(Patient_ID %in% high_ptids ~ "High",
                           Patient_ID %in% low_ptids ~ "Low"),
         group = factor(group, levels=c("Low","High"))) %>%
  as.data.frame() %>%
  filter(!is.na(group))

# Create survival object
surv_tbl$SurvObj <- with(surv_tbl, Surv(OS_MONTHS, OS_STATUS == "DECEASED"))

survdiff(SurvObj ~ group, data=surv_tbl)

# Fit survival curve
km_by_hr <- survfit(SurvObj ~ group, data=surv_tbl, conf.type="log-log", type="kaplan-meier")
summary(km_by_hr)
km_by_hr

# Plot
gg <- ggsurvplot(km_by_hr, 
                 data=surv_tbl, 
                 risk.table=T,
                 palette="Set1",
                 conf.int = TRUE)
gg
ggsave(paste("./",subtype, "SurvivalMTLRStageI10FoldCV", model, "_KMCurve.pdf", sep=""), plot=print(gg), device="pdf", width=6, height=6, dpi=300)

table(surv_tbl$OS_STATUS,surv_tbl$group)
