# mutcount_pancancer_analysis.R
# Evaluate performance of a regression CNN prediction mutation count
#
# Eliana Marostica 5/22/20

library(tidyverse)
library(ggthemes)
library(pROC)
library(e1071)


scan_preds_trues <- function(pred_filepath, true_filepath){
  preds <- scan(pred_filepath)
  trues <- scan(true_filepath)
  
  stopifnot(length(preds) == length(trues))
  stopifnot(length(preds) > 0)
  stopifnot(is.numeric(preds))
  stopifnot(is.numeric(trues))
  return(list("preds" = preds, "trues"=trues))
}

aggregate_preds_trues <- function(ptid_filepath, preds, trues){
  # aggregate to patient
  ptids <- read_delim(ptid_filepath, delim=" ", col_names=F, col_types=cols()) %>%
    separate(X1, into=c("ptid", "path"), 12) %>%
    select(-path) %>%
    rename(mutcount = X2)
  
  stopifnot(nrow(ptids) == length(trues))
  
  pred_tbl <- tibble("ptid"=ptids$ptid, "mutcount"=preds)
  true_tbl <- tibble("ptid"=ptids$ptid, "mutcount"=trues)
  
  agg_pred <- pred_tbl %>%
    group_by(ptid) %>%
    summarize(med_mutcount = median(mutcount))
  agg_true <- true_tbl %>%
    group_by(ptid) %>%
    summarize(med_mutcount = median(mutcount))
  return(list("agg_pred" = agg_pred,
              "agg_true" = agg_true))
}

agg_regression_to_binary <- function(train_pred_filepath, ptid_train, agg_pred){
  train_preds <- scan(train_pred_filepath)
  ptids_train <- read_delim(ptid_train, delim=" ", col_names=F, col_types = cols()) %>%
    separate(X1, into=c("ptid", "path"), 12) %>%
    select(-path) %>%
    rename(mutcount = X2)
  
  agg_train_pred <- tibble("ptid"=ptids_train$ptid, "mutcount"=train_preds) %>%
    group_by(ptid) %>%
    summarize(med_mutcount = median(mutcount))
  
  med_patient_pred <- median(agg_train_pred$med_mutcount)
  
  agg_pred_grouped <- agg_pred %>%
    mutate(pred_group = case_when(med_mutcount < med_patient_pred ~ "low",
                                  med_mutcount >= med_patient_pred ~ "high"))
  return(agg_pred_grouped)
}

agg_regression_to_tertile <- function(train_pred_filepath, ptid_train, agg_pred){
  train_preds <- scan(train_pred_filepath)
  ptids_train <- read_delim(ptid_train, delim=" ", col_names=F, col_types = cols()) %>%
    separate(X1, into=c("ptid", "path"), 12) %>%
    select(-path) %>%
    rename(mutcount = X2)
  
  agg_train_pred <- tibble("ptid"=ptids_train$ptid, "mutcount"=train_preds) %>%
    group_by(ptid) %>%
    summarize(med_mutcount = median(mutcount))
  
  lower <- quantile(agg_train_pred$med_mutcount, seq(0, 1, 1/3))[[2]]
  upper <- quantile(agg_train_pred$med_mutcount, seq(0, 1, 1/3))[[3]]
  
  agg_pred_grouped <- agg_pred %>%
    mutate(pred_group = case_when(med_mutcount < lower ~ "low",
                                  med_mutcount >= lower & med_mutcount < upper ~ "med",
                                  med_mutcount >= upper ~ "high"))
  return(agg_pred_grouped)
}

#################################
subtype <- "KIRC"
model <- "Res50"
date <- "20200609_2"
predspace <- TRUE

train_pred_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                             ifelse(predspace,"_trainPredictionsLogNorm.txt","_trainPredictions.txt"), sep="")
val_pred_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                           ifelse(predspace,"_valPredictionsLogNorm.txt","_valPredictions.txt"), sep="")
test_pred_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                            ifelse(predspace,"_testPredictionsLogNorm.txt","_testPredictions.txt"), sep="")
train_true_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                             ifelse(predspace,"_trainTrueLogNorm.txt","_trainTrue.txt"), sep="")
val_true_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                           ifelse(predspace,"_valTrueLogNorm.txt","_valTrue.txt"), sep="")
test_true_filepath <- paste("./", subtype, "PanCancerMutCountRegression", model, date, 
                            ifelse(predspace,"_testTrueLogNorm.txt","_testTrue.txt"), sep="")
ptid_train <- "./mutcount_KIRC_pancancer_trainShuffled.txt"
ptid_val <- "./mutcount_KIRC_pancancer_val.txt"
ptid_test <- "./mutcount_KIRC_pancancer_test.txt"

pred_filepath <- test_pred_filepath
true_filepath <- test_true_filepath
ptid_filepath <- ptid_test

preds_trues_list <- scan_preds_trues(pred_filepath, true_filepath)
preds <- preds_trues_list$preds
trues <- preds_trues_list$trues
cor(preds,trues, method = "spearman")
plot(preds,trues)

agg_list <- aggregate_preds_trues(ptid_filepath, preds, trues)
agg_pred <- agg_list$agg_pred
agg_true <- agg_list$agg_true
cor(agg_pred$med_mutcount, agg_true$med_mutcount, method = "spearman")
plot(agg_pred$med_mutcount, agg_true$med_mutcount)


agg_results <- left_join(agg_pred, agg_true, by="ptid") %>%
  rename("pred" = med_mutcount.x,
         "true" = med_mutcount.y)

lm <- lm(true ~ pred, data=agg_results)
spm <- cor.test(agg_results$true, agg_results$pred, method="spearman")

gg <- ggplot(agg_results, aes(x=pred, y=true)) +
  geom_point(alpha=0.7) +
  geom_smooth(method='lm') +
  xlab("Predicted Log-Normalized Value") +
  ylab("True Log-Normalized Value") +
  scale_color_colorblind() +
  labs(title=paste("Tumor Mutation Count 10-Fold CV Predictions", model, date),
       subtitle=paste("rho=", round(spm$estimate, 3), ", p=", round(spm$p.value, 4), sep=""))

gg
ggsave(paste("./",subtype, "MutationCount", model, date, "_scatter.pdf", sep=""), plot=print(gg), device="pdf", width=6, height=6, dpi=300)
