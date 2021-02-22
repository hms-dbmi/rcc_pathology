# subtype_bwh_analysis.R
# Generates ROC curves and calculates one-vs-rest and multiclass AUCs for BWH subtype classfiication. 
#
# Eliana Marostica
# 12/23/2020

library(tidyverse)
library(pROC)
library(ggthemes)
library(viridis)
library(ggrepel)


###################################################
# tile-level ROC and AUCs
###################################################

date <- "20201212"
models <- c("VGG16", "IncV3", "Res50")

for(model in models){
  
  # read in predictions and true values
  true_file <- paste("./results/SubtypeNoMiTFBWH", model, date, "_bwh_testTrue.txt", sep="")
  pred_file <- paste("./results/SubtypeNoMiTFBWH", model, date, "_bwh_testPredictions.txt", sep="")

  true <- read_delim(true_file, delim=" ", col_names=F) %>%
    rename("pRCC" = X1,
           "ccRCC" = X2,
           "chRCC" = X3) %>%
    select(pRCC, ccRCC, chRCC)
  pred <- read_delim(pred_file, delim=" ", col_names=F) %>%
    rename("pRCC" = X1,
           "ccRCC" = X2,
           "chRCC" = X3) %>%
    select(pRCC, ccRCC, chRCC)
  
  
  # generate ROC
  roclist <- list("pRCC" = roc(true[['pRCC']],pred[['pRCC']]),
                  "ccRCC" = roc(true[['ccRCC']],pred[['ccRCC']]),
                  "chRCC" = roc(true[['chRCC']],pred[['chRCC']]))
  
  # plot ROC curves
  g <- ggroc(roclist, legacy.axes=T, size=1, alpha=0.9) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    theme_minimal() +
    scale_color_colorblind() +
    ggtitle(model) +
    xlab("False Positive Rate") +
    ylab("True Positive Rate") +
    theme(legend.position = "bottom") +
    guides(color=guide_legend(title="Subtype")) #+
  
  # save curves
  ggsave(paste("./results/SubtypeNoMiTFBWH", model, date, "BWH_test_roc.png",sep=""), g, "png", dpi=350, units="in", width=4, height=4)
  ggsave(paste("./Figures/SubtypeNoMiTFBWH", model, "BWH_test_roc.pdf",sep=""), g, "pdf", dpi=350, units="in", width=4, height=4)
  
  # save one-vs-rest and multiclass auc
  response <- factor(colnames(true)[apply(true,1,which.max)], levels=c("pRCC","ccRCC","chRCC"))
  capture.output(print(roclist),
                 file = paste("./results/SubtypeNoMiTF", model, date, split, "BWH_auc.txt",sep=""))
  capture.output(print(multiclass.roc(response, as.matrix(pred))$auc),
                 file = paste("./results/SubtypeNoMiTF", model, date, split, "BWH_auc.txt",sep=""),
                 append = TRUE)
  
}



###################################################
# ROCs and AUC aggregated to patient-level
###################################################

date <- "20201212"
models <- c("Res50")

for(model in models){
  
  # read in predictions, true labels, and patient ids
  true_file <- paste("./results/SubtypeNoMiTFBWH", model, date, "_bwh_testTrue.txt", sep="")
  pred_file <- paste("./results/SubtypeNoMiTFBWH", model, date, "_bwh_testPredictions.txt", sep="")
  id_file <- paste("./results/SubtypeNoMiTFBWH", model, date, "_bwh_testIDs.txt", sep="")
  
  ids <- scan(id_file, what=character())
  
  true <- read_delim(true_file, delim=" ", col_names=F) %>%
    rename("pRCC" = X1,
           "ccRCC" = X2,
           "chRCC" = X3) %>%
    select(pRCC, ccRCC, chRCC) %>%
    mutate(ids = ids) %>%
    group_by(ids) %>%
    summarize("pRCC" = median(pRCC),
              "ccRCC" = median(ccRCC),
              "chRCC" = median(chRCC))
  pred <- read_delim(pred_file, delim=" ", col_names=F) %>%
    rename("pRCC" = X1,
           "ccRCC" = X2,
           "chRCC" = X3) %>%
    select(pRCC, ccRCC, chRCC) %>%
    mutate(ids = ids) %>%
    group_by(ids) %>%
    summarize("pRCC" = median(pRCC),
              "ccRCC" = median(ccRCC),
              "chRCC" = median(chRCC))
  
  
  # generate ROC
  roclist <- list("pRCC" = roc(true[['pRCC']],pred[['pRCC']]),
                  "ccRCC" = roc(true[['ccRCC']],pred[['ccRCC']]),
                  "chRCC" = roc(true[['chRCC']],pred[['chRCC']]))
  
  
  # plot ROC curves
  g <- ggroc(roclist, legacy.axes=T, size=1, alpha=0.9) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    theme_minimal() +
    scale_color_colorblind() +
    ggtitle(model) +
    xlab("False Positive Rate") +
    ylab("True Positive Rate") +
    theme(legend.position = "bottom") +
    guides(color=guide_legend(title="Subtype")) #+
  
  # save ROC curves
  ggsave(paste("./results/SubtypeNoMiTFBWH", model, date, "BWH_test_agg_roc.png",sep=""), g, "png", dpi=350, units="in", width=4, height=4)
  ggsave(paste("./Figures/SubtypeNoMiTFBWH", model, "BWH_test_agg_roc.pdf",sep=""), g, "pdf", dpi=350, units="in", width=4, height=4)
  
  # save one-vs-rest and multiclass auc to text file
  capture.output(print(roclist),
                 file = paste("./results/SubtypeNoMiTFBWH", model, date, "BWH_test_agg_auc.txt",sep=""))
}

