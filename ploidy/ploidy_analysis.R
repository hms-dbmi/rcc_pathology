# ploidy_analysis.R
# ROC curve creation for ploidy >2 prediction
#
# 2021.01.01 Eliana Marostica

library(tidyverse)
library(pROC)
library(ggthemes)
library(viridis)
library(ggrepel)

# Functions

#' Create a roc object.
#' Helper function.
#' 
#' @param task The prediction task.
#' @param subtype The renal cell carcinoma subtype.
#' @param model The neural network architecture.
#' @param split train val or test
create_roc <- function(task, subtype, model, date, split){
  
  true_file <- paste("./results/", subtype, task, model, date, "Trues_", split, ".txt", sep="")
  pred_file <- paste("./results/", subtype, task, model, date, "Predictions_", split, ".txt", sep="")
  
  
  true <- scan(true_file)
  pred <- scan(pred_file)
  
  return(roc(true, pred))
}


create_roc_agg <- function(subtype, model, date, split){
  if(split == "train"){
    filename <- paste("./data/ploidy_", str_to_lower(subtype), "_trainShuffled.txt", sep="")
  } else{
    filename <- paste("./data/ploidy_", str_to_lower(subtype), "_", split, ".txt", sep="")
  }
  
  ids <- read_delim(filename, delim=" ", col_names = F) %>%
    separate(X1, into=c("Sample_ID", "file"), 15)
  
  true_file <- paste("./results/", subtype, "Ploidy", model, date, "Trues_", split, ".txt", sep="")
  pred_file <- paste("./results/", subtype, "Ploidy", model, date, "Predictions_", split, ".txt", sep="")
  true <- tibble("id" = ids$Sample_ID, "ploidy" = scan(true_file)) %>%
    group_by(id) %>%
    summarize(ploidy = median(ploidy))
  pred <- tibble("id" = ids$Sample_ID, "ploidy" = scan(pred_file)) %>%
    group_by(id) %>%
    summarize(ploidy = median(ploidy))
  
  return(roc(true$ploidy, pred$ploidy))
}


#################################################
# Run
#################################################

models <- c("Res50")
subtypes <- c("KIRP")
date <- "20210101"

for(subtype in subtypes){
  for(model in models){
    roc_kirp_test <- create_roc("Ploidy", subtype, model, date, "test")
    
    roclist <- list("test"=roc_kirp_test)
    
    g <- ggroc(roclist, legacy.axes=T, size = 1, alpha=0.9) +
      geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
      scale_color_colorblind() +
      ggtitle(paste("Ploidy", subtype, model, sep=" ")) +
      xlab("False Positive Rate") + 
      ylab("True Positive Rate") +
      theme(legend.position = "bottom") +
      guides(color=guide_legend(title="Model"))
  }
  capture.output(print(roclist),
                 file = paste("./results/Ploidy", subtype, model, date, "_auc.txt",sep=""))
  ggsave(paste("./results/Ploidy", subtype, model, date, "_roc.png",sep=""), g, "png", width=4, height=4)
  
}


# Aggregate to patient level

for(subtype in subtypes){
  for(model in models){
    roc_test <- create_roc_agg(subtype, model, date, "test")
    
    roclist <- list("test"=roc_test)
    
    g <- ggroc(roclist, legacy.axes=T, size = 1, alpha=0.9) +
      geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
      scale_color_colorblind() +
      ggtitle(paste("Ploidy", subtype, model, "Patient-Aggregated", sep=" ")) +
      xlab("False Positive Rate") + 
      ylab("True Positive Rate") +
      theme(legend.position = "bottom") +
      guides(color=guide_legend(title="Model"))
  }
  capture.output(print(roclist),
                 file = paste("./results/Ploidy", subtype, model, date, "_agg_auc.txt",sep=""))
  ggsave(paste("./results/Ploidy", subtype, model, date, "_agg_roc.png",sep=""), g, "png", width=4, height=4)
  ggsave(paste("./results/Ploidy", subtype, model, date, "_agg_roc.pdf",sep=""), g, "pdf", width=4, height=4)
  
}

