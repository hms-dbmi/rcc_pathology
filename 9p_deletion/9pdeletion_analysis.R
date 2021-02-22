# 9pdeletion_analysis.R
# ROC curve creation for CDKN2A prediction
#
# 2020.12.22 Eliana Marostica

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
  
  true_file <- paste("./results/", subtype, task, model, date, "_", split, "True.txt", sep="")
  pred_file <- paste("./results/", subtype, task, model, date, "_", split, "Predictions.txt", sep="")
  
  
  true <- scan(true_file)
  pred <- scan(pred_file)
  
  return(roc(true, pred))
}


create_roc_agg <- function(subtype, model, date, split){
  if(split == "train"){
    filename <- paste("./data/9pdeletion_", str_to_lower(subtype), "_trainShuffled.txt", sep="")
  } else{
    filename <- paste("./data/9pdeletion_", str_to_lower(subtype), "_", split, ".txt", sep="")
  }
  
  ids <- read_delim(filename, delim=" ", col_names = F) %>%
    separate(X1, into=c("Sample_ID", "file"), 15)
  
  true_file <- paste("./results/", subtype, "9pDeletion", model, date, "_", split, "True", ".txt", sep="")
  pred_file <- paste("./results/", subtype, "9pDeletion", model, date, "_", split, "Predictions", ".txt", sep="")
  true <- tibble("id" = ids$Sample_ID, "deletion" = scan(true_file)) %>%
    group_by(id) %>%
    summarize(deletion = median(deletion))
  pred <- tibble("id" = ids$Sample_ID, "deletion" = scan(pred_file)) %>%
    group_by(id) %>%
    summarize(deletion = median(deletion))
  
  return(roc(true$deletion, pred$deletion))
}


#################################################
# Run
#################################################

models <- c("Res50")
subtypes <- c("KIRC")
date <- "20210122"


for(model in models){
  roc_kirc <- create_roc("9pDeletion", "KIRC", model, date, "test")
  roc_kirp <- create_roc("9pDeletion", "KIRP", model, date, "test")
  
  roclist <- list("ccRCC"=roc_kirc, "pRCC"=roc_kirp)
  
  g <- ggroc(roclist, legacy.axes=T, size = 1, alpha=0.9) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    scale_color_colorblind() +
    ggtitle(paste("9pDeletion", subtype, model, sep=" ")) +
    xlab("False Positive Rate") + 
    ylab("True Positive Rate") +
    theme(legend.position = "bottom") +
    guides(color=guide_legend(title="Model"))
}

capture.output(print(roclist),
               file = paste("./results/9pDeletionKIRCKIRP", model, date, "_auc.txt",sep=""))
ggsave(paste("./results/9pDeletionKIRCKIRP", model, date, "_roc.png",sep=""), g, "png", width=4, height=4)
ggsave(paste("./results/9pDeletionKIRCKIRP", model, date, "_roc.pdf",sep=""), g, "pdf", width=4, height=4)










