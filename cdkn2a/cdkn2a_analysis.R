# cdkn2a_analysis.R
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
    filename <- paste("./data/cdkn2a_", str_to_lower(subtype), "_trainShuffled.txt", sep="")
  } else{
    filename <- paste("./data/cdkn2a_", str_to_lower(subtype), "_", split, ".txt", sep="")
  }
  
  ids <- read_delim(filename, delim=" ", col_names = F) %>%
    separate(X1, into=c("Sample_ID", "file"), 15)
  
  true_file <- paste("./results/", subtype, "CDKN2A", model, date, "_", split, "True", ".txt", sep="")
  pred_file <- paste("./results/", subtype, "CDKN2A", model, date, "_", split, "Predictions", ".txt", sep="")
  true <- tibble("id" = ids$Sample_ID, "cdkn2a" = scan(true_file)) %>%
    group_by(id) %>%
    summarize(cdkn2a = median(cdkn2a))
  pred <- tibble("id" = ids$Sample_ID, "cdkn2a" = scan(pred_file)) %>%
    group_by(id) %>%
    summarize(cdkn2a = median(cdkn2a))
  
  return(roc(true$cdkn2a, pred$cdkn2a))
}



#################################################
# Run
#################################################

models <- c("Res50")
date <- "20201221"

# Plot ccRCC and pRCC curves on same plot

for(model in models){
  roc_kirc <- create_roc_agg("KIRC", model, date, "test")
  roc_kirp <- create_roc_agg("KIRP", model, date, "test")
  
  roclist <- list("ccRCC"=roc_kirc, "pRCC"=roc_kirp)
  
  g <- ggroc(roclist, legacy.axes=T, size = 1, alpha=0.9) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    scale_color_colorblind() +
    ggtitle(paste("CDKN2A", subtype, model, "Patient-Aggregated", sep=" ")) +
    xlab("False Positive Rate") + 
    ylab("True Positive Rate") +
    theme(legend.position = "bottom") +
    guides(color=guide_legend(title="Model"))
}
capture.output(print(roclist),
               file = paste("./results/CDKN2A", subtype, model, date, "_agg_auc.txt",sep=""))
ggsave(paste("./results/CDKN2A", subtype, model, date, "_agg_roc.png",sep=""), g, "png", width=4, height=4)
ggsave(paste("./results/CDKN2A", subtype, model, date, "_agg_roc.pdf",sep=""), g, "pdf", width=4, height=4)




