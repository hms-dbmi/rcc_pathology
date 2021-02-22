# subtype_noMiTF_analysis.R
# Generates ROC curves and calculates one-vs-rest and multiclass AUCs for subtype classfiication. 
#
# 2020.12.14 Eliana Marostica

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
splits <- c("train", "val", "test")

for(model in models){
  for(split in splits){
    
    # read in predictions and true values
    true_file <- paste("./results/SubtypeNoMiTF", model, date, "_", split, "True.txt", sep="")
    pred_file <- paste("./results/SubtypeNoMiTF", model, date, "_", split, "Predictions.txt", sep="")
    
    true <- read_delim(true_file, delim=" ", col_names=F) %>%
      rename("pRCC" = X1,
             "ccRCC" = X2,
             "chRCC" = X3)
    pred <- read_delim(pred_file, delim=" ", col_names=F) %>%
      rename("pRCC" = X1,
             "ccRCC" = X2,
             "chRCC" = X3)
    
    
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
    
    # save auc
    response <- factor(colnames(true)[apply(true,1,which.max)], levels=c("pRCC","ccRCC","chRCC"))
    capture.output(print(roclist),
                   file = paste("./results/SubtypeNoMiTF", model, date, split, "_auc.txt",sep=""))
    capture.output(print(multiclass.roc(response, as.matrix(pred))$auc),
                   file = paste("./results/SubtypeNoMiTF", model, date, split, "_auc.txt",sep=""),
                   append = TRUE)

    
    # save curve
    ggsave(paste("./results/SubtypeNoMiTF", model, date, split, "_roc.png",sep=""), g, "png", dpi=350, units="in", width=4, height=4)
    ggsave(paste("./Figures/SubtypeNoMiTF", model, date, split,"_roc.pdf",sep=""), g, "pdf", units="in", width=4, height=4)
  }
}



