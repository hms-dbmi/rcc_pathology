# malignancy_subtype_analysis.R
# ROC curve creation for malignancy and subtype tasks
# Includes CPTAC analysis for these tasks
#
# 2019.12.02 Eliana Marostica

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
create_roc <- function(task, subtype, model, cptac=F){
  
  if(cptac){
    true_file <- paste("./results/", subtype, task, model, "Talos_cptac_True.txt", sep="")
    pred_file <- paste("./results/", subtype, task, model, "Talos_cptac_Predictions.txt", sep="")
  } else{
    true_file <- paste("./results/", subtype, task, model, "Talos_test_True.txt", sep="")
    pred_file <- paste("./results/", subtype, task, model, "Talos_test_Predictions.txt", sep="")
  }
  
  
  true <- scan(true_file)
  pred <- scan(pred_file)
  
  return(roc(true, pred))
}


#' Given the prediction task and the renal cell carcinoma subtype, create a ROC curve and save it.
#' 
#' @param task The prediction task.
#' @param subtype The renal cell carcinoma subtype.
#' @examples 
#' save_rocplot("TumorNormal", "KIRC")
save_rocplot <- function(task, subtype, cptac=F) {
  ## calculate roc for vgg16, incv3, and res50
  roc_VGG16 <- create_roc(task, subtype, "VGG16", cptac)
  roc_IncV3 <- create_roc(task, subtype, "IncV3", cptac)
  roc_Res50 <- create_roc(task, subtype, "Res50", cptac)
  
  ## plot curve
  readable_subtype <- ifelse(subtype=="KICH","Chromophobe RCC", ifelse(subtype=="KIRC", "Clear Cell RCC", "Papillary RCC"))
  
  roclist <- list("ResNet-50"=roc_Res50, 
                  "Inception v3"=roc_IncV3,
                  "VGG-16"=roc_VGG16)
  
  g <- ggroc(roclist, legacy.axes=T, size = 1, alpha=0.9) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    #theme_minimal() +
    #scale_color_viridis(discrete = TRUE, option = "D", name="Model") +
    scale_color_colorblind() +
    #guides(fill=guide_legend(title="Neural Network Architecture")) +
    ggtitle(paste(readable_subtype, ifelse(cptac,"CPTAC",""))) +
    xlab("False Positive Rate") + 
    ylab("True Positive Rate") +
    theme(legend.position = "bottom") +
    guides(color=guide_legend(title="Model")) #+
    # annotate("text", x=0.75, y=0.25,
    #          label=paste(paste("ResNet-50 (AUC ", round(roc_Res50$auc, digits=3), ")",sep=""),
    #                      paste("Inception v3 (AUC ", round(roc_IncV3$auc, digits=3), ")",sep=""),
    #                      paste("VGG-16 (AUC ", round(roc_VGG16$auc, digits=3), ")",sep=""), sep="\n"))
  
  ## save curve
  
  ggsave(paste("./results/",task, subtype, ifelse(cptac, "_cptac", ""), "_roc.png",sep=""), g, "png", width=4, height=4)
}




#################################################
# Run
#################################################

save_rocplot("TumorNormal", "KICH")
save_rocplot("TumorNormal", "KIRC")
save_rocplot("TumorNormal", "KIRP")

save_rocplot("TumorNormal", "KIRC", cptac=T)
