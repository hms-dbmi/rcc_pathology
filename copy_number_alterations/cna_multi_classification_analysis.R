# cna_multi_classification_analysis.R
# Creates ROC curves for CNA multi-label classification
#
# 2020.05.12. Eliana Marostica


library(tidyverse)
library(pROC)
library(ggthemes)
library(viridis)
library(ggrepel)
library(randomcoloR)


# parameters
subtype <- "KIRC"
genes <- c("VHL", "FH", "FLCN", "SDHB", "SDHD", "MET", "EGFR", "KRAS", "MYC", "BCL2", "AKT2", "TSC1", "TSC2", "TP53", "RB1", "PTEN", "NF1", "NF2", "WT1")
agg <- TRUE
date <- "20200506"

for(model in c("VGG16", "IncV3", "Res50")){
  roc_outpath <- paste("./", subtype, "CNAMultiLabelDataAug", model, date, ifelse(agg,"Aggregated",""), "_testROC.png",sep="")
  auc_outpath <- paste("./",subtype, "CNAMultiLabelDataAug", model, date, ifelse(agg,"Aggregated",""), "_testAUC.tsv",sep="")
  true_file <- paste("./", subtype, "CNAMultiLabelDataAug", model, date, "_testTrue.txt", sep="")
  pred_file <- paste("./", subtype, "CNAMultiLabelDataAug", model, date, "_testPredictions.txt", sep="")
  ptids <- paste("./cna_", subtype, "_test2.txt", sep="")
  
  true <- read_delim(true_file, delim=" ", col_names=F)
  colnames(true) <- genes
  head(true)
  
  pred <- read_delim(pred_file, delim=" ", col_names=F)
  colnames(pred) <- genes
  head(pred)
  
  input <- read_delim(ptids, col_names=F, delim=" ") %>%
    separate(X1, into=c("ptid", "path"), sep=12)
  
  stopifnot(sum(input[,3:ncol(input)] != true) == 0)
  
  ptids <- input$ptid
  
  if(agg){
    true <- true %>%
      mutate(ptid = ptids) %>%
      group_by(ptid) %>%
      summarize("VHL" = median(VHL), 
                "FH" = median(FH), 
                "FLCN" = median(FLCN), 
                "SDHB" = median(SDHB), 
                "SDHD" = median(SDHD), 
                "MET" = median(MET), 
                "EGFR" = median(EGFR), 
                "KRAS" = median(KRAS), 
                "MYC" = median(MYC), 
                "BCL2" = median(BCL2), 
                "AKT2" = median(AKT2), 
                "TSC1" = median(TSC1), 
                "TSC2" = median(TSC2), 
                "TP53" = median(TP53), 
                "RB1" = median(RB1), 
                "PTEN" = median(PTEN), 
                "NF1" = median(NF1), 
                "NF2" = median(NF2), 
                "WT1" = median(WT1))
    pred <- pred %>%
      mutate(ptid = ptids) %>%
      group_by(ptid) %>%
      summarize("VHL" = median(VHL), 
                "FH" = median(FH), 
                "FLCN" = median(FLCN), 
                "SDHB" = median(SDHB), 
                "SDHD" = median(SDHD), 
                "MET" = median(MET), 
                "EGFR" = median(EGFR), 
                "KRAS" = median(KRAS), 
                "MYC" = median(MYC), 
                "BCL2" = median(BCL2), 
                "AKT2" = median(AKT2), 
                "TSC1" = median(TSC1), 
                "TSC2" = median(TSC2), 
                "TP53" = median(TP53), 
                "RB1" = median(RB1), 
                "PTEN" = median(PTEN), 
                "NF1" = median(NF1), 
                "NF2" = median(NF2), 
                "WT1" = median(WT1))
  }
  
  roclist <- list("VHL" = roc(true[["VHL"]],pred[["VHL"]], ci = T), 
                  "FH" = roc(true[["FH"]],pred[["FH"]], ci = T), 
                  "FLCN" = roc(true[["FLCN"]],pred[["FLCN"]], ci = T), 
                  "SDHB" = roc(true[["SDHB"]],pred[["SDHB"]], ci = T), 
                  "SDHD" = roc(true[["SDHD"]],pred[["SDHD"]], ci = T), 
                  "MET" = roc(true[["MET"]],pred[["MET"]], ci = T), 
                  "EGFR" = roc(true[["EGFR"]],pred[["EGFR"]], ci = T), 
                  "KRAS" = roc(true[["KRAS"]],pred[["KRAS"]], ci = T), 
                  "MYC" = roc(true[["MYC"]],pred[["MYC"]], ci = T), 
                  "BCL2" = roc(true[["BCL2"]],pred[["BCL2"]], ci = T), 
                  "AKT2" = roc(true[["AKT2"]],pred[["AKT2"]], ci = T), 
                  "TSC1" = roc(true[["TSC1"]],pred[["TSC1"]], ci = T), 
                  "TSC2" = roc(true[["TSC2"]],pred[["TSC2"]], ci = T), 
                  "TP53" = roc(true[["TP53"]],pred[["TP53"]], ci = T), 
                  "RB1" = roc(true[["RB1"]],pred[["RB1"]], ci = T), 
                  "PTEN" = roc(true[["PTEN"]],pred[["PTEN"]], ci = T), 
                  "NF1" = roc(true[["NF1"]],pred[["NF1"]], ci = T), 
                  "NF2" = roc(true[["NF2"]],pred[["NF2"]], ci = T), 
                  "WT1" = roc(true[["WT1"]],pred[["WT1"]], ci = T))
  
  auclist <- round(c("VHL" = roc(true[["VHL"]],pred[["VHL"]])$auc, 
                     "FH" = roc(true[["FH"]],pred[["FH"]])$auc, 
                     "FLCN" = roc(true[["FLCN"]],pred[["FLCN"]])$auc, 
                     "SDHB" = roc(true[["SDHB"]],pred[["SDHB"]])$auc, 
                     "SDHD" = roc(true[["SDHD"]],pred[["SDHD"]])$auc, 
                     "MET" = roc(true[["MET"]],pred[["MET"]])$auc, 
                     "EGFR" = roc(true[["EGFR"]],pred[["EGFR"]])$auc, 
                     "KRAS" = roc(true[["KRAS"]],pred[["KRAS"]])$auc, 
                     "MYC" = roc(true[["MYC"]],pred[["MYC"]])$auc, 
                     "BCL2" = roc(true[["BCL2"]],pred[["BCL2"]])$auc, 
                     "AKT2" = roc(true[["AKT2"]],pred[["AKT2"]])$auc, 
                     "TSC1" = roc(true[["TSC1"]],pred[["TSC1"]])$auc, 
                     "TSC2" = roc(true[["TSC2"]],pred[["TSC2"]])$auc, 
                     "TP53" = roc(true[["TP53"]],pred[["TP53"]])$auc, 
                     "RB1" = roc(true[["RB1"]],pred[["RB1"]])$auc, 
                     "PTEN" = roc(true[["PTEN"]],pred[["PTEN"]])$auc, 
                     "NF1" = roc(true[["NF1"]],pred[["NF1"]])$auc, 
                     "NF2" = roc(true[["NF2"]],pred[["NF2"]])$auc, 
                     "WT1" = roc(true[["WT1"]],pred[["WT1"]])$auc), digits=3)
  
  # rename roclist for more informative legend labels (incl. AUC for each gene)
  genes_aucs <- c()
  for (i in seq_along(auclist)){
    genes_aucs <- c(genes_aucs,paste(names(auclist)[i]," (AUC=", auclist[[i]], ")", sep=""))
  }
  names(roclist) <- genes_aucs
  
  #aucdf <- data.frame(auclist)
  #write_tsv(aucdf, auc_outpath, col_names = F)
  
  readable_subtype <- ifelse(subtype=="KICH","Chromophobe RCC", ifelse(subtype=="KIRC", "Clear Cell RCC", "Papillary RCC"))
  
  # Generate distinct color palette for 13 different genes
  palette <- distinctColorPalette(19)
  
  # plot
  g <- ggroc(roclist, legacy.axes=T, size=1,  alpha=0.7) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
    scale_color_manual(values=unname(palette)) +
    ggtitle(paste(readable_subtype, model)) +
    xlab("False Positive Rate") + 
    ylab("True Positive Rate") +
    guides(color=guide_legend(title="Gene"))
  
  ggsave(roc_outpath, g, "pdf", width=7.5, height=6)
  
}

