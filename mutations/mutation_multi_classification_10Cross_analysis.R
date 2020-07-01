# Analyze Mutation (not CNA) Multilabel 10-fold cross-validation results.
#
# 2020.03.30 Eliana Marostica
#


library(tidyverse)
library(purrrlyr)
library(pROC)
library(ggthemes)

subtype <- "KIRP"
model <- "Res50"

genes <- c("MET", "KMT2C", "KMT2D", "SETD2", "FAT1", "BAP1")
agg_test_pred <- data.frame("ptid" = c(), "MET" = c(), "KMT2C" = c(), "KMT2D" = c(), "SETD2" = c(), "FAT1" = c(), "BAP1" = c())
agg_test_true <- data.frame("ptid" = c(), "MET" = c(), "KMT2C" = c(), "KMT2D" = c(), "SETD2" = c(), "FAT1" = c(), "BAP1" = c())


for(foldI in 0:9){
  trainPred <- read_delim(paste("./",subtype, "MutMultiAug310FoldCV", model, "_F", foldI, "_trainPredictions_trans.txt", sep=""), col_names=F, delim=" ", col_types = 'dddddd')
  testPred  <- read_delim(paste("./",subtype, "MutMultiAug310FoldCV", model, "_F", foldI, "_testPredictions_trans.txt", sep=""), col_names=F, delim=" ", col_types = 'dddddd')
  trainTrue <- read_delim(paste("./",subtype, "MutMultiAug310FoldCV", model, "_F", foldI, "_trainTrue_trans.txt", sep=""), col_names=F, delim=" ", col_types = 'dddddd')
  testTrue  <- read_delim(paste("./",subtype, "MutMultiAug310FoldCV", model, "_F", foldI, "_testTrue_trans.txt", sep=""), col_names=F, delim=" ", col_types = 'dddddd')
  
  # aquire patient ids for each image patch in same order
  traintest <- read_delim(paste("./mutNOTcna_", subtype, "_filtered_trainTest10FoldCV.txt", sep=""), col_names=F, delim=" ", col_types = cols()) %>%
    separate(X1, into=c("ptid","path"), ifelse(subtype=="CPTAC", 9, 12)) %>% 
    select(-path)
  test_ptids <- traintest %>%
    filter(X8 == foldI+1)
  test_ptids <- test_ptids$ptid
  stopifnot(nrow(trainPred) == nrow(trainTrue), 
            nrow(testPred) == nrow(testTrue), 
            nrow(trainPred) + nrow(testPred) == nrow(traintest), 
            nrow(trainTrue) + nrow(testTrue) == nrow(traintest),
            nrow(testPred) == length(test_ptids),
            nrow(testTrue) == length(test_ptids))
  
  testPred$ptid <- test_ptids
  testPred <- testPred[,c(7,1,2,3,4,5,6)]
  colnames(testPred) <- c("ptid", genes)
  testTrue$ptid <- test_ptids
  testTrue <- testTrue[,c(7,1,2,3,4,5,6)]
  colnames(testTrue) <- c("ptid", genes)
  
  
  agg_test_pred <- rbind(agg_test_pred, testPred)
  agg_test_true <- rbind(agg_test_true, testTrue)
  
}


# Aggregate to patient-level
agg_true <- agg_test_true %>%
  group_by(ptid) %>%
  summarize(MET = median(MET),
            KMT2C = median(KMT2C),
            KMT2D = median(KMT2D),
            SETD2 = median(SETD2),
            FAT1 = median(FAT1),
            BAP1 = median(BAP1))
agg_pred <- agg_test_pred %>%
  group_by(ptid) %>%
  summarize(MET = median(MET),
            KMT2C = median(KMT2C),
            KMT2D = median(KMT2D),
            SETD2 = median(SETD2),
            FAT1 = median(FAT1),
            BAP1 = median(BAP1))

roclist <- list("MET" = roc(agg_true[["MET"]], agg_pred[["MET"]]),
                "KMT2C" = roc(agg_true[["KMT2C"]], agg_pred[["KMT2C"]]),
                "KMT2D" = roc(agg_true[["KMT2D"]], agg_pred[["KMT2D"]]),
                "SETD2" = roc(agg_true[["SETD2"]], agg_pred[["SETD2"]]),
                "FAT1" = roc(agg_true[["FAT1"]], agg_pred[["FAT1"]]),
                "BAP1" = roc(agg_true[["BAP1"]], agg_pred[["BAP1"]]))
auclist <- round(c("MET" = roc(agg_true[["MET"]], agg_pred[["MET"]])$auc,
                   "KMT2C" = roc(agg_true[["KMT2C"]], agg_pred[["KMT2C"]])$auc,
                   "KMT2D" = roc(agg_true[["KMT2D"]], agg_pred[["KMT2D"]])$auc,
                   "SETD2" = roc(agg_true[["SETD2"]], agg_pred[["SETD2"]])$auc,
                   "FAT1" = roc(agg_true[["FAT1"]], agg_pred[["FAT1"]])$auc,
                   "BAP1" = roc(agg_true[["BAP1"]], agg_pred[["BAP1"]])$auc), digits=3)

genes_aucs <- c()
for (i in seq_along(auclist)){
  genes_aucs <- c(genes_aucs,paste(names(auclist)[i]," (AUC=", auclist[[i]], ")", sep=""))
}
names(roclist) <- genes_aucs
aucdf <- data.frame(auclist)
write_tsv(aucdf, paste("./MutMultiAug310FoldCVAggPatient", subtype, model, "_", "auc.tsv",sep=""), col_names = F)

readable_subtype <- ifelse(subtype=="KICH","Chromophobe RCC", ifelse(subtype=="KIRC", "Clear Cell RCC", "Papillary RCC"))

g <- ggroc(roclist, legacy.axes=T, size=1,  alpha=0.7) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", linetype="dashed") +
  scale_color_colorblind() +
  ggtitle(paste(readable_subtype, model)) +
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  guides(color=guide_legend(title="Gene"))

ggsave(paste("./MutMultiAug310FoldCVAggPatient", subtype, model, "_", "roc.png",sep=""), g, "png", width=7.5, height=6)

