# cna_binary_classification_analysis.R
# Calculates AUC's for CNA binary classification and saves to file.
#
# 2020.05.06. Eliana Marostica 

library(tidyverse)
library(pROC)

subtypes <- c("KIRC", "KIRP")
models <- c("VGG16", "IncV3", "Res50")
genes <- c("VHL", "FH", "FLCN", "SDHB", "SDHD", "MET", "TSC1", "TSC2")

s <- c()
m <- c()
g <- c()
aucs <- c()
for(subtype in subtypes){
  for(model in models){
    for(gene in genes){
      s <- c(s,subtype)
      m <- c(m,model)
      g <- c(g,gene)
      
      true_file <- paste("./", subtype, "CNABinary", gene, model, "_testTrue.txt", sep="")
      pred_file <- paste("./", subtype, "CNABinary", gene, model, "_testPredictions.txt", sep="")
      
      true <- as.numeric(scan(true_file))
      
      pred <- as.numeric(scan(pred_file))
      
      aucs <- c(aucs,round(roc(true, pred)$auc, digits = 3))
    }
  }
}

aucs <- tibble("Subtype"=s, "Model"=m, "Gene"=g, "AUC"=aucs)

aucs <- aucs %>%
  mutate(Subtype = case_when(Subtype == "KIRP" ~ "pRCC",
                             Subtype == "KIRC" ~ "ccRCC"),
         Model = case_when(Model == "IncV3" ~ "Inception v3",
                           Model == "VGG16" ~ "VGG-16",
                           Model == "Res50" ~ "ResNet-50")) %>%
  unite(Subtype_Model, Subtype, Model) 

# order x axis 
aucs$Subtype_Model <- factor(aucs$Subtype_Model, levels=unique(aucs$Subtype_Model))

# plot
g <- ggplot(aucs, aes(Subtype_Model, Gene, fill=AUC)) +
  geom_tile() +
  geom_text(aes(label=AUC)) +
  theme_minimal() +
  #theme(axis.text.x = element_text(angle=30,size=10,hjust=1)) +
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  scale_fill_continuous(type="viridis") 

ggsave("./cna_pancancer_gene_aucs_heatmap.png", g, "pdf", width=7.5, height=6)

