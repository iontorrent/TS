# Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved
# This script performs t test between 2 groups and generate 'volvano' plot.

options(warn=1)

args <- commandArgs(trailingOnly=TRUE)

input_file  <- ifelse(is.na(args[1]), "rpm.bcmatrix.xls", args[1])
fig_file <- ifelse(is.na(args[2]), "volcano_plot.png", args[2])
r_util_fun <- ifelse(is.na(args[3]), "utilitFunctioins.R", args[3])
user_target_regions <- ifelse(is.na(args[4]), "user_target_regions.bed", args[4])
title    <- ifelse(is.na(args[5]), "Clustering Heatmap", args[5])
n_barcode <- as.numeric(ifelse(is.na(args[6]),"0",args[6]))
group1 <- ifelse(is.na(args[7]), "group1", args[7])
group2 <- ifelse(is.na(args[8]), "group2", args[8])
data_file <- ifelse(is.na(args[9]), "fold_change_p_value.xls", args[9])

###- source utilitFunctioins
source(r_util_fun)



###############################################################################
doFCTplot <- function(log2rpm, group1, group2, fig_file, data_file, top_gene_label = FALSE) {
  fct <- t(apply(log2rpm, 1, function(x){
    pvalue <- t.test(x[group1], x[group2], na.rm = T)$p.value
    fcvalue <- mean(x[group1], na.rm = T) - mean(x[group2], na.rm = T)
    return(c(pvalue, fcvalue))
  }))
  colnames(fct) <- c('pvalue', 'FoldChange_log2')

  write.csv(fct, file=data_file)

  fc_sorted <- sort(fct[, 'FoldChange_log2'], index.return=T)
  idx_top_fc <- c(fc_sorted$ix[1:5], fc_sorted$ix[(nrow(fct)-4) : nrow(fct)])

  png(file=fig_file, width=800, height=600)
  plot(fct[-idx_top_fc, 2], -log10(fct[-idx_top_fc, 1]), pch = 16, col = "blue",
       xlim = c(min(fct[, 2]) - 0.1, max(fct[, 2] + 0.1)), ylim=c(-0.05, max(-log10(fct[, 1]), na.rm=T) + 0.1),
       xlab = paste("log2(FC)"), ylab = "p values ( -log10(p) )",
       main = title
  )
  abline(v=c(-1,1))
  abline(h=5)

  ## label top 5 genes with higest fold change
  if (top_gene_label) {
    fc_sorted <- sort(fct[, 'FoldChange_log2'], index.return=T)
    idx_gene <- fc_sorted$ix[1:5]
    #gene_name <- sapply(rownames(fct)[idx_gene], function(x) {unlist(strsplit(x, ' '))[1]})
    gene_name <- rownames(fct)[idx_gene]
    points(fct[idx_gene, 2], -log10(fct[idx_gene, 1]), pch=21:25, col=1:5, bg=1:5, cex=1.2)
    legend('bottomleft', legend=gene_name, pch=21:25, col=1:5, pt.bg=1:5)
    idx_gene <- fc_sorted$ix[(nrow(fct)-4) : nrow(fct)]
    #gene_name <- sapply(rownames(fct)[idx_gene], function(x) {unlist(strsplit(x, ' '))[1]})
    gene_name <- rownames(fct)[idx_gene]
    points(fct[idx_gene, 2], -log10(fct[idx_gene, 1]), pch=21:25, col=1:5, bg=1:5, cex=1.2)
    legend('bottomright', legend=gene_name, pch=21:25, col=1:5, pt.bg=1:5)
  }
  else {
    points(fct[idx_top_fc, 2], -log10(fct[idx_top_fc, 1]), pch = 16, col = "blue")
  }
  dev.off()
}
###############################################################################

##- read table data
if( !file.exists(input_file) ) {
    write(sprintf("ERROR: Could not locate input file %s\n",input_file),stderr())
    q(status=1)
}


# read in rpm matrix file and check expected format
data <- read.table(input_file, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 2 data column plus row ids in data file %s\n", input_file), stderr())
    q(status=1)
}
nrows <- nrow(data)
if( nrows < 1 ) {
    write(sprintf("ERROR: Expected at least 1 row of data plus header line in data file %s\n", input_file), stderr())
    q(status=1)
}

# grab row names and strip extra columns
lnames <- data[[1]]
data <- data[, -c(1,2), drop=FALSE]
if( n_barcode > 0 ) {
    data <- data[, 1:n_barcode, drop=FALSE]
}
data <- as.matrix(data)
ncols <- ncol(data)
if( ncols < 1 ) {
    write(sprintf("ERROR: Expected at least 1 data column after removing annotation columns in %s\n", input_file), stderr())
    q(status=1)
}
rownames(data) <- lnames

##########################################################
##- run analysis
idx_group1 <- colnames(data) %in% unlist(strsplit(group1, ','))
idx_group2 <- colnames(data) %in% unlist(strsplit(group2, ','))

doFCTplot(log2(data + 1), idx_group1, idx_group2, fig_file, data_file, top_gene_label=T)

q()
