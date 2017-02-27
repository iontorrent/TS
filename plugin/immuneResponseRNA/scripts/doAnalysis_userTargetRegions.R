# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
# This script creates adefault heatmap from a given matrix file (table with no row/col headers).

options(warn=1)
library(RColorBrewer)
library(ggplot2)

args <- commandArgs(trailingOnly=TRUE)

input_file  <- ifelse(is.na(args[1]), "rpm.bcmatrix.xls", args[1])
output_file <- ifelse(is.na(args[2]), "cluster_heatmap.png", args[2])
r_util_fun <- ifelse(is.na(args[3]), "utilitFunctioins.R", args[3])
user_target_regions <- ifelse(is.na(args[4]), "user_target_regions.bed", args[4])
title    <- ifelse(is.na(args[5]), "Clustering Heatmap", args[5])
keytitle <- ifelse(is.na(args[6]), "log2(RPM+1)", args[6])
n_barcode <- as.numeric(ifelse(is.na(args[7]),"0",args[7]))
group1 <- ifelse(is.na(args[8]), "group1", args[8])
group2 <- ifelse(is.na(args[9]), "group2", args[9])
out_fig2 <- ifelse(is.na(args[10]), "fc_user_targets.png", args[10])

###- source utilitFunctioins
source(r_util_fun)

#######
if( !file.exists(input_file) ) {
    write(sprintf("ERROR: Could not locate input file %s\n",input_file),stderr())
    q(status=1)
}


# read in matrix file and check expected format
data <- read.table(input_file, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 1 data column plus row ids in data file %s\n", input_file), stderr())
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

gene_selected <- getGeneAmpliconFromBED(user_target_regions)
idx_data <- rownames(data) %in% gene_selected[, 'GeneSymbol']

##- Get the rpm for selected genes, and tranform them into log2 scale
data_selected <- log2(data[idx_data, ] + 1)

#----------------------------------------------------------------

# view needs to scale by number of rows but keeping titles areas at same absolute sizes
# E.g. at 900 height want lhei=c(1.4,5,0.25,0.9)
#wid <- 200+50*ncols
fig_width <- 800
fig_height <- 200 + 10 * nrow(data_selected)
if (fig_height < 600) {
	fig_height <- 600
}
a <- 900 * 1.4 / fig_height
c <- 900 * 0.25 / fig_height
d <- 900 * 0.9 / fig_height
b <- 7.55 - a - c - d

hmcol <- colorRampPalette(rev(brewer.pal(name="RdBu",n=8)))

png(output_file, width = fig_width, heigh = fig_height)
#heatmap_2( data_selected, col=hmcol, main=title, symkey=FALSE,
#    lmat=rbind(c(0,3,0),c(2,1,0),c(0,0,0),c(0,4,0)), lwid=c(1,5,0.2), lhei=c(a,b,c,d),
#    density.info="none", trace="none", key.abs=TRUE,
#    key.xlab = "", key.title = keytitle, cexRow=1.2, cexCol=1.5, margins=c(8,9) )
heatmap_2( data_selected, col=hmcol, main=title, symkey=FALSE, cexRow=1.5, cexCol=1.5,
  lmat=rbind(c(4,3,0), c(2,1,0)), lwid=c(0.75,2,1), lhei=c(0.75,2),
  density.info="none", trace="none", key.abs=TRUE,
  key.xlab = "", key.title = keytitle, keysize=1, margins=c(10,8) )
dev.off()

################################################################
##- Fold change, check to see if group is defined
if( ! (group1 == 'group1' || group2 == 'group2') ) {
    g1 <- unlist(strsplit(group1, ','))
    g2 <- unlist(strsplit(group2, ','))
    if (length(g1) > 1 & length(g2) > 1) {
        col_g1 <- which(colnames(data_selected) %in% g1)
        col_g2 <- which(colnames(data_selected) %in% g2)
        fold_change <- data.frame(Gene=factor(rownames(data_selected)),
            Fold_Change = (rowMeans(data_selected[, col_g1]) - rowMeans(data_selected[, col_g2])))
        fold_change$color <- ifelse(fold_change$Fold_Change >= 0, "firebrick1", "steelblue")
        fold_change$hjust <- ifelse(fold_change$Fold_Change >= 0, 1.3, -0.3)
        sorted <- sort(fold_change$Fold_Change, index.return=T)$ix
        fold_change_sorted <- fold_change[sorted, ]
        fold_change_sorted$sorted = 1:nrow(fold_change)

        p <- ggplot(fold_change_sorted, aes(sorted, Fold_Change, label = Gene, hjust = hjust)) +
            geom_text(aes(y = 0, color = color), size=rel(1.5)) +
            geom_bar(stat = "identity", aes(fill = color)) +
            theme(axis.title.y = element_blank(), axis.text.y = element_blank(), legend.position = "none") +
            coord_flip()

        ggsave(filename = out_fig2, width = 4, height = 3)
    }
}

q()
