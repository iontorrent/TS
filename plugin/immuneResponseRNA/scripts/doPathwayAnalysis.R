# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
# This script creates a default heatmap from a given matrix file (table with no row/col headers).

options(warn=1)
library(RColorBrewer)

args <- commandArgs(trailingOnly=TRUE)

rpm_file <- ifelse(is.na(args[1]),"rpm.xls",args[1])
r_util_fun <- ifelse(is.na(args[2]),"utilitFunctioins.R",args[2])
nBarcode <- as.numeric(ifelse(is.na(args[3]),"0",args[3]))
outfile_prefix <- ifelse(is.na(args[4]),"Heatmap",args[4])
##- output 3 heatmaps for interferon signaling pathway, checkpoint pathway, and drug targets
outfile_interferon <- paste(outfile_prefix, '_interferon_heatmap.png', sep='')
outfile_checkpoint <- paste(outfile_prefix, '_checkpoint_heatmap.png', sep='')
outfile_drugtarget <- paste(outfile_prefix, '_drugtarget_heatmap.png', sep='')


source(r_util_fun)

if( !file.exists(rpm_file) ) {
    write(sprintf("ERROR: Could not locate input file %s\n", rpm_file), stderr())
    q(status=1)
}


# read in matrix file and check expected format
data <- read.table(rpm_file, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 1 data column plus row ids in data file %s\n", rpm_file),stderr())
    q(status=1)
}
nrows <- nrow(data)
if( nrows < 1 ) {
    write(sprintf("ERROR: Expected at least 1 row of data plus header line in data file %s\n", rpm_file),stderr())
    q(status=1)
}

# grab row names and strip extra columns
lnames <- data[[1]]
data <- data[,-c(1,2),drop=FALSE]
if( nBarcode > 0 ) {
    data <- data[,1:nBarcode,drop=FALSE]
}
data <- as.matrix(data)
ncols <- ncol(data)
if( ncols < 1 ) {
    write(sprintf("ERROR: Expected at least 1 data column after removing annotation columns in %s\n",nFileIn),stderr())
    q(status=1)
}
rownames(data) <- lnames
data <- log2(data + 1)

# color threshold, fixed at 0 for lowest value
ncolors = 100
maxd <- max(data)
if( maxd <= 0 ) { maxd = 1 }
pbreaks <- seq( 0, maxd, maxd/ncolors )

colors <- colorRampPalette(rev(brewer.pal(name="RdBu",n=8)))

interferon_genes <- c('OAS3',
'IFIT1',
'IFITM1',
'IFITM2',
'ISG15',
'ISG20',
'IFIT3',
'IFIT2',
'CXCL11',
'CXCR5',
'CYBB',
'HLA-B',
'ICAM1',
'IFNB1',
'IL1B',
'IRF1',
'PSMB9',
'STAT1',
'TAP1',
'BCL6',
'CXCL13',
'IFNG',
'TBX21',
'CIITA',
'EIF2AK2',
'FASLG',
'GBP1',
'IRF9',
'OAS1')

checkpoint_genes <- c('C10orf54',
'IDO2',
'TDO2',
'ADORA2A',
'CEACAM1',
'ENTPD1',
'PVR',
'HLA-DQA1',
'BUB1' )

drugtarget_genes <- c('IL2',
'TNFRSF9',
'CD40',
'STAT3',
'TLR9',
'CD70',
'PMEL',
'MS4A1',
'LAG3',
'TNFRSF18',
'TNFRSF4',
'PDCD1',
'CTLA4',
'SLAMF7',
'KIR2DL1',
'KLRD1',
'IL10',
'IL12A',
'IL12B',
'CD27',
'IDO1')


gene_set <- list('interferon' = interferon_genes, 'checkpoint' = checkpoint_genes, 'drugtarget' = drugtarget_genes)
gene_set_tiles <- c('Interferon signaling pathway', 'Checkpoint pathway', 'Drug targets')
outfile <- c( paste(outfile_prefix, '_interferon_heatmap.png', sep=''),
    paste(outfile_prefix, '_checkpoint_heatmap.png', sep=''),
    paste(outfile_prefix, '_drugtarget_heatmap.png', sep='') )


for(i in seq_along(gene_set_tiles)) {
    genes <- gene_set[[i]]
    title <- gene_set_tiles[i]
    keytitle <- 'log2(RPM + 1)'

    use_row <- rownames(data) %in%  genes
    if (sum(use_row) > 2) {
        ncolors = 100
        maxd <- max(data[use_row, ])
        if( maxd <= 0 ) { maxd = 1 }
        pbreaks <- seq( 0, maxd, maxd/ncolors )

        # view needs to scale by number of rows but keeping titles areas at same absolute sizes
        # E.g. at 900 height want lhei=c(1.4,5,0.25,0.9)
        #wid <- 200+50*ncols
        width <- 800
        height <- 200 + 12 * sum(use_row)
        if (height < 600) {
	        height <- 600
        }
        a <- 900 * 1.4 / height
        c <- 900 * 0.25 / height
        d <- 900 * 0.9 / height
        b <- 7.55 - a - c - d

        png(outfile[i], width=width, height=height)
        heatmap_2( data[use_row, ], col=colors, main=title, symkey=FALSE,
            lmat=rbind(c(4,3,0), c(2,1,0)), lwid=c(1,3,0.2), lhei=c(1,3),
            density.info="none", trace="none", breaks=pbreaks, key.abs=TRUE, labRow=rownames(data[use_row, ]),
            key.xlab = "", keysize=1, key.title = keytitle,  margins=c(12,8)
        )
        dev.off()
    }
}



q()
