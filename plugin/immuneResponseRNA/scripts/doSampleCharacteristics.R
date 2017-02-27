# Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved
library(RColorBrewer)
library(lattice); library(gplots)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),".rpm.bcmatrix.xls",args[1])
n_barcode <- as.numeric(ifelse(is.na(args[2]),"0",args[2]))
nFileOut <- ifelse(is.na(args[3]),"top5exp.png",args[3])
nFileOut2 <- ifelse(is.na(args[4]),"bot5expt.png",args[4])
nFileOut3 <- ifelse(is.na(args[5]),"housekeeping.png",args[5])
fout_hk_scaled <- ifelse(is.na(args[6]),"housekeeping_normalized.tab",args[6])
title    <- ifelse(is.na(args[7]),"Horizontal Bar Chart",args[7])
xtitle   <- ifelse(is.na(args[8]),"Counts",args[8])
dscale   <- as.numeric(ifelse(is.na(args[9]),"1",args[9]))
r_util_fun <- ifelse(is.na(args[10]), "utilityFunctions.R", args[10])
target_bed_file <- ifelse(is.na(args[11]), "target_bed_file.bed", args[11])
run_report_id <- ifelse(is.na(args[12]), "12345", args[12])

##- source utilitFunctioins
source(r_util_fun)

### Read data
if( !file.exists(nFileIn) ) {
    write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
    q(status=1)
}


# read in matrix file and check expected format
origin_data <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(origin_data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 1 data column plus row ids in data file %s\n", nFileIn), stderr())
    q(status=1)
}
nrows <- nrow(origin_data)
if( nrows < 1 ) {
    write(sprintf("ERROR: Expected at least 1 row of data plus header line in data file %s\n", nFileIn), stderr())
    q(status=1)
}

# grab row names and strip extra columns
lnames <- origin_data[[1]]
rpm_data <- origin_data[, -c(1,2), drop=FALSE]
if( n_barcode > 0 ) {
    rpm_data <- rpm_data[, 1:n_barcode, drop=FALSE]
}
rpm_data <- as.matrix(rpm_data)
ncols <- ncol(rpm_data)
if( ncols < 1 ) {
    write(sprintf("ERROR: Expected at least 1 data column after removing annotation columns in %s\n", input_file), stderr())
    q(status=1)
}
##- rownames(rpm_data) <- lnames
##
##- write rpm_data to RPM file without annotations
##- for CHP file conversion
outfile <- sub('.rpm.bcmatrix.xls', '_rpm_forCHP.tab', nFileIn)
outdata <- log2(rpm_data + 1)
colnames(outdata) <- paste(colnames(outdata), paste('R', run_report_id, sep=''), sep='-')
outdata <- cbind('Target' = origin_data[, 2], outdata)
write.table(outdata, file=outfile, row.names=F, col.names=T, quote=F, sep="\t")

rownames(rpm_data) <- lnames

#####################################################################
##- top x% and bottom x%. Default to top5 and bot5
##- union of genes from each barcode
top5 <- FALSE
bot5 <- FALSE
if (top5) {
    idx_top5 <- c()
    idx_bot5 <- c()
    for(i in seq(ncol(rpm_data))) {
        non_zero <- rpm_data[, i] > 0.0
        low_up = quantile(rpm_data[non_zero, i], probs = c(0.05, 0.95), na.rm = T)
        idx_up <- rpm_data[, i] >=low_up[2]
        idx_low <- rpm_data[, i] <= low_up[1] & non_zero
        if(i == 1) {
            idx_top5 <- idx_up
            idx_bot5 <- idx_low
        } else {
            idx_top5 <- idx_top5 | idx_up
            idx_bot5 <- idx_bot5 | idx_low
        }
    }
    top5exp <- rpm_data[idx_top5, ]
    bot5exp <- rpm_data[idx_bot5, ]
    idx_top5_sorted <- sort(rowMeans(top5exp), index.return=T)$ix
    idx_bot5_sorted <- sort(rowMeans(bot5exp), index.return=T)$ix

    png(nFileOut,width=800,height=600)
    ncolor <- length(idx_top5)
    if (ncolor > 11) {
        ncolor <- 11
    }
    hmcol <- colorRampPalette(rev(brewer.pal(name="RdBu",n=ncolor)))
    levelplot(as.matrix(t(log2(top5exp[idx_top5_sorted,] + 1))), aspect='xy', col.regions = hmcol, scales=list(x=list(rot=90)),
        xlab='Samples', ylab='Genes', main='Highest 5 percent expression (log2(RPM))')
    dev.off()

    ###############################
    ## bottom 5%
    png(nFileOut2,width=800,height=600)
    ncolor <- length(idx_bot5)
    if (ncolor > 11) {
        ncolor <- 11
    }
    hmcol <- colorRampPalette(rev(brewer.pal(name="RdBu",n=ncolor)))
    levelplot(as.matrix(t(log2(bot5exp[idx_bot5_sorted,] + 1))), aspect='xy', col.regions = hmcol, scales=list(x=list(rot=90)),
        xlab='Samples', ylab='Genes', main='Lowest 5 percent expression (log2(RPM))')
    dev.off()
}

###############################
##- Housekeeping genes
#- Get housekeeping genes from target design bed file. If no housekeeping genes are found in the bed file, use
#- these predefined housekeeping genes.
hkgene_predefined <- c("ABCF1", "G6PD", "GUSB", "HMBS", "LMNA", "LRP1", "POLR2A", "SDHA", "TBP", "TFRC", "TUBB")
#housename <- sapply(housekeeping, function(zz) {unlist(strsplit(zz, ' '))[1]})
##- The function getHKgenesFromBED return a dataframe with 2 columns: GeneSymbol and AmpliconID
hkgenes <- getHKgenesFromBED(target_bed_file)[, 'GeneSymbol']
if (length(hkgenes) < 3) {
    hkgenes <- hkgene_predefined
}

idx_housekeeping <- match(hkgenes, rownames(rpm_data))
idx_housekeeping2 <- idx_housekeeping[!is.na(idx_housekeeping)]
hkgenes2 <- hkgenes[!is.na(idx_housekeeping)]

## plot if housekeeping genes are found
if(length(idx_housekeeping2) > 1) {
    plot_data <- t(log2(rpm_data[idx_housekeeping2, ] + 1))
    sample_names <- rownames(plot_data)
    if (n_barcode < 2) {
        sample_names <- colnames(rpm_data)
    }

    if (n_barcode < 2) {
        dat2 <- data.frame('Genes'=hkgenes2, 'log2RPM'=log2(rpm_data[idx_housekeeping2, ] + 1), 'SampleName'=colnames(rpm_data))
    } else {
        dat2 <- reshape(data.frame(plot_data), direction='long', varying=colnames(plot_data), idvar='SampleName',
            timevar='Genes', v.names='log2RPM', times=colnames(plot_data), ids=rownames(plot_data))
        dat2$SampleName <- factor(dat2$SampleName, levels=sample_names)
    }

    p_colors <- c("#3F3F3F",brewer.pal(10,"Paired"), 'cyan', 'green', 'blue', 'brown', 'orange')[1:nrow(plot_data)]
    #p_colors <- c("#3F3F3F",brewer.pal(10,"Set3"), 'cyan', 'green', 'blue', 'brown', 'orange')[1:nrow(plot_data)]

    ##- if samples names exceed 15 characters, put the legend under, otherwise, at right.
    max_char_sample_name <- 15
    for(i in seq(along=sample_names)) {
        if (max_char_sample_name < nchar(sample_names[i])) {
            max_char_sample_name <- nchar(sample_names[i])
        }
    }

    num_column <- length(sample_names) %/% 5 + 1

    png(file=nFileOut3, width=800, height=600)
    if (max_char_sample_name  < 15) {
        print(barchart(log2RPM ~ Genes, dat2, groups=SampleName,
            auto.key = list(space = 'right', columns=num_column),
            par.settings = simpleTheme(col=p_colors),
            scales=list(x=list(rot=45)),
            main='House Keeping Genes')
        )
    } else {
        print(barchart(log2RPM ~ Genes, dat2, groups=SampleName,
            auto.key = list(space = 'bottom', columns=num_column),
            par.settings = simpleTheme(col=p_colors),
            scales=list(x=list(rot=45)),
            main='House Keeping Genes')
        )
    }
    dev.off()
}

#####################################
##- Perform scaling with housekeeping genes
#- Take geometic means of housekeeping genes.
#- The RPM is normalized counts, but not log2 transformed. Need to add 1 to the counts.
#- Originally we used RPM for housekeeping normalization.
####- log2(RPM+1) - log2(HK geometric mean of RPM+1)
#- We now use read count for housekeeping normalization
#- Prefer the new calculation:
####- log2(count+1) - log2(HK geometric mean of count+1) + log2(10^6)

#- The nFileIn variable is file for RPM, not count input data. It has xxx.rpm.bcmatrix.xls extension. The file we need is
#- xxx.bcmatrix.xls
count_file <- sub('.rpm.bcmatrix.xls', '.bcmatrix.xls', nFileIn)

### Read data
if( !file.exists(count_file) ) {
    write(sprintf("ERROR: Could not locate input file %s\n", count_file), stderr())
    q(status=1)
}

# read in matrix file and check expected format
count_data <- read.table(count_file, header=TRUE, sep="\t", as.is=TRUE, check.names=F, comment.char="")
ncols <- ncol(count_data)
if( ncols < 2 ) {
    write(sprintf("ERROR: Expected at least 1 data column plus row ids in data file %s\n", count_file), stderr())
    q(status=1)
}
nrows <- nrow(count_data)
if( nrows < 1 ) {
    write(sprintf("ERROR: Expected at least 1 row of data plus header line in data file %s\n", count_file), stderr())
    q(status=1)
}

# grab row names and strip extra columns
lnames <- count_data[[1]]
count_data2 <- count_data[, -c(1,2), drop=FALSE]
if( n_barcode > 0 ) {
    count_data2 <- count_data2[, 1:n_barcode, drop=FALSE]
}
count_data2 <- as.matrix(count_data2)
ncols <- ncol(count_data2)
if( ncols < 1 ) {
    write(sprintf("ERROR: Expected at least 1 data column after removing annotation columns in %s\n", count_file), stderr())
    q(status=1)
}

rownames(count_data2) <- lnames

if (n_barcode < 2) {
    geo_mean_log2 <- mean(log2(count_data2[idx_housekeeping2, ] + 1))
} else {
    geo_mean_log2 <- colMeans(log2(count_data2[idx_housekeeping2, ] + 1))
}
scaled_log2_count <- t( log2( t(count_data2 + 1) ) - geo_mean_log2 + log2(10^6))


##- Add run report_id to the sample names to distinguish duplicate samples names from multiple runs.
colnames(scaled_log2_count) <- paste(colnames(scaled_log2_count), paste('R', run_report_id, sep=''), sep='-')
out_scaled_log2_count <- cbind(count_data[, 1:2], scaled_log2_count, count_data[, (n_barcode+3):ncol(count_data)])
write.table(out_scaled_log2_count, file=fout_hk_scaled, row.names=F, col.names=T, quote=F, sep="\t")

#############
##- Output only nornalized values for CHP converter with amplicon_id as the first column
out_chp <- data.frame('Target' = count_data[, 2], scaled_log2_count)
outfile <- sub('_count.xls', '.forCHP.tab', fout_hk_scaled)
write.table(out_chp, file=outfile, row.names=F, col.names=T, quote=F, sep="\t")



q()
