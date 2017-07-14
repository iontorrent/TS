# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Calculate the Insert composition Bias Score, given amplicon summary file and optional
# low rep threshold (0.2) and nomalizaion factor (0.1)
# Note: default normalization is 10x just so number can conviently be represented at 2 decimal places

args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"fine_coverage.xls",args[1])
if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

thresh <- as.numeric(ifelse(is.na(args[2]),"0.2",args[2]))
q_norm <- as.numeric(ifelse(is.na(args[3]),"0.1",args[3]))

rcov <- read.table(nFileIn, header=TRUE, as.is=TRUE, sep="\t", comment.char="", quote="")
rep <- rcov$total_reads

tlen <- rcov$contig_end - rcov$contig_srt + 1
pcgc <- 100 * rcov$gc / tlen

# remove zero's option
remove0s = FALSE
if( remove0s ) {
  pcgc <- pcgc[rep>0]
  rep <- rep[rep>0]
}

ndata <- length(rep)
if( ndata < 2 )
{
  write(sprintf("ERROR: No coverage property field found in data file %s\n",nFileIn),stderr())
  q(status=1)
}
if( ndata < 4 )
{
  write(0,stdout())
  q(status=0)
}

quartiles <- quantile(pcgc) 
Q1 <- quartiles[[2]]
Q4 <- quartiles[[4]]

rep <- rep / mean(rep)
q1d <- rep[pcgc<=Q1]
qmd <- rep[(pcgc>=Q1)&(pcgc<=Q4)]
q4d <- rep[pcgc>=Q4]

fq1d = sum(q1d<thresh) / length(q1d)
fqmd = sum(qmd<thresh) / length(qmd)
fq4d = sum(q4d<thresh) / length(q4d)
rm1 = fq1d - fqmd
rm4 = fq4d - fqmd

#icbs = 0.5*(abs(rm1)+abs(rm4))/sd(rep)
icbs = sqrt(0.5*(rm1*rm1+rm4*rm4))/sd(rep)
write(sprintf("%.3f",icbs/q_norm),stdout())
quit(status=0)
