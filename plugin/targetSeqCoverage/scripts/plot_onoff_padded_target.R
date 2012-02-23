# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)

nFileIn <- ifelse(is.na(args[1]),"coverage_onoff_target.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"coverage_onoff_target.png",args[2])

if( !file.exists(nFileIn) )
{
     write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
     q()
}
ccov <- read.table(nFileIn, header=TRUE)

if( length(ccov$Mapped_Reads) == 0 )
{
     write(sprintf("No data output to %s\n",nFileOut),stderr())
     q()
}

# check for ojn/off target presence in data
onTarg = ccov$On_Target
haveTarg = !is.null(onTarg)

ylim = max(ccov$Mapped_Reads)
b = 10^(as.integer(log10(ylim)))
ylim = b * as.integer(1 + ylim/b)

png(nFileOut,width=800,height=800)
par(mfrow=c(1,1),bty = 'n',mar=(c(5,5,4,1)+0.1))
if( haveTarg )
{
     targCover = 100*sum(onTarg) / sum(ccov$Mapped_Reads)
     barplot( t(cbind(onTarg,ccov$Mapped_Reads-onTarg)), beside=FALSE,
          names.arg=ccov$Chromosome, yaxs = 'i', col=c("grey","white"), ylim=c(0,ylim),
          xlab="Chomosome", ylab="Number of Reads",
          main=sprintf("Padded Target Coverage by Chromosome\n   %.2f%% of mapped reads on target",targCover) )
} else
{
     barplot( ccov$Mapped_Reads, names.arg=ccov$Chromosome, yaxs = 'i', ylim=c(0,ylim),
          xlab="Chomosome", ylab="Number of Reads",
          main="Padded Target Coverage by Chromosome" )
}

q()

