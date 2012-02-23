# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
args <- commandArgs(trailingOnly=TRUE)
nFileIn <- ifelse(is.na(args[1]),"map_onoff_target.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"map_onoff_target.png",args[2])

# read in data
if( !file.exists(nFileIn) )
{
     write(sprintf("Could not locate input file %s\n",nFileIn),stderr())
     q()
}
mydata <- read.table(nFileIn, header=TRUE)
chr.unique = unique(mydata$Chromosome)

# define multiplot layout
numplot = length(chr.unique)
if( numplot == 0 ) {
     write(sprintf("No data output to %s\n",nFileOut),stderr())
     q()
}
ncol = floor(sqrt(numplot))
nrow = ceiling(numplot/ncol)
if( ncol == 2 && nrow == 4 ) {
     ncol = 3
     nrow = 3
}
wd = ifelse(ncol==1,640,ifelse(ncol<4,ncol*320,1280))
ht = ifelse(nrow==1,320,ifelse(nrow<6,nrow*160,960))
png(nFileOut,width=wd,height=ht)
par(mfrow=c(nrow,ncol))

# this gives common height based of maximum hits across all chromosomes
ymax <- max(mydata$Count)
b <- 10^(as.integer(log10(ymax))-1)
ymax <- b * (as.integer(ymax/b)+1)
if( ymax < 10 ) ymax = 10

colm = c( "#000000","#FF0000", "#909090", "#5070F0" )
par(bty = 'n',mar=c(2,3,4,1),oma = c(0,0,3,0))
for( chr in chr.unique )
{
     sel = (mydata$Chromosome == chr)
     pos <- mydata$End[sel]
     cnt <- mydata$Count[sel]
     col <- colm[mydata$On_Target[sel]+1]
     
     # use 10x zoom if data woould disappear on common axis
     ymx = ymax
     title = chr
     if( ymax > 1000 && max(cnt) < 0.1*ymax )
     {
	ymx = as.integer(0.1*ymax)
        title = sprintf("%s (10x zoom)",chr)
     }
     barplot( cnt, names.arg=pos, ylim=c(0,ymx), xaxs = 'i', yaxs = 'i', border = NA, space = 0,
          axisnames = FALSE, main=title, col=col, ann=FALSE, xaxt='n' )    
}
mtext("On/Off Target Coverage (5+ Starts/100b)", outer = TRUE, cex = 1.5)

q()
