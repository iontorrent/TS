# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# Create scatterplots of flow vs. score.

library(geneplotter)

t10   = read.table('trim.10.out', col.names=c('n','q','a','s','h'))
ok    = which(0<t10$a & t10$a<250)
xlab  = 'first flow of P1'
ylab  = 'euclidead distance to ideal P1'
title = basename(getwd())

png("SFFTrim.smooth.png", width=600, height=600, title=title)
smoothScatter(t10$a[ok], t10$s[ok], xlab=xlab, ylab=ylab, main=title)
dev.off()

png("SFFTrim.detail.png", width=600, height=600, title=title)
plot(t10$a[ok], t10$s[ok], col=rgb(0,0,1,1/255), xlab=xlab, ylab=ylab, main=title)
dev.off()

