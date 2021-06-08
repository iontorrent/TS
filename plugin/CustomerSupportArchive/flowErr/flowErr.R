
system(sprintf("mkdir -p %s",plotDir))

myPch <- 21:24
names(myPch) <- c("A","C","G","T")
myCol <- c("green","blue","black","red")
names(myCol) <- c("A","C","G","T")

x <- read.table(flowErrFile,header=TRUE,sep="\t",as.is=TRUE)
flow <- 1:length(x$n_aligned)
offset<-max(x$n_aligned)*0.01 #when we drop to 1% of the reads, our scale may be peculiar so discount it
ylim <- 1.05*range(c(0,(x$n_hp_err)/(x$n_aligned+offset)),c(0,0.05))
png(sprintf("%s/%s.allFlows.png",plotDir,analysisName),width=600,height=600)
plot(flow,x$n_hp_err/x$n_aligned,xlab="Flow",ylab="Error Rate",pch=myPch[x$base],bg=myCol[x$base],ylim=ylim)
abline(h=seq(0,1,by=0.025),col="grey",lty=2,lwd=2)
badFlow <- ((x$n_hp_err/x$n_aligned) > 0.05)
if (any(badFlow)) {
  for(i in which(badFlow)) {
    text(i,0.002+(x$n_hp_err/x$n_aligned)[i],as.character(i),adj=c(0.5,0))
  }
}
title(analysisName)
legend("topright",inset=0.01,names(myCol),fill=myCol)
dev.off()

png(sprintf("%s/%s.100Flows.png",plotDir,analysisName),width=600,height=600)
maxFlow <- min(max(flow),100)
xlim <- c(min(flow),maxFlow)
ylim <- 1.05 * range(c(0,(x$n_hp_err/(x$n_aligned+offset))[x$flow <= maxFlow]),c(0,0.05))
plot(flow,x$n_hp_err/x$n_aligned,xlab="Flow",ylab="Error Rate",pch=myPch[x$base],bg=myCol[x$base],xlim=xlim,ylim=ylim)
abline(h=seq(0,1,by=0.01),col="grey",lty=2,lwd=2)
badFlow <- ((x$n_hp_err/x$n_aligned) > 0.05)
if (any(badFlow)) {
  for(i in which(badFlow)) {
    text(i,0.002+(x$n_hp_err/x$n_aligned)[i],as.character(i),adj=c(0.5,0))
  }
}
title(analysisName)
legend("topright",inset=0.01,names(myCol),fill=myCol)
dev.off()

#drop the error table for replotting
write.table(x, sprintf("%s/flowErr.txt",plotDir))

