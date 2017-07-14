
#run a custom bubbleplots
library(torrentR)
library(ggplot2);
library(Hmisc)

v=Sys.getenv("TSP_LIMIT_OUTPUT",unset=NA)

#this time, we'll subsample the chips and the reads to make this fast!
basemaxFlows<-1200 #upgrade timing to encompass long runs
sampleFlowsBy<-3 #step through 3  at a time because we like fine resolution - override this later to make finite samples
#sample levels

#first check to see if we have any dat files at all
#then do the right number of flows
testFlow<-Sys.glob(sprintf("%s/acq_*.dat",datDir))
maxFlows<-min(basemaxFlows,length(testFlow)-1)

if(is.na(v)){
    #sampleFlowsBy update
    sampleFlowsBy<-floor(maxFlows/35+1) # about 35 samples for a full run
} else {
    sampleFlowsBy<-floor(maxFlows/5+1) # reduce to 5 samples for smaller/faster processing
}

#nothing to do here
if (maxFlows<1){quit(save="no",status=1)}

temp <- readDat(testFlow[1],col=0,row=0)
nCol <- temp$nCol
nRow <- temp$nRow

#get exclusion masks, if available
ex.mask<-rep(FALSE,nRow*nCol)
in.mask<-rep(TRUE, nRow*nCol)
if(sigprocDir != "") {
  bfMaskFile <- paste(sigprocDir,"/bfmask.bin",sep="")
  if (!file.exists(bfMaskFile)){
    bfMaskFile<-paste(sigprocDir,"/analysis.bfmask.bin",sep="")
  }
  bf <- readBeadFindMask(bfMaskFile)
  if(!is.null(bf$maskExclude)) {
    ex.mask<-which(bf$maskExclude>0)
    in.mask<-which(bf$maskExclude<1)
  }
}

#variable range
jumpSize<-1 # variance jump to check at minimum
stablerange<-c(0.05,0.95)
masked<-0.05 #assume 5% masked at least
safety<-0.01
gbscheme<-rgb(0,1:99/100,99:1/100)

#do the fast pass on summary statistics
for (iFlow in seq(0,maxFlows,by=sampleFlowsBy)){
	dat<-readDat(sprintf("%s/acq_%04d.dat",datDir,iFlow),returnWellSD=TRUE,returnSignal=FALSE,returnWellLag=TRUE)

	dv<-dat$wellSD
	zerolevel<-quantile(dv[in.mask],prob=c(masked)) 
	zerolevel<-max(zerolevel,jumpSize)
	bottom<-log(quantile(dv[dv>zerolevel],prob=stablerange[1]))
	top<-max(log(quantile(dv[dv>zerolevel],prob=stablerange[2])),bottom)
	dvp<-dv
  #fold into range
	dvp[dv>bottom]<-pmax(exp(bottom+safety),dv[dv>bottom])
	dvp[dv>exp(top-safety)]<-exp(top-safety)
  #fold into range
  dvp[ex.mask]<-bottom-1 #out of range to suppress invisible regions
	png(sprintf("%s/%s-var-%03d.png",plotDir,expName,iFlow),width=1400,height=1400)
	image(matrix(log(dvp+1),dat$nCol,dat$nRow),zlim=c(bottom,top),col=gbscheme,xlab=paste("<-- Width = ",dat$nCol," wells -->",sep=""),ylab=paste("<-- Height = ",dat$nRow," wells -->",sep=""), main=sprintf("%03d flow log(sd) of well",iFlow))
	dev.off()

  # now do the same thing using lagged sd
 	dv<-dat$wellLag
	zerolevel<-quantile(dv[in.mask],prob=c(masked)) 
	zerolevel<-max(zerolevel,jumpSize)
	bottom<-log(quantile(dv[dv>zerolevel],prob=stablerange[1]))
	top<-max(log(quantile(dv[dv>zerolevel],prob=stablerange[2])),bottom)
	dvp<-dv
  #fold into range
	dvp[dv>bottom]<-pmax(exp(bottom+safety),dv[dv>bottom])
	dvp[dv>exp(top-safety)]<-exp(top-safety)
  #fold into range
  dvp[ex.mask]<-bottom-1 #out of range to suppress invisible regions
	png(sprintf("%s/%s-lag-%03d.png",plotDir,expName,iFlow),width=1400,height=1400)
	image(matrix(log(dvp+1),dat$nCol,dat$nRow),zlim=c(bottom,top),col=gbscheme,xlab=paste("<-- Width = ",dat$nCol," wells -->",sep=""),ylab=paste("<-- Height = ",dat$nRow," wells -->",sep=""), main=sprintf("%03d flow lagged(sd) of well",iFlow))
	dev.off()
 
}

#generate animated gif
command <- sprintf("convert -delay 50 -loop 100 %s/%s-var-*.png %s/%s.flow.gif",plotDir,expName,plotDir,expName)
system(command)

command <- sprintf("convert -delay 50 -loop 100 %s/%s-lag-*.png %s/%s.lag.flow.gif",plotDir,expName,plotDir,expName)
system(command)


ft.dim<-c(dat$nCol,dat$nRow) #default for 314

if (TRUE){
#do a slow pass on the first few flows with residuals

jumpSize<-30 # variance jump to check
stablerange<-c(0.01,0.8)
masked<-0.05 #assume 5% masked at least
safety<-0.01
pre.exclude<-0.151 #different for 316????
gbscheme<-rgb(0,1:99/100,99:1/100)

#magic time span between valves
#ts<-15:75
ts<-1:61
ts.min<-1
ts.max<-61
#my.x<-dat$frameMid
my.x<-as.vector((dat$frameStart+dat$frameEnd)/2)
my.xout<-seq(1,5,length=61) #evenly interpolated frames in range of interest

flowPattern<-seq(0,maxFlows,by=sampleFlowsBy)
#accumulate statistics
flow.scale<-rep(0,length(flowPattern))
flow.top<-rep(0,length(flowPattern))
flow.outlier<-rep(0,length(flowPattern))

sample.well.rate<-floor(1+(ft.dim[1]+ft.dim[2])/1500) #about 500 per side no matter how large?
col.vec<-seq(1,ft.dim[1],by=sample.well.rate)
row.vec<-seq(1,ft.dim[2],by=sample.well.rate)
col.pat<-rep(col.vec,length(row.vec))
row.pat<-rep(row.vec,rep(length(col.vec),length(row.vec)))
pat.idx<-row.pat*ft.dim[1]+col.pat+1

pat.ex<-pat.idx %in% ex.mask

nSignal<-length(pat.idx) #subsampled now
nSample<-10000
ox<-sample((1:nSignal)[!pat.ex],nSample,replace=T) #indexes below even the pattern

#do some sampling here to fix a scale
#do a rapid pass to match scale
#Bad person for not refactoring this into functions
SeqKount<-0
for (iFlow in sample(flowPattern,5)){
  SeqKount<-SeqKount+1
  my.signal<-readDat(sprintf("%s/acq_%04d.dat",datDir,iFlow),col=col.pat[ox],row=row.pat[ox], loadMinTime=1,loadMaxTime=5)$signal
	#my.signal<-readDat(sprintf("%s/acq_%04d.dat",datDir,iFlow))$signal[pat.idx[ox],]
	#my.signal<-t(apply(my.signal,1,function(x){approx(my.x,x,my.xout)$y}))
  ts<-1:dim(my.signal)[2]
  ts.min<-min(ts)
  ts.max<-max(ts)

	#generate a magic signal
	ty.ox<-apply(my.signal,2,median) #total reference over whole chip!!!
	ty.median<-mean(range(ty.ox[ts]))  #midrange
	ty.inter<-approxfun(ts,ty.ox[ts],rule=2)
	ty.which<-which.min((ty.ox[ts]-ty.median)^2)
	#complicated

	t.which<-apply(my.signal[,ts],1,function(x){which.min((x-ty.median)^2)})
	scale.fac<-pmax(t.which/ty.which,0.05)
	tmax<-pmin(floor(ts.max*scale.fac),ts.max)
	
  dv<-sapply(1:nSample,function(iSignal){		
    scale.ts<-(ts-ts.min)/scale.fac[iSignal]+ts.min
		ty.test<-ty.inter(scale.ts)
		sd((my.signal[iSignal,ts]-ty.test)[1:tmax[iSignal]])
    })


	med.scale<-median(dv,na.rm=T) #exclude excluded regions
	med.IQR<-IQR(dv,na.rm=T)
	top<-pmax(med.scale+4*med.IQR,5*med.scale) #up this?
	#accumulate statistics for Marina
	flow.scale[SeqKount]<-med.scale
	flow.top[SeqKount]<-top
  print(iFlow)
}

stable.scale<-median(flow.scale[1:SeqKount])
stable.top<-median(flow.top[1:SeqKount])

#subsample to increase speed

track.weird<-rep(0,length(pat.idx))

SeqKount<-0
for (iFlow in flowPattern){
  my.signal<-readDat(sprintf("%s/acq_%04d.dat",datDir,iFlow),col=col.pat,row=row.pat, loadMinTime=1,loadMaxTime=5)$signal
	#my.signal<-readDat(sprintf("%s/acq_%04d.dat",datDir,iFlow))$signal[pat.idx,]
	#my.signal<-t(apply(my.signal,1,function(x){approx(my.x,x,my.xout)$y}))
  SeqKount<-SeqKount+1

	#generate a magic signal
	ty.ox<-apply(my.signal[ox,],2,median)
	ty.median<-mean(range(ty.ox[ts]))  #midrange
	ty.inter<-approxfun(ts,ty.ox[ts],rule=2)
	ty.which<-which.min((ty.ox[ts]-ty.median)^2)
	#complicated
	t.which<-apply(my.signal[,ts],1,function(x){which.min((x-ty.median)^2)})
	scale.fac<-pmax(t.which/ty.which,0.05)
	tmax<-pmin(floor(ts.max*scale.fac),ts.max)
	
  dv<-sapply(1:nSignal,function(iSignal){		
    scale.ts<-(ts-ts.min)/scale.fac[iSignal]+ts.min
		ty.test<-ty.inter(scale.ts)
		sd((my.signal[iSignal,ts]-ty.test)[1:tmax[iSignal]])
    })

	t.full<-rep(NA,nSignal)
	t.full<-ts.max-t.which
  t.full[pat.ex]<- (-1)

	med.scale<-median(dv,na.rm=T)
	med.IQR<-IQR(dv,na.rm=T)

	top<-stable.top
	
  weird<-sum(dv[!pat.ex]>top,na.rm=T)/sum(!pat.ex)
  track.weird[dv>top]<-track.weird[dv>top]+1 #these objects became weird in this flow
	bottom<-0.01
	dvp<-pmin(pmax(dv,bottom+0.01),top-0.01)

  dvp[pat.ex]<-bottom-1
	png(sprintf("%s/%s-res-%03d.png",plotDir,expName,iFlow),width=1400,height=1400)
	image(matrix(dvp,length(col.vec),length(row.vec)),zlim=c(bottom,top),col=gbscheme,xlab=sprintf("<-- Width = %d wells -->",ft.dim[1]),ylab=sprintf("<-- Height = %d wells -->",ft.dim[2]), main=sprintf("%03d flow sd of well,median=%4.1f outlier level=%4.1f outlier percent=%2.1f",iFlow,med.scale,top,weird*100))
	dev.off()

	#accumulate statistics for Marina
	flow.scale[SeqKount]<-med.scale
	flow.top[SeqKount]<-stable.top
	flow.outlier[SeqKount]<-weird*100

	png(sprintf("%s/%s-time-%03d.png",plotDir,expName,iFlow),width=1400,height=1400)
	image(matrix(t.full,length(col.vec),length(row.vec)),zlim=c(0,61),col=gbscheme,xlab=sprintf("<-- Width = %d wells -->",ft.dim[1]),ylab=sprintf("<-- Height = %d wells -->",ft.dim[2]), main=sprintf("%03d time-warp of well,median=%d",iFlow,ty.which))
	dev.off()
  print(iFlow)
}

#weird map
track.weird[pat.ex]<-(-1)
total.weird<-sum(track.weird[!pat.ex]>0)/sum(!pat.ex)
	png(sprintf("%s/%s.outlier-all.png",plotDir,expName),width=1400,height=1400)
	image(matrix(pmin(track.weird,1),length(col.vec),length(row.vec)),zlim=c(0,1),col=gbscheme,xlab=sprintf("<-- Width = %d wells -->",ft.dim[1]),ylab=sprintf("<-- Height = %d wells -->",ft.dim[2]), main=paste("fraction cells ever weird",round(100*total.weird,3)))
	dev.off()

#write out weird map to verify
weird.table<-data.frame(well=pat.idx,outlierEver=track.weird)
write.csv(weird.table,sprintf("%s/%s.alien.csv",plotDir,expName))

#generate animated gif
command <- sprintf("convert -delay 50 -loop 100 %s/%s-res-*.png %s/%s.flow.res.gif",plotDir,expName,plotDir,expName)
system(command)
command <- sprintf("convert -delay 50 -loop 100 %s/%s-time-*.png %s/%s.flow.time.gif",plotDir,expName,plotDir,expName)
system(command)

flow.all<-cbind(flow=flowPattern,mediansd=flow.scale,sdthreshold=flow.top,outlierpct=flow.outlier)
write.csv(flow.all,file=sprintf("%s/%s.flow.all.csv",plotDir,expName),quote=F,row.names=F)

#plot per flow
png(sprintf("%s/%s-outlier-by-flow.png",plotDir,expName),width=800,height=800)
out.range<-range(flow.outlier,0,10)
plot(flowPattern,flow.outlier,xlab="flow",ylab="outlier pct", ylim=out.range, main=sprintf("%s outliers by flow", expName))
dev.off()

}



