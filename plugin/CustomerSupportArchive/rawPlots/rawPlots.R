theme_set(theme_grey(18))
nflows = length(flows)
bf = readBeadFindMask(maskFile);
nucs = c("T","A","C","G")

# Size of region (in wells)
regionW = 15;
regionH = 15;

isProtonChip <- function(chipType) {
  firstLetter <- substr(chipType,1,1)
  return(firstLetter == "9" | firstLetter == "P" | firstLetter == "5" | firstLetter == "G" )
}

if(isProtonChip(chipType)) {
  # Timeframe for zooms (in seconds)
  timeZoomBfLo = 1
  timeZoomBfHi = 6
  timeZoomNucLo = 1
  timeZoomNucHi = 5
  # For bb we use 4 evenly-spaced regions along inlet-to-outlet axis
  nRegion = 4
  regionSep = 1/nRegion
  xVal = seq(regionSep/2,1-regionSep/2,length=nRegion)
  yVal = 1 - xVal 
  if(multiLane == 'False'){
    regionParam = lapply(1:length(xVal),function(z){
          list(
              name=sprintf("region%02d",z),
              x=xVal[z],
              y=0.5,
              w=regionW,
              h=regionH
          )
        })
  }else{
    xVal = xVal[activeLanes]
    regionParam = lapply(1:length(yVal),function(z){
          list(
              name=sprintf("region%02d",z),
              x=xVal,
              y=yVal[z],
              w=regionW,
              h=regionH
          )
        })
  }
} else {
  # 3-series chips
  # Timeframe for zooms (in seconds)
  timeZoomBfLo = 1
  timeZoomBfHi = 5
  timeZoomNucLo = 1
  timeZoomNucHi = 3
  # 3-series chips
  regionParam = list(
    list(name="Inlet",      x=0.9, y=0.9, w=regionW, h=regionH),
    list(name="Middle",     x=0.5, y=0.5, w=regionW, h=regionH),
    list(name="Outlet",     x=0.1, y=0.1, w=regionW, h=regionH),
    list(name="LowerRight", x=0.7, y=0.1, w=regionW, h=regionH)
  )
}

nCol = bf$nCol
nRow = bf$nRow
makeRegion = function(r) {
  minCol = floor(r$x * nCol)
  minRow = floor(r$y * nRow)
  return(list(
    minCol=minCol,
    maxCol=minCol+r$w,
    minRow=minRow,
    maxRow=minRow+r$h,
    name=r$name
  ))
}
regions = lapply(regionParam,makeRegion)

#nColBin = 100
nColBin = min(bf$nCol/4, 100);
#nRowBin = 100
nRowBin = min(bf$nRow/4, 100);
if(isProtonChip(chipType) & bf$nCol==1200 & bf$nRow==800) {
  nColBin = bf$nCol/4
  nRowBin = bf$nRow/4
  png(plotFile <- sprintf("%s/bfDetail-%s.png", plotDir, expName), width=850, height=650)
  image(matrix(bf$maskBead,nrow=bf$nCol,ncol=bf$nRow),col=c("black","white"),xaxt="n",yaxt="n")
  title(sprintf("%s\nBeadfind - one well per plotted pixel, black=empty",expName))
  abline(v=seq(0,1,length=1+12),col="red",lwd=1.5)
  abline(h=seq(0,1,length=1+8),col="red",lwd=1.5)
  dev.off()
}

bfBin <- chipPlot(bf$col,bf$row,bf$maskBead,maxCol=bf$nCol,maxRow=bf$nRow,doPlot=FALSE,nColBin=nColBin,nRowBin=nRowBin)
zMat <- formImageMatrix(bfBin$binnedCol,bfBin$binnedRow,bfBin$binnedVal,bfBin$nColBin,bfBin$nRowBin)
png(plotFile <- sprintf("%s/regionMap-%s.png", plotDir, expName), width=850, height=650)
image(zMat,col=colorRampPalette(c("black","white"))(256),zlim=c(0,1),xaxt="n",yaxt="n")
title(sprintf("%s\nRegions of interest",expName))
lapply(regions,function(z){
      color_region = c('red', 'blue', 'green', 'purple')
      x <- c(z$minCol,z$maxCol)/nCol
      y <- c(z$minRow,z$maxRow)/nRow
      id <- z$name
      rgid <- substr(id, nchar(id) - 1, nchar(id))
      rgid <- as.numeric(rgid)
      polygon(c(x,rev(x)),rep(y,c(2,2)),border="blue",lwd=3,density=100,col=color_region[rgid])
    })
if(isProtonChip(chipType) & bf$nCol==1200 & bf$nRow==800) {
  abline(v=seq(0,1,length=1+12),col="red",lwd=1.5)
  abline(h=seq(0,1,length=1+8),col="red",lwd=1.5)
}
dev.off()
#system(sprintf("eog %s",plotFile))

bfdats = list();
bfsubs = list();
bfEmptySubs = list();
bufferList = list();
for (i in c(1,3,4)) {
  preBf = sprintf("%s/beadfind_pre_%04d.dat",datDir, i);
  if(file.exists(preBf)) {
    bfdats[[i]] = loadDat(preBf, bf, regions)
    bfsubs[[i]] = BFBgSub(bfdats[[i]])

    bfRange = quantile(as.numeric(bfsubs[[i]]$signal), c(.03,.97))
    bfRange = expandRange2(bfRange, .1)
    png(sprintf("%s/sub-prebeadfind-%d-bg-%s.png", plotDir, i, expName), width=850, height=600)
    plotSig(bfsubs[[i]], sprintf("Bg Sub Pre Beadfind %d %s", i, expName), ylim=bfRange)
    dev.off();

    bfEmptySubs[[i]] = BFEmptySub(bfdats[[i]]);
    bfRange = quantile(as.numeric(bfEmptySubs[[i]]$signal), c(.03,.999))
    bfRange = expandRange2(bfRange, .1)
    png(sprintf("%s/empty-sub-prebeadfind-%d-bg-%s.png", plotDir, i, expName), width=850, height=600)
    plotSig(bfEmptySubs[[i]], sprintf("Empty Sub Pre Beadfind %d %s", i, expName), ylim=bfRange)
    dev.off();
    n = names(bfEmptySubs[[i]]$diffMean)
    for (x in 1:length(bfEmptySubs[[i]]$diffMean)) {
      lname = sprintf("vec_beadfind_pre_000%d_%s",i,n[x]);
      bufferList[[lname]] = as.vector(bfEmptySubs[[i]]$diffMean[[x]]);
      lname = sprintf("sum_beadfind_pre_000%d_%s",i,n[x]);
      bufferList[[lname]] = sum(bfEmptySubs[[i]]$diffMean[[x]]);
    }
    timeBoundLo = which(bfsubs[[i]]$time > timeZoomBfLo)[1]
    timeBoundHi = rev(which(bfsubs[[i]]$time < timeZoomBfHi))[1]
    timeRange = timeBoundLo:timeBoundHi
    bfRange = quantile(as.numeric(bfsubs[[i]]$signal[,timeRange]), c(.03,.99))
    bfRange = expandRange2(bfRange, .1)
    png(sprintf("%s/sub-zoom-prebeadfind-%d-bg-%s.png", plotDir, i, expName), width=850, height=600)
    plotSig(bfsubs[[i]], sprintf("Bg Sub Pre Beadfind %d %s", i, expName), xlim=c(timeZoomBfLo,timeZoomBfHi))
    dev.off();

    bfRange = quantile(as.numeric(bfdats[[i]]$signal), c(.03,.97))
    bfRange = expandRange2(bfRange, .1)
    png(sprintf("%s/raw-prebeadfind-%d-%s.png", plotDir, i, expName), width=850, height=600)
    plotSig(bfdats[[i]], sprintf( "Raw Pre Beadfind %d %s", i, expName), ylim=bfRange)
    dev.off()
  }
}

j = toJSON(bufferList[order(names(bufferList))])
jj = gsub(',\"', ',\n\"', j)
jj = sub('{','{\n',jj,perl=T);
jj = sub('}','\n}',jj,perl=T);
write(jj, file=sprintf('%s/results.json', plotDir));
dats = list();
subs = list()
for (i in 0:(nflows-1)) {
  print(paste("Loading ", flows[i+1]));
  dats[[i+1]] = loadDat(sprintf("%s/acq_%04d.dat",datDir, flows[i+1]), bf, regions)
  subs[[i+1]] = bgSub(dats[[i+1]])
}

rRange = c(0,0);
bgRange = c(0,0);
for (i in 1:length(dats)) {
  rRange = range(c(rRange), quantile(as.numeric(dats[[i]]$signal), c(.02,.99)))
  bgRange = range(c(bgRange), quantile(as.numeric(subs[[i]]$signal), c(.02,.99)))
}

rRange = expandRange2(rRange, .1)
bgRange = expandRange2(bgRange, 1)

for (i in (1:length(dats))) {
  print(paste("Plotting ", flows[i]));
  flow = flows[i]
  nuc = nucs[flow%%4+1]
  png(sprintf("%s/sub-%s-%d-bg-%s.png", plotDir, nuc, flow, expName), width=850, height=600)
  plotSig(subs[[i]], sprintf("BgSub Flow %d (Nuc %s) %s", flow, nuc,expName),ylim=bgRange)
  dev.off();
  png(sprintf("%s/raw-flow-%s-%d-%s.png", plotDir, nuc, flow, expName), width=850, height=600)
  plotSig(dats[[i]], sprintf( "Raw Flow %d (Nuc %s) %s", flow, nuc, expName),ylim=rRange)
  dev.off()
}

rRange = c(0,0);
bgRange = c(0,0);
for (i in 1:length(dats)) {
  timeBoundLo = which(dats[[1]]$time > timeZoomNucLo)[1]
  timeBoundHi = rev(which(dats[[1]]$time < timeZoomNucHi))[1]
  timeRange = timeBoundLo:timeBoundHi
  rRange = range(c(rRange), quantile(as.numeric(dats[[i]]$signal[,timeRange]), c(.01,.99)))
  bgRange = range(c(bgRange), quantile(as.numeric(subs[[i]]$signal[,timeRange]), c(.01,.99)))
}

rRange = expandRange2(rRange, .1)
bgRange = expandRange2(bgRange, .6)

for (i in (1:length(dats))) {
  print(paste("Plotting ", i));
  flow = flows[i]
  nuc = nucs[flow%%4+1]
  png(sprintf("%s/raw-flow-zoom-%s-%d-%s.png", plotDir, nuc, flow, expName), width=900, height=600)
 plotEmptyVsSig(dats[[i]], sprintf( "Raw Flow %d (Nuc %s) %s", flow, nuc, expName),ylim=rRange, xlim=c(timeZoomNucLo,timeZoomNucHi))
  dev.off()
}
