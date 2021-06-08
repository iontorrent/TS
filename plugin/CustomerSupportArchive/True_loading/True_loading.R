args = commandArgs(TRUE)

########## debug ############
# analysisRoot = '/results-ddn2/Home/minReadLength_300_19972/block_X0_Y0'
# outRoot = './tmp2'
# lib.loc = './'
#############################

if (!exists('analysisRoot')){
  if (length(args)==0){
    cat("Usage: Rscript True_loading analysisRoot [outRoot] [lib.loc]\n")
    quit()
  } else if (length(args)==1) {
    analysisRoot = args[1]   # analysis directory
    outRoot = './'
    lib.loc = './'
  } else {
    analysisRoot = args[1]   # analysis directory
    outRoot = args[2]        # output directory
    lib.loc = args[3]        # extra library location
  }
}


library(torrentR)
library(rhdf5, lib.loc=lib.loc)
library(ggplot2)
library(reshape2)
library(jsonlite, lib.loc=lib.loc)

#sigprocRoot = Sys.getenv("SIGPROC_DIR")
#basecallerRoot = Sys.getenv("BASECALLER_DIR")
sigprocRoot = sprintf('%s', analysisRoot)
basecallerRoot = sprintf('%s', analysisRoot)

name = substr(analysisRoot, regexpr(pattern ="Home/",analysisRoot)+nchar('Home/'), nchar(analysisRoot))


#
sigprocRoot
basecallerRoot
name

# read copy count
wellsFile =sprintf('%s/1.wells', sigprocRoot) 
copyCount = h5read(wellsFile, '/wells_copies')

# read beadfind mask
bfmaskb = readBeadFindMask(sprintf('%s/bfmask.bin', basecallerRoot))
bfmaska = readBeadFindMask(sprintf('%s/analysis.bfmask.bin', sigprocRoot))
# list bfmask
print(sprintf('%-25s%10s%10s', '', 'sigproc', 'basecall'))
for (n in names(bfmaska)){
  if (substr(n,1,4) == "mask"){
    print(sprintf('%-25s%10d%10d', n, sum(bfmaska[[n]]), sum(bfmaskb[[n]])))
  }
}


# populate library, polyclonal and low quality masks
mask = {}
#mask$addressable = (bfmaska$maskBead==1 | bfmaska$maskEmpty==1| bfmaska$maskIgnore==1)
mask$addressable = (bfmaska$maskBead==1 | bfmaska$maskEmpty==1)
mask$bead = bfmaskb$maskBead ==1
mask$lib = bfmaskb$maskLib == 1
mask$poly = bfmaskb$maskLib & bfmaska$maskFilteredBadPPF
mask$lowQuality = bfmaskb$maskLib & 
  (bfmaskb$maskFilteredBadKey | 
     bfmaskb$maskFilteredBadResidual|
     bfmaskb$maskFilteredShort|
     bfmaskb$maskIgnore|
     bfmaskb$maskPinned|
#     bfmaskb$maskWashout|
     (bfmaskb$maskFilteredBadPPF & ! bfmaska$maskFilteredBadPPF))
mask$badKey = bfmaskb$maskFilteredBadKey ==1
mask$passFilter = mask$lib & !(mask$poly | mask$lowQuality)

# populate statistics
fstats = data.frame()
for (n in names(mask)){
  #print(sprintf('%-25s%10d%10.1f%%', n, sum(mask[[n]]), 100.*sum(mask[[n]]/sum(mask$lib))))
  fstats = rbind(fstats, data.frame(n, sum(mask[[n]]), round(100.*sum(mask[[n]]/length(mask$lib)),1), round(100.*sum(mask[[n]]/sum(mask$bead)),1)))
}
names(fstats) = c('category', 'counts', 'percentage by well (%)', 'percentage by bead (%)')
mask$row = bfmaska$row
mask$col = bfmaska$col
#fstats

# pipeline stat
df = data.frame(mask$row, mask$col, as.vector(copyCount))
names(df) = c('row', 'col', 'copyCount')
filter = rep('none', nrow(df))
filter[mask$bead] = 'bead'
filter[mask$lib] = 'lib'
filter[mask$passFilter] = 'passFilter'
filter[mask$lowQuality] = 'lowQuality'
filter[mask$badKey] = 'badKey'
filter[mask$poly] = 'polyclonal'
df$bead = mask$bead
df$lib = mask$lib
df$passFilter = mask$passFilter
df$lowQuality = mask$lowQuality
df$badKey = mask$badKey
df$poly = mask$poly
#str(df)
# plot histogram
breaks = seq(0,10, by=0.05)
h = hist(1, breaks, plot=F)
hist.table = data.frame(h$mids)
names(hist.table) = c('copyCount')
apparent.mean2 = list()
apparent.sd2 = list()
for (t in c('bead', 'lib', 'passFilter', 'lowQuality', 'badKey', 'poly')){
  x = df$copyCount[ df[[t]] ]
  h = hist(x[x > breaks[1] & x<breaks[length(breaks)] ],breaks=breaks, plot = F)
  hist.table[[t]] = h$count
  apparent.mean2[[t]] = mean(x, na.rm = T)
  apparent.sd2[[t]] = sd(x, na.rm = T)
}
png(sprintf('%s/copycount_pipeline.png', outRoot), width = 600, height = 400)
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
hist.table.long = melt(hist.table, id = 'copyCount')
g = ggplot(data = hist.table.long, aes(x=copyCount, y=value, color=variable)) + geom_line()  + 
  theme(text = element_text(size=15),plot.title = element_text(size = 10))  + ggtitle(sprintf('%s\n%s','Pipeline stat',name))  +scale_colour_manual(values=cbPalette)
copy.count.limit = hist.table$copyCount[cumsum(hist.table$bead) > sum(hist.table$bead)*0.999][1]
if (! is.na(copy.count.limit)){
  new.x.limit = copy.count.limit-copy.count.limit%%0.5+0.5
  g = g+ xlim(0,new.x.limit)
}
print(g)
dev.off()

hist.table.apparent = as.list(hist.table)

# true stat
df = data.frame(mask$row, mask$col, as.vector(copyCount))
names(df) = c('row', 'col', 'copyCount')
filter = rep('none', nrow(df))
filter[mask$bead] = 'bead'
filter[mask$lib] = 'lib'
filter[mask$passFilter] = 'passFilter'
filter[mask$lowQuality] = 'lowQuality'
filter[mask$badKey] = 'badKey'
filter[mask$poly] = 'polyclonal'

df$addressable = mask$addressable
df$bead = mask$bead & ! mask$badKey
df$lib = mask$lib & ! mask$badKey
df$passFilter = mask$passFilter 
df$lowQuality = mask$lowQuality& ! mask$badKey
df$badKey = mask$badKey
df$poly = mask$poly
#str(df)
# plot histogram
breaks = seq(0,10, by=0.05)
h = hist(1, breaks, plot=F)
hist.table = data.frame(h$mids)
names(hist.table) = c('copyCount')
true.median = list()
true.sd = list()
for (t in c('bead', 'lib', 'passFilter', 'lowQuality', 'badKey', 'poly')){
  x = df$copyCount[ df[[t]] ]
  h = hist(x[x > breaks[1] & x<breaks[length(breaks)] ],breaks=breaks, plot = F)
  hist.table[[t]] = h$count
  true.median[[t]] = median(x, na.rm = T)
  true.sd[[t]] = sd(x, na.rm = T)
  
}
png(sprintf('%s/copycount_true.png', outRoot), width = 600, height = 400)
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
hist.table.long = melt(hist.table, id = 'copyCount')
g = ggplot(data = hist.table.long, aes(x=copyCount, y=value, color=variable)) + geom_line()  + 
  theme(text = element_text(size=15),plot.title = element_text(size = 10))  + ggtitle(sprintf('%s\n%s','True stat',name))  +scale_colour_manual(values=cbPalette)
copy.count.limit = hist.table$copyCount[cumsum(hist.table$bead) > sum(hist.table$bead)*0.999][1]
if (! is.na(copy.count.limit)){
  new.x.limit = copy.count.limit-copy.count.limit%%0.5+0.5
  g = g+ xlim(0,new.x.limit)
}
print(g)
dev.off()
hist.table.true = as.list(hist.table)

# parse stastistics
stats = list()
stats$meta$analysisRoot = analysisRoot
stats$meta$sigprocRoot = sigprocRoot
stats$meta$basecallerRoot = basecallerRoot
stats$meta$outRoot = outRoot
stats$meta$workingDirectory = getwd()
stats$meta$numWells = length(mask$bead)
stats$meta$numAddressableWells = sum(mask$addressable)


stats$true$bead = sum(mask$bead & ! mask$badKey)
stats$true$lib = sum(mask$lib & ! mask$badKey)
stats$true$lowQuality = sum(mask$lowQuality & ! mask$badKey)
stats$true$polyClonal = sum(mask$poly)
stats$true$passFilter = sum(mask$passFilter)
stats$true$loadingPercent = stats$true$bead/sum(mask$addressable)*100
stats$true$polyClonalPercent = stats$true$polyClonal/stats$true$lib*100
stats$true$lowQualityPercent = stats$true$lowQuality/(stats$true$lib)*100
stats$true$usablePercent = stats$true$passFilter/stats$true$lib*100
stats$true$histogram = hist.table.true

stats$apparent$bead = sum(mask$bead)
stats$apparent$lib = sum(mask$lib)
stats$apparent$lowQuality = sum(mask$lowQuality)
stats$apparent$polyClonal = sum(mask$poly)
stats$apparent$passFilter = sum(mask$passFilter)
stats$apparent$loadingPercent = stats$apparent$bead/sum(mask$addressable)*100
stats$apparent$polyClonalPercent = stats$apparent$polyClonal/stats$apparent$lib*100
stats$apparent$lowQualityPercent = stats$apparent$lowQuality/(stats$apparent$lib)*100
stats$apparent$usablePercent = stats$apparent$passFilter /stats$apparent$lib*100
stats$apparent$histogram = hist.table.apparent

# calculate mean and sd copy count
apparent.mean = list()
apparent.sd = list()
true.mean = list()
true.sd = list()
for (t in c('bead', 'lib', 'passFilter', 'lowQuality', 'badKey', 'poly')){
  apparent.mean[[t]] = sum(stats$apparent$histogram$copyCount * stats$apparent$histogram[[t]])/sum(stats$apparent$histogram[[t]])
  apparent.sd[[t]] = sqrt(sum( (stats$apparent$histogram$copyCount - apparent.mean[[t]])^2 *stats$apparent$histogram[[t]]  )/sum(stats$apparent$histogram[[t]]))

  true.mean[[t]] = sum(stats$true$histogram$copyCount * stats$true$histogram[[t]])/sum(stats$true$histogram[[t]])
  true.sd[[t]] = sqrt(sum( (stats$true$histogram$copyCount - true.mean[[t]])^2 *stats$true$histogram[[t]]  )/sum(stats$true$histogram[[t]]))
}
stats$copyCount$apparentMean = apparent.mean
stats$copyCount$apparentSD = apparent.sd
stats$copyCount$trueMean = true.mean
stats$copyCount$trueSD = true.sd



# per region stat for thumbnail
if (stats$meta$numWells == 960000){
  loading_true_tn = matrix(0, nrow=8, ncol=12)
  loading_apparent_tn = matrix(0, nrow=8, ncol=12)
  for (r in 1:8){
    for (c in 1:12){
      m = mask$row >= (r-1)*100 & mask$row <r*100  & mask$col >=(c-1)*100 & mask$col < c*100 # row & col is 0-based here...
      loading_true_tn[[r,c]]=sum(mask$bead[m] & ! mask$badKey[m])/sum(mask$addressable[m])*100
      loading_apparent_tn[[r,c]]=sum(mask$bead[m])/sum(mask$addressable[m])*100
    }
  }

  # plot loading per block
  library(gplots)
  plotPerBlock = function(val, filename, titleText){
  #filename = sprintf("true_loading_per_block.png")
  png(filename, width = 800, height = 600)
  cellnote = matrix(sprintf("%4.2f", val[nrow(val):1,]), nrow=8)
  pal <- colorRampPalette(c(rgb(0.96,0.96,1), rgb(0.1,0.1,0.9)), space = "rgb")
  #Plot the matrix
  heatmap.2(val[nrow(val):1,], Rowv=FALSE, Colv=FALSE, dendrogram="none", main=titleText, xlab="Columns", ylab="Rows", col=pal, tracecol="#303030", trace="none", 
            cellnote=cellnote, notecol="black", notecex=1.2, keysize = 1.5, margins=c(5, 5), breaks = seq(quantile(val, 0.05, na.rm=T), quantile(val, 1., na.rm=T), length.out = 101),
            labRow = 8:1)
  dev.off() 
  }
  
  plotPerBlock(loading_true_tn, sprintf("%s/true_loading_per_block.png", outRoot), sprintf("True loading\n%s", name))
  plotPerBlock(loading_apparent_tn, sprintf("%s/pipeline_loading_per_block.png", outRoot), sprintf("Pipeline loading\n%s", name))
  
  write.table(loading_true_tn[8:1,], sprintf("%s/true_loading_per_block.txt", outRoot), sep="\t", row.names = F, col.names = F)
  write.table(loading_apparent_tn[8:1,], sprintf("%s/apparent_loading_per_block.txt", outRoot), sep="\t", row.names = F, col.names = F)
  
}

str(stats)

# write json
jsonTxt = toJSON(stats, pretty = TRUE);
write(jsonTxt,file=sprintf("%s/results.json", outRoot))

#### generate spatial maps

if (stats$meta$numWells == 960000){
  # thumbnail chunk size
  chunkRow = 20
  chunkCol = 20
} else {
  # full chip chunk size
  chunkRow = 148/2
  chunkCol = 184/2
}

spatialMap = {}
numRows = length(unique(df$row))
minRow = min(unique(df$row))
maxRow = max(unique(df$row))

numCols = length(unique(df$col))
minCol = min(unique(df$col))
maxCol = max(unique(df$col))

rowStart = seq(minRow, maxRow, chunkRow)
colStart = seq(minCol, maxCol, chunkRow)


# 
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
df$usable = df$passFilter
df$loading = df$bead
tags = c('addressable', 'loading', 'usable', 'lowQuality', 'badKey', 'poly')
#tags = c('addressable')
for (t in tags){
  spatialMap[[t]] =  matrix(0, nrow=length(rowStart), ncol=length(colStart))
}
start_time <- Sys.time()
for (ir in 1:length(rowStart) ){
  for (ic in 1:length(colStart) ){
    r = rowStart[ir]
    c = colStart[ic]
    m = df$row >= r & df$row < r+ chunkRow & df$col >= c & df$col < c + chunkCol
    numAddressable = 0
    numBeads = 0
    numTotal = 0
    for (t in tags){
      if (t == 'addressable'){
        numAddressable =sum(df[[t]][m])
        numTotal = sum(m)
        spatialMap[[t]][ir,ic] =  numAddressable/numTotal
      } else if (t == 'loading'){
        numBeads = sum(df[[t]][m])
        if (numAddressable>0){
          spatialMap[[t]][ir,ic] =  numBeads/numAddressable
        } else {
          spatialMap[[t]][ir,ic] = NA
        }
        
      } else {
        if (numBeads >0){
          spatialMap[[t]][ir,ic] = sum(df[[t]][m])/numBeads
        } else {
          spatialMap[[t]][ir,ic] = NA
        }
      }
      
    }
  }
}
end_time <- Sys.time()
end_time - start_time
save(spatialMap, file=sprintf('%s/spatial.data.R', outRoot) )
#imageWithHist(t(spatialMap[[tags[5]]]) )

for (t in tags){
  z = t(spatialMap[[t]]) 
  png(plotFile <- sprintf("%s/spatial_true_plot_%s.png",outRoot, t),width=500,height=550)
  par(mai=c(1,0.8,0.8,1))
  depthLimLower <- quantile(z,prob=0.005,na.rm=TRUE)
  depthLimUpper <- quantile(z,prob=0.995,na.rm=TRUE)
  c(depthLimLower, depthLimUpper)
  imageWithHist(z, zlim=c(depthLimLower,depthLimUpper),header=sprintf('True %s %%',t),col=jet.colors(256))
  title(name)
  dev.off()
}




print("True_loading.R completed successfully.")


