args = commandArgs(TRUE)

## -- debug start --
#analysisRoot = '/results-ddn2/Home/Auto_user_f8-1481-ChipCC10_L1_EC400_VA6_200M_reuseWellv2-KS_15086_27734/'
#analysisRoot = '/results-ddn2/Home/Auto_sierra_Vb06-33-25Feb-PQRun-CS_15162_27851/'
#outRoot = './tmp2'
#lib.loc = './'
#scriptRoot = './'
## -- debug end --

if (! exists('analysisRoot') ){
if (length(args)==0 ){
  cat("Usage: Rscript True_loading_fullchip analysisRoot [outRoot] [lib.loc] [scriptRoot]\n")
  quit()
} else if (length(args)==1) {
  analysisRoot = args[1]   # analysis directory
  outRoot = './'
  lib.loc = './'
  scriptRoot = './'
} else {
  analysisRoot = args[1]   # analysis directory
  outRoot = args[2]        # output directory
  lib.loc = args[3]        # extra library location
  scriptRoot = args[4]     # location of True_loading.R script
  
}
}
options(error = recover)

print("Start fullchip True Loading...")

name = substr(analysisRoot, regexpr(pattern ="Home/",analysisRoot)+nchar('Home/'), nchar(analysisRoot))


library(parallel)
library(torrentR)
# define function to process one block
processBlock = function(block.folder){
  blockAnalysisRoot = sprintf('%s/outputs/SigProcActor-00/%s', analysisRoot, block.folder)
  blockOutRoot = sprintf('%s/%s', outRoot, block.folder)
  cmd = sprintf('mkdir %s',  blockOutRoot)
  system(cmd)
  cmd = sprintf('Rscript %s/True_loading.R %s %s %s 1>/dev/null 2>&1', scriptRoot, blockAnalysisRoot, blockOutRoot, lib.loc)
  print(cmd)
  system(cmd)
}

file.path(analysisRoot, "outputs/SigProcActor-00")
block.folder.list = dir(path=file.path(analysisRoot, "outputs/SigProcActor-00") , pattern = '^block_X')
block.folder.list

active.lanes = function(analysisRoot){
	    # return logical flag for whether a lane is active
	    explog.file = sprintf('%s/explog_final.txt', analysisRoot)
    if (! file.exists(explog.file)){
	            explog.file = sprintf('%s/explog.txt', analysisRoot)
        }
        cat(explog.file, '\n')
        f = file(explog.file, "r")
	    active = rep(F, 4)
	    while (1){
		            line = readLines(f, n = 1)
	            if (length(line) == 0) break
		            if ( grepl("^LanesActive", line) ){
				                l = as.integer(gsub("LanesActive", "", strsplit(line, ":")[[1]][1] ))
		                cat(line, l, '\n')
				            if (strsplit(line, ":")[[1]][2] == "yes"){
						                    active[l] = T
				            }
				        }
		            
		        }
	        return(active)
}

is.Valkyrie = function(analysisRoot){
	    # return logical flag for whether a lane is active
	    explog.file = sprintf('%s/explog_final.txt', analysisRoot)
    if (! file.exists(explog.file)){
	            explog.file = sprintf('%s/explog.txt', analysisRoot)
        }
        cat(explog.file, '\n')
        f = file(explog.file, "r")
	    active = rep(F, 4)
	    while (1){
		            line = readLines(f, n = 1)
	            if (length(line) == 0) break
		            if ( grepl("^Platform:", line) ){
				                if (grepl("Valkyrie", line)){
							                return(T)
		                } else {
					                return(F)
				            }
		            }
		        }    
}


# for lanes 

block.coord = lapply(strsplit(gsub('block_X', '', block.folder.list), '_Y'), as.integer)
block.coord
block.x = sapply(block.coord, function(x){x[1]})
unique.x = sort(unique(block.x))
num.x = length(unique.x)
lane.x = list()
str(lane.x)


if (is.Valkyrie(analysisRoot)) {
	# Valyrie analysis: only have blocks that are active
	lane.active = active.lanes(analysisRoot)
	 num.lanes = sum(lane.active)
	l.idx = 1
	  for (l in 1:4){
		  if (lane.active[l]){
		      lane.x[[l]] = unique.x[(num.x/num.lanes*(l.idx-1)+1):(num.x/num.lanes*(l.idx))]
		      l.idx = l.idx + 1
		  } else {
		    lane.x[[l]] = integer(0) # empty array
		  }
	   }
} else {
	# Non-Valyrie analysis: all blocks are there
  num.lanes = 4
  for (l in 1:num.lanes){
      lane.x[[l]] = unique.x[(num.x/num.lanes*(l-1)+1):(num.x/num.lanes*(l))]
}
}
str(lane.x)

# proces all blocks - parallel
ret = mclapply(block.folder.list, processBlock, mc.cores = 12)
str(ret)



# combine results - full chip
fullchip.results = list()

library(rjson)
for (block.folder in block.folder.list){
  blockOutRoot = sprintf('%s/%s', outRoot, block.folder)
  results.file = sprintf('%s/results.json', blockOutRoot)
  if (  ! file.exists(results.file) ){
    next
  }
  results = fromJSON(file = results.file)
  if (length(fullchip.results)==0){
    # initialize fullchip results
    fullchip.results$meta$numWells = 0
    fullchip.results$meta$numAddressableWells = 0
    fullchip.results$true$bead = 0
    fullchip.results$true$lib = 0
    fullchip.results$true$lowQuality = 0
    fullchip.results$true$polyClonal = 0
    fullchip.results$true$passFilter = 0
    fullchip.results$true$histogram$copyCount = results$true$histogram$copyCount
    fullchip.results$true$histogram$bead = 0
    fullchip.results$true$histogram$lib = 0
    fullchip.results$true$histogram$passFilter = 0
    fullchip.results$true$histogram$lowQuality = 0
    fullchip.results$true$histogram$badKey = 0
    fullchip.results$true$histogram$poly = 0
    
    fullchip.results$apparent$bead = 0
    fullchip.results$apparent$lib = 0
    fullchip.results$apparent$lowQuality = 0
    fullchip.results$apparent$polyClonal = 0
    fullchip.results$apparent$passFilter = 0
    fullchip.results$apparent$histogram$copyCount = results$apparent$histogram$copyCount
    fullchip.results$apparent$histogram$bead = 0
    fullchip.results$apparent$histogram$lib = 0
    fullchip.results$apparent$histogram$passFilter = 0
    fullchip.results$apparent$histogram$lowQuality = 0
    fullchip.results$apparent$histogram$badKey = 0
    fullchip.results$apparent$histogram$poly = 0
    
    fullchip.results$truePerRegion = list()
    
    # initialize lanes
    for (lane in 1:4){
	    fullchip.results[[sprintf('lane%d',lane)]]$meta$numWells = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$meta$numAddressableWells = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$bead = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$lib = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$lowQuality = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$polyClonal = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$passFilter = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$copyCount = results$true$histogram$copyCount
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$bead = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lib = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$passFilter = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lowQuality = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$badKey = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$poly = 0
	    
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$bead = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$lowQuality = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$polyClonal = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$passFilter = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$copyCount = results$apparent$histogram$copyCount
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$bead = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lib = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$passFilter = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lowQuality = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$badKey = 0
	    fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$poly = 0
	    
	    fullchip.results[[sprintf('lane%d',lane)]]$truePerRegion = list()
	 }

   }   
  
  
  fullchip.results$meta$numWells = fullchip.results$meta$numWells + results$meta$numWells
  fullchip.results$meta$numAddressableWells = fullchip.results$meta$numAddressableWells + results$meta$numAddressableWells
  
  fullchip.results$true$bead = fullchip.results$true$bead + results$true$bead
  fullchip.results$true$lib = fullchip.results$true$lib + results$true$lib
  fullchip.results$true$lowQuality = fullchip.results$true$lowQuality + results$true$lowQuality
  fullchip.results$true$polyClonal = fullchip.results$true$polyClonal + results$true$polyClonal
  fullchip.results$true$passFilter = fullchip.results$true$passFilter + results$true$passFilter
  fullchip.results$true$histogram$bead = fullchip.results$true$histogram$bead + results$true$histogram$bead
  fullchip.results$true$histogram$lib = fullchip.results$true$histogram$lib + results$true$histogram$lib
  fullchip.results$true$histogram$passFilter = fullchip.results$true$histogram$passFilter + results$true$histogram$passFilter
  fullchip.results$true$histogram$lowQuality = fullchip.results$true$histogram$lowQuality + results$true$histogram$lowQuality
  fullchip.results$true$histogram$badKey = fullchip.results$true$histogram$badKey + results$true$histogram$badKey
  fullchip.results$true$histogram$poly = fullchip.results$true$histogram$poly + results$true$histogram$poly
  
  fullchip.results$apparent$bead = fullchip.results$apparent$bead + results$apparent$bead
  fullchip.results$apparent$lib = fullchip.results$apparent$lib + results$apparent$lib
  fullchip.results$apparent$lowQuality = fullchip.results$apparent$lowQuality + results$apparent$lowQuality
  fullchip.results$apparent$polyClonal = fullchip.results$apparent$polyClonal + results$apparent$polyClonal
  fullchip.results$apparent$passFilter = fullchip.results$apparent$passFilter + results$apparent$passFilter
  fullchip.results$apparent$histogram$bead = fullchip.results$apparent$histogram$bead + results$apparent$histogram$bead
  fullchip.results$apparent$histogram$lib = fullchip.results$apparent$histogram$lib + results$apparent$histogram$lib
  fullchip.results$apparent$histogram$passFilter = fullchip.results$apparent$histogram$passFilter + results$apparent$histogram$passFilter
  fullchip.results$apparent$histogram$lowQuality = fullchip.results$apparent$histogram$lowQuality + results$apparent$histogram$lowQuality
  fullchip.results$apparent$histogram$badKey = fullchip.results$apparent$histogram$badKey + results$apparent$histogram$badKey
  fullchip.results$apparent$histogram$poly = fullchip.results$apparent$histogram$poly + results$apparent$histogram$poly
  
  block.x = as.integer(unlist(strsplit(gsub('block_X', '', block.folder), '_Y'))[1])
  for (lane in 1:4){
          if (!(block.x %in% lane.x[[lane]])) next
	  fullchip.results[[sprintf('lane%d',lane)]]$meta$numWells = fullchip.results[[sprintf('lane%d',lane)]]$meta$numWells + results$meta$numWells
	  fullchip.results[[sprintf('lane%d',lane)]]$meta$numAddressableWells = fullchip.results[[sprintf('lane%d',lane)]]$meta$numAddressableWells + results$meta$numAddressableWells

	  fullchip.results[[sprintf('lane%d',lane)]]$true$bead = fullchip.results[[sprintf('lane%d',lane)]]$true$bead + results$true$bead
	  fullchip.results[[sprintf('lane%d',lane)]]$true$lib = fullchip.results[[sprintf('lane%d',lane)]]$true$lib + results$true$lib
	  fullchip.results[[sprintf('lane%d',lane)]]$true$lowQuality = fullchip.results[[sprintf('lane%d',lane)]]$true$lowQuality + results$true$lowQuality
	  fullchip.results[[sprintf('lane%d',lane)]]$true$polyClonal = fullchip.results[[sprintf('lane%d',lane)]]$true$polyClonal + results$true$polyClonal
	  fullchip.results[[sprintf('lane%d',lane)]]$true$passFilter = fullchip.results[[sprintf('lane%d',lane)]]$true$passFilter + results$true$passFilter
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$bead = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$bead + results$true$histogram$bead
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lib = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lib + results$true$histogram$lib
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$passFilter = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$passFilter + results$true$histogram$passFilter
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lowQuality = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$lowQuality + results$true$histogram$lowQuality
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$badKey = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$badKey + results$true$histogram$badKey
	  fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$poly = fullchip.results[[sprintf('lane%d',lane)]]$true$histogram$poly + results$true$histogram$poly

	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$bead = fullchip.results[[sprintf('lane%d',lane)]]$apparent$bead + results$apparent$bead
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib = fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib + results$apparent$lib
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$lowQuality = fullchip.results[[sprintf('lane%d',lane)]]$apparent$lowQuality + results$apparent$lowQuality
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$polyClonal = fullchip.results[[sprintf('lane%d',lane)]]$apparent$polyClonal + results$apparent$polyClonal
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$passFilter = fullchip.results[[sprintf('lane%d',lane)]]$apparent$passFilter + results$apparent$passFilter
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$bead = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$bead + results$apparent$histogram$bead
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lib = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lib + results$apparent$histogram$lib
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$passFilter = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$passFilter + results$apparent$histogram$passFilter
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lowQuality = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$lowQuality + results$apparent$histogram$lowQuality
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$badKey = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$badKey + results$apparent$histogram$badKey
	  fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$poly = fullchip.results[[sprintf('lane%d',lane)]]$apparent$histogram$poly + results$apparent$histogram$poly

  }

 
  # per region true statistics
  fullchip.results$truePerRegion[[block.folder]] = results$true
  fullchip.results$truePerRegion[[block.folder]]$numWells = results$meta$numWells
  fullchip.results$truePerRegion[[block.folder]]$numAddressableWells = results$meta$numAddressableWells
  
}

# calculate statistics - full chip
fullchip.results$true$loadingPercent = fullchip.results$true$bead/fullchip.results$meta$numAddressableWells*100
fullchip.results$true$polyClonalPercent = fullchip.results$true$polyClonal/fullchip.results$true$lib*100
fullchip.results$true$lowQualityPercent = fullchip.results$true$lowQuality/(fullchip.results$true$lib)*100
fullchip.results$true$usablePercent = fullchip.results$true$passFilter/fullchip.results$true$lib*100

fullchip.results$apparent$loadingPercent = fullchip.results$apparent$bead/fullchip.results$meta$numAddressableWells*100  #fullchip.results$meta$numWells*100
fullchip.results$apparent$polyClonalPercent = fullchip.results$apparent$polyClonal/fullchip.results$apparent$lib*100
fullchip.results$apparent$lowQualityPercent = fullchip.results$apparent$lowQuality/(fullchip.results$apparent$lib)*100
fullchip.results$apparent$usablePercent = fullchip.results$apparent$passFilter/fullchip.results$apparent$lib*100

for (lane in 1:4){
	fullchip.results[[sprintf('lane%d',lane)]]$true$loadingPercent = fullchip.results[[sprintf('lane%d',lane)]]$true$bead/fullchip.results[[sprintf('lane%d',lane)]]$meta$numAddressableWells*100
	fullchip.results[[sprintf('lane%d',lane)]]$true$polyClonalPercent = fullchip.results[[sprintf('lane%d',lane)]]$true$polyClonal/fullchip.results[[sprintf('lane%d',lane)]]$true$lib*100
	fullchip.results[[sprintf('lane%d',lane)]]$true$lowQualityPercent = fullchip.results[[sprintf('lane%d',lane)]]$true$lowQuality/(fullchip.results[[sprintf('lane%d',lane)]]$true$lib)*100
	fullchip.results[[sprintf('lane%d',lane)]]$true$usablePercent = fullchip.results[[sprintf('lane%d',lane)]]$true$passFilter/fullchip.results[[sprintf('lane%d',lane)]]$true$lib*100

	fullchip.results[[sprintf('lane%d',lane)]]$apparent$loadingPercent = fullchip.results[[sprintf('lane%d',lane)]]$apparent$bead/fullchip.results[[sprintf('lane%d',lane)]]$meta$numAddressableWells*100  #fullchip.results$meta$numWells*100
	fullchip.results[[sprintf('lane%d',lane)]]$apparent$polyClonalPercent = fullchip.results[[sprintf('lane%d',lane)]]$apparent$polyClonal/fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib*100
	fullchip.results[[sprintf('lane%d',lane)]]$apparent$lowQualityPercent = fullchip.results[[sprintf('lane%d',lane)]]$apparent$lowQuality/(fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib)*100
	fullchip.results[[sprintf('lane%d',lane)]]$apparent$usablePercent = fullchip.results[[sprintf('lane%d',lane)]]$apparent$passFilter/fullchip.results[[sprintf('lane%d',lane)]]$apparent$lib*100
}



# plot true statistics - per region
x.all = c()
y.all = c()
for (block.folder in names(fullchip.results$truePerRegion) ){
  xy = as.numeric(strsplit(gsub('_Y', ',',  gsub('block_X', '', block.folder)), ',')[[1]])
  x.all = c(x.all, xy[[1]])
  y.all = c(y.all, xy[[2]])
  x.unique = sort(unique(x.all))
  y.unique = sort(unique(y.all))
}

nrow = length(y.unique)
ncol = length(x.unique)
loading_true = matrix(0, nrow=nrow, ncol=ncol)
usable_true = matrix(0, nrow=nrow, ncol=ncol)
poly_true = matrix(0, nrow=nrow, ncol=ncol)
lowQ_true = matrix(0, nrow=nrow, ncol=ncol)
for (r in 1:nrow){
  for (c in 1:ncol){
    block.folder = sprintf('block_X%d_Y%d', x.unique[[c]], y.unique[[r]])
    if (block.folder %in% names(fullchip.results$truePerRegion)){
      loading_true[[r,c]]= as.numeric(fullchip.results$truePerRegion[[block.folder]]$loadingPercent)
      usable_true[[r,c]] = as.numeric(fullchip.results$truePerRegion[[block.folder]]$usablePercent)
      poly_true[[r,c]] = as.numeric(fullchip.results$truePerRegion[[block.folder]]$polyClonalPercent)
      lowQ_true[[r,c]] = as.numeric(fullchip.results$truePerRegion[[block.folder]]$lowQualityPercent)
    }
}
}

# plot loading per block
library(gplots)
plotPerBlock = function(val, filename, titleText, pal = NULL){
  png(filename, width = 800, height = 600)
  cellnote = matrix(sprintf("%4.1f", val[nrow(val):1,]), nrow=8)
  if (is.null(pal)){
    pal <- colorRampPalette(c(rgb(0.96,0.96,1), rgb(0.1,0.1,0.9)), space = "rgb")
  }
  
  #Plot the matrix
  heatmap.2(val[nrow(val):1,], Rowv=FALSE, Colv=FALSE, dendrogram="none", main=titleText, xlab="Columns", ylab="Rows", col=pal, tracecol="#303030", trace="none", 
            cellnote=cellnote, notecol="black", notecex=1.2, keysize = 1.5, margins=c(5, 5), breaks = seq(quantile(val, 0.05, na.rm=T), quantile(val, 1., na.rm=T), length.out = 101),
            labRow = 8:1)
  dev.off() 
}

pal <- colorRampPalette(c(rgb(0.96,0.96,0.96), rgb(0.1,0.1,0.9)), space = "rgb")
plotPerBlock(loading_true, sprintf("%s/true_loading_per_block.png", outRoot), sprintf("True loading %%\n%s", name), pal)
pal <- colorRampPalette(c(rgb(0.96,0.96,0.96), rgb(0.4,0.9,0.1)), space = "rgb")
plotPerBlock(usable_true, sprintf("%s/true_usable_per_block.png", outRoot), sprintf("True usable %%\n%s", name), pal)
pal <- colorRampPalette(c(rgb(0.96,0.96,0.96), rgb(0.9,0.1,0.1)), space = "rgb")
plotPerBlock(poly_true, sprintf("%s/true_poly_per_block.png", outRoot), sprintf("True polyclonal %%\n%s", name), pal)
pal <- colorRampPalette(c(rgb(0.96,0.96,0.96), rgb(0.9,0.1,0.9)), space = "rgb")
plotPerBlock(lowQ_true, sprintf("%s/true_lowQuality_per_block.png", outRoot), sprintf("True Low quality %%\n%s", name), pal)



write.table(loading_true[nrow:1,], sprintf("%s/true_loading_per_block.txt", outRoot), sep="\t", row.names = F, col.names = F)
write.table(usable_true[nrow:1,], sprintf("%s/true_usable_per_block.txt", outRoot), sep="\t", row.names = F, col.names = F)







# calculate mean and sd copy count
apparent.mean = list()
apparent.sd = list()
true.mean = list()
true.sd = list()
for (t in c('bead', 'lib', 'passFilter', 'lowQuality', 'badKey', 'poly')){
  apparent.mean[[t]] = sum(fullchip.results$apparent$histogram$copyCount * fullchip.results$apparent$histogram[[t]])/sum(fullchip.results$apparent$histogram[[t]])
  apparent.sd[[t]] = sqrt(sum( (fullchip.results$apparent$histogram$copyCount - apparent.mean[[t]])^2 *fullchip.results$apparent$histogram[[t]]  )/sum(fullchip.results$apparent$histogram[[t]]))
  
  true.mean[[t]] = sum(fullchip.results$true$histogram$copyCount * fullchip.results$true$histogram[[t]])/sum(fullchip.results$true$histogram[[t]])
  true.sd[[t]] = sqrt(sum( (fullchip.results$true$histogram$copyCount - true.mean[[t]])^2 *fullchip.results$true$histogram[[t]]  )/sum(fullchip.results$true$histogram[[t]]))
}
fullchip.results$copyCount$apparentMean = apparent.mean
fullchip.results$copyCount$apparentSD = apparent.sd
fullchip.results$copyCount$trueMean = true.mean
fullchip.results$copyCount$trueSD = true.sd



str(fullchip.results)
library(jsonlite, lib.loc=lib.loc)
jsonTxt = jsonlite::toJSON(fullchip.results, pretty = TRUE);
write(jsonTxt,file=sprintf("%s/results.json", outRoot))


# plot histogram
library(ggplot2)
library(reshape2)
png(sprintf('%s/copycount_true.png', outRoot), width = 600, height = 400)
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
hist.table = data.frame(fullchip.results$true$histogram)
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
png(sprintf('%s/copycount_pipeline.png', outRoot), width = 600, height = 400)
cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
hist.table = data.frame(fullchip.results$apparent$histogram)
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

# combine and plot spatial maps
first.data = T
spatial.data=list()
for (r in 1:nrow){
  spatial.data[r] = list()
  for (c in 1:ncol){
    block.folder = sprintf('block_X%d_Y%d', x.unique[[c]], y.unique[[r]])
    if (block.folder %in% names(fullchip.results$truePerRegion)){
        fn = sprintf('%s/%s/spatial.data.R', outRoot, block.folder)
        if (file.exists(fn)){
          load(fn)  # load spatialMap
          
          # Initialize full chip heat map
          if (first.data){
            first.data = F
            spatialMapFull = list()
            for (t in names(spatialMap)){
              spatialMapFull[[t]] = matrix(0, nrow = nrow*nrow(spatialMap[[t]]), ncol =  ncol*ncol(spatialMap[[t]]))
            }
          }
          
          # 
          for (t in names(spatialMap)){
            num.rows.map = nrow(spatialMap[[t]])
            num.cols.map = ncol(spatialMap[[t]])
            spatialMapFull[[t]][ ( (r-1)*num.rows.map + 1) : (r*num.rows.map), ((c-1)*num.cols.map +1): (c*num.cols.map)  ] = spatialMap[[t]]
          }

        }
    }
  }
}

# plot heatmap
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
for (t in names(spatialMapFull)){
  z = t(spatialMapFull[[t]]) 
  png(plotFile <- sprintf("%s/spatial_true_plot_%s.png",outRoot, t),width=500,height=550)
  par(mai=c(1,0.8,0.8,1))
  depthLimLower <- quantile(z,prob=0.005,na.rm=TRUE)
  depthLimUpper <- quantile(z,prob=0.995,na.rm=TRUE)
  imageWithHist(z, zlim=c(depthLimLower,depthLimUpper),header=sprintf('True %s %%',t),col=jet.colors(256), asp=ncol(z)/nrow(z))
  title(name)
  dev.off()
}

print("True_loading_fullchip.R completed successfully.")
