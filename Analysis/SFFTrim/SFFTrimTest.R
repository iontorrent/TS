# Test code for adapter_searcher.h.

library(torrentR)

# Read in simulated data, and compute true flow number of adapter start:
sim   = read.table("sim-xdb-p1.frame")
n.sim = nrow(sim)
order = paste(rep("TACGTACGTCTGAGCATCGATCGATGTACAGC",10), collapse='')  # XDB
flow  = sapply(1:n.sim, function(i){max(which(seqToFlow(substr(as.character(sim[i,]$simSeq), 1, sim[i,2]+1), order) > 0))}) - 1

# Read in results of SFFTrimTest:
res   = read.table("test.out", col.names=c('base','flow','scor'))

# Generate scatter plot comparing true adapter start, and trimmed adapter start:
nflow    = ncol(sim) - 2
plot.col = rgb(0,0,1,0.5)
line.col = 'darkgray'

png("test.png", width=600, height=600)
plot(flow, res$flow, col=plot.col, xlab='true flow', ylab='trimmed flow')
xline(nflow, col=line.col)
yline(nflow, col=line.col)
lines(c(0,400), c(0,400),   col=line.col)
lines(c(0,400), c(0,400)-4, col=line.col)
dev.off()

# How many adapter sequences where missed by the trimmer?
n.missed = length(which(flow < res$flow))

# How many were trimmed with reasonable accuracy?
tol      = 4
n.good   = length(which((flow - res$flow <= tol) | (nflow<=flow & nflow-res$flow<=tol)))

# How many were trimmed prematurely?
n.prem   = length(which((flow - res$flow >  tol) & (flow<nflow  | nflow-res$flow>tol)))

# Print summary of results:
sink("test.results")
cat(sprintf("missed:    %5.1f%%\n", 100*n.missed / n.sim))
cat(sprintf("found:     %5.1f%%\n", 100*n.good   / n.sim))
cat(sprintf("premature: %5.1f%%\n", 100*n.prem   /n.sim))
sink()

