# Grab the NucStep files for the middle patch, 
# subract the empties from the beads,
# and plot the results for the early flows.
# Goal is to be able to easily disntiguish individual flows by eye.
args <- commandArgs()
if (length(args[7]) > 0){
    print(args[7])
    nucstep.path = paste(args[7], "NucStep", sep='/')
}else{
    nucstep.path = paste(Sys.getenv("RUNINFO__SIGPROC_DIR"), "NucStep", sep='/')
}
time.file  = paste(nucstep.path, "NucStep_frametime.txt",    sep='/')
bead.file  = paste(nucstep.path, "NucStep_middle_bead.txt",  sep='/')
emty.file  = paste(nucstep.path, "NucStep_middle_empty.txt", sep='/')
trace.time = as.numeric(read.table(time.file))
bead.trace = as.matrix(read.table(bead.file))
emty.trace = as.matrix(read.table(emty.file))
num.flows  = dim(bead.trace)[1]
flow.nucs  = bead.trace[,2]
sig.col.0  = 8
sig.col.1  = sig.col.0 + length(trace.time) - 1
sig.cols   = sig.col.0:sig.col.1
n.sig.cols = length(sig.cols)
diff.trace = matrix(as.numeric(bead.trace[,sig.cols]) - as.numeric(emty.trace[,sig.cols]), nrow=num.flows, ncol=n.sig.cols)

flow2color = function(flow)
{
	nuc.col = list("T"='red', "A"='green', "C"='blue', "G"='black')
    as.character(nuc.col[flow.nucs[flow]])
}

flows.per.plot = 4  # plot this many flows in each plot
nrow = 4            # arrange plots in temporal order with this many rows of plots
ncol = 8            # and this many columns

plot.flows = function(first, last)
{
	ylim = 1.01 * range(diff.trace)
	lwd  = 4

    plot(trace.time, trace.time, ylim=ylim, type="n", xlab="", ylab="")

    for(flow in first:last){
        col = flow2color(flow)
        lty = flow %% flows.per.plot + 1
        lines(trace.time, diff.trace[flow,], col=col, lty=lty, lwd=lwd)
    }

    text = sapply(first:last, function(flow){sprintf("%d",flow)})
    fill = sapply(first:last, flow2color)
    lty  = first:last %% flows.per.plot + 1
    legend("bottomleft", legend=text, col=fill, lty=lty, title="flow", lwd=lwd, seg.len=8)
}

# Generate one big png with all the plots:
output.path  = Sys.getenv("TSP_FILEPATH_PLUGIN_DIR")
png.file = paste(output.path, "flow-by-flow.png", sep='/')
png(png.file, width=3840, height=2400, bg=rgb(0.9,0.9,0.9))
par(mfrow=c(nrow,ncol), mar=c(1,1,1,1))

for(first in flows.per.plot * (1:(nrow*ncol) - 1) + 1){
    last = first + flows.per.plot - 1
    plot.flows(first, last)
}

dev.off()

