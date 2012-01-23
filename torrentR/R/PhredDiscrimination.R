PhredDiscrimination <- function(sff)
{
	# Compute the discrimination power of the Q-score
	# assingment in an sff file, per Ewing & Green 1998.

	# Get the list of Q scores, in sorted order, and
	# the number of bases assigned each Q score:
	rng    = range(sff$qual)
	breaks = rng[1]:(rng[2]+1) - 0.5
	qhist  = hist(sff$qual, breaks=breaks, plot=F)
	nbins  = length(qhist$mids)
	qvals  = qhist$mids[nbins:1]
	cnts   = qhist$counts[nbins:1]

	# Convert from Q scores to error probablities:
	errp   = sapply(qvals, function(q){10**(-q/10)})

	# Compute B_r, P_r and r, as defined in Ewing & Green:
	B_r    = cumsum(cnts)
	nquals = length(sff$qual)
	P_r    = B_r / nquals
	r      = cumsum(errp * cnts) / B_r

	list(r = r, P_r = P_r)
}

PhredDiscriminationList <- function(sff.files)
{
	# Compute discrimination for each of a list of sff files.
	sff.dis = list()
	for(sff.file in sff.files){
		cat(sff.file, "\n")
		sff = readSFF(sff.file)
		dis = PhredDiscrimination(sff)
		sff.dis[[sff.file]] = dis
	}
	
	sff.dis
}

PlotDiscrimination <- function(sff.dis, r.max=0, P_r.max=0)
{
	# Plot a set of discrimination curves on a single set of axes.
	if(r.max == 0)
		r.max = max(unlist(lapply(sff.dis, function(x){max(x$r)})))
	
	if(P_r.max == 0)
		P_r.max = max(unlist(lapply(sff.dis, function(x){max(x$P_r[which(x$r<r.max)])})))

	plot(0, 0, xlim=c(0,r.max), ylim=c(0,P_r.max), xlab="r", ylab="P_r", type="n")

	nphred = length(sff.files)
	col    = rainbow(nphred)
	n      = 1
	for(sff.file in sff.files){
		dis = sff.dis[[sff.file]]
		points(dis$r, dis$P_r, col=col[n], type="b", pch=19)
		n = n + 1
	}

	names = unlist(lapply(sff.files, function(name){tmp=sub("rawlib.","",name); sub(".sff","",tmp)}))
	legend("bottomright", names, fill=col)
}

#sff.files = Sys.glob("rawlib.*.sff")
#sff.dis   = PhredDiscriminationList(sff.files)

#PlotDiscrimination(sff.dis)


