################################ exon9CNV (the exon9 amplicon); fixed sd for each run
#cm: cluster mean
#cs: cluster std
#pm: cluster membership probability matrix
#cw: cluster weight
CN3.add = 1
cprob <- function(point, m) {
    # dnorm: normal density function (x, mean, and sd) 
    sapply(1:m$n, function(i){dnorm(point, m$cm[i], m$cs[i])})
}

estep <- function(m) {
    m$pm <- sapply(m$v, function(x){temp <- cprob(x, m) * m$cw; temp/sum(temp)})
    temp <- apply(m$pm, 1, sum)
    m$cw <- temp/sum(temp)   #cluster weight
    return(m)
}

mstep <- function(m) {
    temp <- m$pm * t(replicate(m$n, m$wt))
    pwt <- apply(temp, 1, sum)
    m$cm <- as.vector(temp %*% m$v / pwt)
    dis <- t(replicate(m$n, m$v)) - replicate(length(m$v), m$cm)
    m$cs <- rep(sqrt(sum(apply(temp *  dis * dis, 1, sum)) / sum(pwt)), m$n)  #cluster sd, 3 class
    return(m)
}
  
#log likelihood
ll <- function(m) {
    sum(sapply(m$v, function(x){log(sum(cprob(x, m) * m$cw))}))
}

exon9CNV <- function(nn_sub, exon9_idx) {
    do = as.numeric(nn_sub[exon9_idx,])	
	THRES.CN4 = median(do)+0.8
	exon.CN4.idx = (do>THRES.CN4+0.2)
	do[do>THRES.CN4] = THRES.CN4
	
	amplicon.THRES.CN0 = median(do)-2
	do[do<amplicon.THRES.CN0] = amplicon.THRES.CN0
	
    cm2 <- median(do)
    cm = c(cm2-1, cm2, cm2+exon.CN3.add)
    cs <- c(0.1, 0.1, 0.1) #cluster sd, 3 class
    n <- length(cm) #number of clusters
    cw <- rep(1/n, n) #cluster weight
    pd <- c(cm) # one prior data point
 
    pw <- rep(1, length(pd)) #prior data point weights

    v <- c(pd, do) #all data points for clustering
    wt <- c(pw, rep(1, length(do))) #weights of all data points
    oll <- -.Machine$double.xmax #likelihood from the previous iteration
    np <- length(pd) #number of prior data points

    m <- list(v=v, n=n, wt=wt, cm=cm, cs=cs, pm=0, cw=cw)
    m <- estep(m)
    nll <- ll(m)
    i <- 1
    while(i < 60 && nll - oll > 0.00001) {
        oll <- nll
        m <- mstep(m)
        m <- estep(m)
        nll <- ll(m)
        i <- i+1
    #    cat("iter=",i,"loglikelihood=",nll,"\n")
    }


    # retrieve call / calculate Phred score 
    temp <- sapply(do, function(v){cprob(v,m)})
    call <- apply(temp, 2, which.max)
    # apply(temp, 2, max), maximal density
    temp <- apply(temp, 2, function(v){v/sum(v)})
    # temp is now probability
    temp <- apply(temp, 2, function(v){1-max(v)})
    temp[temp < .Machine$double.xmin] <- .Machine$double.xmin
    Phred <- -10 * log10(temp)
    Phred[Phred>100] <- 100
    
    ExonOut <- list(call=call, Phred=Phred, m=m, do=do, np=np, exon.CN4.idx=exon.CN4.idx)
    return(ExonOut)
}



######################fit gene level; fixed sd for each amplicon in a run
estepGene <- function(m) {
  m$pm <- apply(m$v, 2, function(x){temp <- cprobGene(x, m) * m$cw; temp/sum(temp)})
  temp <- apply(m$pm, 1, sum)
  m$cw <- temp/sum(temp)   #cluster weight
  return(m)
}

mstepGene <- function(m) {
  temp <- m$pm * t(replicate(m$n, m$wt))
  pwt <- apply(temp, 1, sum)
  m$cm <- apply(temp %*% t(m$v), 2, function(v){v/pwt})
  for(i in 1:m$na) {
    dis <- t(replicate(m$n, m$v[i,])) - replicate(length(m$v[i,]), m$cm[,i])
    m$cs[,i] <- rep(sqrt(sum(apply(temp *  dis * dis, 1, sum)) / sum(pwt)), m$n)   #cluster sd;
  }
  return(m)
}
  
#log likelihood
llGene <- function(m) {
  sum(apply(m$v, 2, function(x){log(sum(cprobGene(x, m) * m$cw))}))
}

cprobGene <- function(point, m) {
  cprobGenei <- function(point, mu, sd) {
    prod(sapply(1:length(point), function(i){dnorm(point[i], mu[i], sd[i])}))
  }
  sapply(1:m$n, function(i){cprobGenei(point, m$cm[i,], m$cs[i,])})
}

# check individual amplicon
cprobGeneIDX <- function(point, m, idx) {
  cprobGenei <- function(point, mu, sd) {
    sapply(idx, function(i){dnorm(point[i], mu[i], sd[i])})
  }
  sapply(1:m$n, function(i){cprobGenei(point, m$cm[i,], m$cs[i,])})
}

# Voting...


geneCNV <- function(nn_sub, geneamp_idx) {
    do <- as.matrix(nn_sub[geneamp_idx,]) #real data
    na <- length(geneamp_idx) #number of amplicons
    samples <- dim(nn_sub)[2]
	
    cm2 <- apply(do, 1, median)
    cm = rbind(cm2-1, cm2, cm2+CN3.add)
	
  	gene.CN4.idx = NULL
    for(d_idx in 1:dim(do)[1]) {
		THRES.CN4 = median(as.numeric(do[d_idx,]))+0.8
		gene.CN4.idx = rbind(gene.CN4.idx, (do[d_idx,]>THRES.CN4+0.2))
	    do[d_idx,do[d_idx,] > THRES.CN4] = THRES.CN4
	}
	  
	# include flooding of CN0 cases for clustering stability
    for(d_idx in 1:dim(do)[1]) {
		amplicon.THRES.CN0 = median(as.numeric(do[d_idx,]))-3
	    do[d_idx,do[d_idx,] < amplicon.THRES.CN0] = amplicon.THRES.CN0
	}
	
    n <- nrow(cm) #number of clusters
    cs <- matrix(0.1, n, na) #cluster sd;
    cw <- rep(1/n, n) #cluster weight

    pd <- t(cm)
    pw <- rep(1, ncol(pd)) #prior data point weights

    pw <- c(0.1,0.1,0.1)  # put less weight on the prior -> lower confidence when inconsistent

    v <- cbind(pd, do) #all data points for clustering
    wt <- c(pw, rep(1, ncol(do))) #weights of all data points
    oll <- -.Machine$double.xmax #likelihood from the previous iteration
    np <- ncol(pd) #number of prior data points

    m <- list(v=v, n=n, na=na, wt=wt, cm=cm, cs=cs, pm=0, cw=cw)
    m <- estepGene(m)
    nll <- llGene(m)
    i <- 1

    while(i < 60 && nll - oll > 0.00001) {
        oll <- nll
        m <- mstepGene(m)
        m <- estepGene(m)
        nll <- llGene(m)
        i <- i+1
#        cat("iter=",i,"loglikelihood=",nll,"\n")
    }

    temp <- apply(do, 2, function(v){cprobGene(v,m)})
    call.allamplicon <- apply(temp, 2, which.max)

    VOTE = NULL
	tempProb = vector("list",na)
    for(idx in 1:length(geneamp_idx)) {
        tempProb[[idx]] <- apply(do, 2, function(v){cprobGeneIDX(v,m,idx)})
        VOTE = rbind(VOTE, as.numeric(apply(tempProb[[idx]], 2, which.max)))
    }
    VOTE.call = NULL
    for(idx in 1:length(call.allamplicon)) {
        VOT = as.data.frame(table(VOTE[,idx]))
        VOT = VOT[order(VOT$Freq, decreasing=TRUE),]
        VOTE.call = c(VOTE.call, as.numeric(as.character(VOT$Var1[1])))
    }
	# remove at most three amplicons for voting...
    
	if(length(which(call.allamplicon == VOTE.call)) == samples) {
	    # when Voting over all amplicons is consistent with VOTE over individual amplicons
	    call = VOTE.call
		print(paste("[Use all amplicons for voting]",sep=""))
	} else {
		MisMatch = NULL
	    for(i in 1:na) {   MisMatch = c(MisMatch, length(which(VOTE[i,] != VOTE.call)))  }
		
    	# exclude the top (at most 3) mismatched amplicons		
		ampIDX = 1:na
		MistMatchIDX = which(MisMatch>0)
		if( length(MistMatchIDX) <=4 ) {
		    ampIDX = setdiff(ampIDX, MistMatchIDX)			
			print(paste("Exclude inconsistent gene-level amplicons:",MistMatchIDX,sep=""))
		} else {
    		MisMatchTMP = cbind(ampIDX, MisMatch)
	    	MisMatchTMP = MisMatchTMP[order(MisMatchTMP[,2], decreasing=T),]
		    ampIDX = MisMatchTMP[-c(1:4),1]
			print(paste("Exclude inconsistent gene-level amplicons:",MisMatchTMP[1:4,1],sep=""))
		}
		
    	# use remaining amplicons for calculate posterior probability
		for(a_idx in 1:length(ampIDX)) {
		    if(a_idx==1) {
			    temp=tempProb[[ampIDX[a_idx]]]
			} else {
			    temp=temp*tempProb[[ampIDX[a_idx]]]
			}
		}		
		# make new call with less amplicons, and trust it anyway
		call <- apply(temp, 2, which.max)	
	}
	temp <- apply(temp, 2, function(v){v/sum(v)})
    # temp is now probability
    temp <- apply(temp, 2, function(v){1-max(v)})
    temp[temp < .Machine$double.xmin] <- .Machine$double.xmin
    Phred <- -10 * log10(temp)
    Phred[Phred>100] <- 100			
		
    GeneOut <- list(call=call, Phred=Phred, m=m, do=do, np=np, na=na, gene.CN4.idx=gene.CN4.idx)
    return(GeneOut)
}

#Reporting <- function(LogFile) {
#	if (!file.exists(LogFile)){    stop(paste("Log File ",LogFile," does not exist.  Please check again.", sep="");    }
#	myLog = read.csv(LogFile, sep="\t") 
#}
