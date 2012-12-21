##########################################################################################
# Copyright (C) 2012Ion Torrent Systems, Inc. All Rights Reserved
# Christian Koller, 12. Nov 2012
#
# The function SimulateAndSolveSeq Simulates random base sequences given phase and noise
# paramters. The return value is a matric of size [10, numBases+nchar(keySeq)] containing
# the fraction of reads with "row"th error occuring before or at base position "column".
#
# numBases        : Number of bases in a random sequence
# numFlows        : Number of flows to be simulated
# numWells        : Number of sequences to be simulated
# noiseSigma      : Standard deviation of white gaussian noise that is added to the signal
# PhaseParameters : Phase parameters, vector of length 3 <CF, IE, Droop>
# flowOrder       : Default "TACGTACGTCTGAGCATCGATCGATGTACAGC"
# keySeq          : Default "TCAG",
# noNegativeSignal: Default TRUE
#                   If true, no negative signal values can occur.
##########################################################################################

SimulateAndSolveSeq <- function(
  numBases,
  numFlows,
  numWells,
  noiseSigma,
  PhaseParameters,
  flowOrder = "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  keySeq = "TCAG",
  noNegativeSignal=TRUE,
  plotFigure=TRUE,
  randSeed=NA
){
    #loading required libraries
    #library(torrentR)
    
    # check consistency of input arguments
    if(nchar(flowOrder) < numFlows){
      flowOrder <- substring(paste(rep(flowOrder,ceiling(numFlows/nchar(flowOrder))),collapse=""),1,numFlows)
    } else if (nchar(flowOrder) > numFlows) {
      flowOrder <- substring(flowOrder, 1, numFlows)
    }
    if(length(PhaseParameters) != 3)
      stop ("Error: <PhaseParameters> needs be a vector of length 3 <cf, ie, dr>")
    
    # Set random number generator
    if (is.numeric(randSeed))
      set.seed(randSeed)
    
    # creating numWells random DNA sequences
    SequenceVector = rep(keySeq, numWells)
    NoisySignal <- matrix(nrow= numWells, ncol=numFlows)
    
    for (i in 1:length(SequenceVector))
        SequenceVector[i] <- paste(SequenceVector[i], paste(sample(c("A", "C", "G", "T"), numBases, replace=TRUE), collapse=""), sep="")
        
    temp <- SimulateCAFIE(SequenceVector, flowOrder ,PhaseParameters[1],PhaseParameters[2],PhaseParameters[3], numFlows, simModel="treePhaserSim")
        
    for (i in 1:length(SequenceVector)) {
        # Apply all sorts of distortions
        NoisySignal[i, ] <- runif(1, min=0.5, max=3)*(temp$sig[i, ] + rnorm(numFlows, sd=noiseSigma))
        if (noNegativeSignal)
          NoisySignal[i, ][(NoisySignal[i, ]<0)] <- 0
    }
    
    
    # Solving Sequences using Treephaser
    Solution <- treePhaser(NoisySignal, flowOrder, PhaseParameters[1], PhaseParameters[2], PhaseParameters[3], keySeq=keySeq, basecaller="treephaser-swan")
    
    
    # Error Analysis
    cumulativeErrorPos <- matrix(0, nrow=10, ncol=numBases+nchar(keySeq))
    errorBasePosition <- matrix(0, nrow=10, ncol=numBases+nchar(keySeq))
    # q17length q20length q30length q47length
    QlengthVector <- matrix(0, nrow=4, ncol=numWells)
    maxerrors <- ceiling(0.02*(numBases+nchar(keySeq)))
    idMax <- 1:maxerrors
    qvals <- c(17, 20, 30, 47)
    meanQreadlength <- rep(0, length(qvals))
    
    for (i in 1:length(SequenceVector)){
      # Translate to flow space
      truth <- seqToFlow(SequenceVector[i],flowOrder,nFlow=numFlows)
      calls <- seqToFlow(Solution$seq[i],flowOrder,nFlow=numFlows)
      errorFlow <- which((truth - calls) != 0)
      errorPos <- 0
      errorBase <- vector(length=length(errorFlow))
      
      Qvector <- rep(NA, maxerrors)
      QlengthVec <- rep(NA, maxerrors)
        
      #if (length(errorFlow)>0){
      #  print(errorFlow)
      #  print(truth[errorFlow])
      #  print(calls[errorFlow])
      #  break
      #}
      for (EF in errorFlow){
        for (k in 1:abs(truth[EF]-calls[EF])) {
            
          errorPos <- errorPos +1
          # Get base position of error
          if (truth[EF]>calls[EF]) {
            basePos <- sum(truth[1:(EF-1)]) + calls[EF] + k
          }
          else {
            basePos <- sum(truth[1:(EF-1)]) + truth[EF]
          }
          if (errorPos<11)
            errorBasePosition[errorPos, basePos] <- errorBasePosition[errorPos, basePos] +1
          if (errorPos<(maxerrors+1)) {
              if (basePos>1) {
                QlengthVec[errorPos] <- basePos-1
                Qvector[errorPos] <- -10*log10((errorPos-1)/QlengthVec[errorPos])
              } else {
                Qvector[errorPos] <- -Inf
                QlengthVec[errorPos] <- 0
              }
                            
          }
        }
        if (errorPos>max(10,maxerrors))
          break
      }
        
      # Calculate mean Q lengths
      if (errorPos == 0){
          QlengthVector[ ,i] <- numBases+nchar(keySeq)
      } else {
          # Find first NA value in vector, i.e., how many errors were logged
          if (all(is.na(Qvector)==FALSE)) {
            firstNA <- maxerrors
          } else {
            firstNA <- min(idMax[is.na(Qvector)])
            QlengthVec[firstNA] <- numBases+nchar(keySeq)
            Qvector[firstNA] <- -10*log10((firstNA-1)/QlengthVec[firstNA])
          }
          QlengthVec <- QlengthVec[1:firstNA]
          Qvector <- Qvector[1:firstNA]
          myMaxId <- idMax[1:firstNA]
          for (q in 1:length(qvals)) {
              testvec <- myMaxId[Qvector>qvals[q]]
              if (length(testvec>0)) {
                  QlengthVector[q,i] <- QlengthVec[max(testvec)]
              } else {
                  QlengthVector[q,i] <- 0
              }
          }
      } 
    } # for sequences
    for (q in 1:length(qvals)) {
        meanQreadlength[q] <- mean(QlengthVector[q, ])
    }
    cumulativeErrorPos[ ,1] <- errorBasePosition[ ,1]
    for (i in 2:(numBases+nchar(keySeq)))
      cumulativeErrorPos[ ,i] <- cumulativeErrorPos[ ,(i-1)] + errorBasePosition[ ,i]
    cumulativeErrorPos <- cumulativeErrorPos / numWells
    
    
    # Figure
    if (plotFigure) {
      g_range <- max(cumulativeErrorPos[1, ])
      BasePosition <- 1:(numBases+nchar(keySeq))
      plot(BasePosition, cumulativeErrorPos[1, ], type="l", col=4, xlab="Base position", ylab="Fraction of sequences with error no. x before pos.")
      lines(BasePosition, cumulativeErrorPos[3, ], type="l", col=6)
      lines(BasePosition, cumulativeErrorPos[5, ], type="l", col=3)
      lines(BasePosition, cumulativeErrorPos[7, ], type="l", col=2)
      grid(NULL)
      legend(1, g_range, c("1st Error", "3rd Error", "5th Error", "7th Error"), col=c(4,6,3,2), lty=c(1,1,1,1))
    }
    
    # Output
    ReturnVal <- list()
    ReturnVal$cumulativeErrorPos <- cumulativeErrorPos
    ReturnVal$meanQ17length <- meanQreadlength[1]
    ReturnVal$meanQ20length <- meanQreadlength[2]
    ReturnVal$meanQ30length <- meanQreadlength[3]
    ReturnVal$meanQ47length <- meanQreadlength[4]
    
    return(ReturnVal)
}