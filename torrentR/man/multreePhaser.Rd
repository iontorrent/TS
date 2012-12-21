\name{multreePhaser}
\alias{multreePhaser}
\title{
  Jointly estimate a DNA sequence from the flow data of multiple reads.
}
\description{
  Given observed flow values and phasing parameters for the process that generated
  them, returns an esimate of the underlying DNA sequence. Solves multiple reads
  of the same sequence jointly to obtain an estiamte of the sequence.
  Version: Sept. 17 / 2012
}
\usage{
  multreePhaser(
    signal,
    flowLimits,
    numFlows,
    flowOrders,
    PhaseParameters,
    keySeq      = "TCAG",
    basecaller  = c("treephaser-swan", "treephaser-adaptive"),
    verbose = 0
  )
}
\arguments{
  \item{signal}{
    The matrix of observed signal values, one row per read and one column per flow.
    If there are X reads per sequence, the number of rows in the matrix must be divisible 
    by X. The solver groups X consecutive reads to estimate nrow(signal)/X DNA sequences.
  }
  \item{activeUntilFlow}{
  	Vector of length nrow(signal). Contains the maximum number of flows that should
  	be used by the solver for any specific read. For a given read x, if flowLimits(x) = 0, 
  	the read will not at all be used in the solving process.
  }
  \item{numFlows}{
  	Number of flows in a given flow order. 
  	-Vector of length X, where X is the maximum number of reads per sequence.
  	-If a single value is passed, the solver assumes all X reads were obtained using the same
  	number of flows.
  }
  \item{flowOrders}{
    The flow cycles - for example "TACG".
    -Vector of length X, where X is the maximum number of reads per sequence.
    -If a single string is passed, the solver assumes all X reads were obtained using the same 
    flow cycle.
  }
  \item{PhaseParameters}{
  	Matrix of size (X,2) containing the phasing parameters. 
  	The first column contains the carry-forward values for the X reads, 
  	the second contains the incomplete extension values for the X reads.	
  }
  \item{keySeq}{
    The known key sequence at the start of the read - will be used
    for key normalization.  If specified as empty, no key normalization will
    be performed. Default: "TCAG"
  }
  \item{basecaller}{
    The variant of treePhaser to be used - options are "treephaser-swan" (the
    default) which performs sliding window normalization, and "treephaser-adaptive"
    which always restarts solving at the beginning of the read.
  }
  \item{verbose}{
  	Switch to have the solver print messages.
  	 (0) No messages (Default)
  	 (1) Messages are displayed after Solving/Normalization/Simulation steps.
  	 (10) All messages are displayed
  }
}
\value{
  The return value is a list with the following elements.
  \item{seq}{
    The estimated sequences.
  }
  \item{nBases}{
    A vector containing the number of Bases called.
  }
}
\examples{ 
	\dontrun{
   numFLows <- c(100, 50)
   flowOrders <- c("TACGTACGTCTGAGCATCGATCGATGTACAGC", "TACG")
   Phasing <- matrix(nrow=2, ncol=2)
   Phasing[1, ] <- c(0.08, 0.05)
   Phasing[2, ] <- c(0.05, 0.07)
   seq1 <- "TCAGACGGTAAGCTAGGTTAGCTTTAATCGGCGTTA"
   seq2 <- "TCAGGTATTACAGGTAGCTGATTAAAGCTCGCTAGCTAGGGATCCA"
   signal<- matrix(nrow=4, ncol=100)
   signal[1, ] <- SimulateCAFIE(seq1,flowOrders[1],Phasing[1,1],Phasing[1,2],0,numFlows[1])$sig
   signal[2, 1:50] <- SimulateCAFIE(seq1,flowOrders[2],Phasing[2,1],Phasing[2,2],0,numFlows[2])$sig
   signal[3, ] <- SimulateCAFIE(seq2,flowOrders[1],Phasing[1,1],Phasing[1,2],0,numFlows[1])$sig
   active_until <- c(100, 50, 100, 0)
   BaseCalls <- multreePhaser(signal, active_until, numFlows, flowOrders, Phasing, basecaller="treephaser-swan")
}
}
\author{
  Christian Koller
}
