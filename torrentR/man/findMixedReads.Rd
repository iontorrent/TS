\name{findMixedReads}
\alias{findMixedReads}
\title{
  Estimate the (col,row) coordinates of mixed reads using results from libMixtureAnalysis()
}
\description{
  Applies the model returned from libMixutureAnalysis to all library reads to estimate the (col,row) coordinates
  of those which are mixed.
}
\usage{
  findMixedReads(
    dataDir,
    libKeySeq,
    mixResults,
    keySnrThreshold = 3
  )
}
\arguments{
  \item{dataDir}{
    The directory in which the 1.wells and bfmask.bin files are located.
  }
  \item{libKeySeq}{
    The key sequence for library fragments
  }
  \item{mixResults}{
    a list object returned from libMixtureAnalysis() which identifies which flows should be used
    for estimation of mixed reads and the mixture model to apply to them.
  }
  \item{keySnrThreshold}{
    The SNR threshold to apply to key sequences, beads with values below this will not be considered as keypass.
  }
}
\author{
Simon Cawley
}

\seealso{
  \code{\link{libMixtureAnalysis}}
}
\examples{

dataDir   <- "/data/my_analysis_directory"
libKeySeq <- "TCAG"
tfKeySeq  <- "ATCG"
tfSeq     <- "ATCGTAGCGTACATCGCGCATCTATATATCGTCAACTACGCTGAGTCGGAGACACGCAGGGATGAGATGG"

## Perform mixture analysis, use to apply to all library reads
#mixModel <- libMixtureAnalysis(dataDir, tfKeySeq, libKeySeq, tfSeq)
#mixRowCol <- findMixedReads(dataDir, libKeySeq, mixModel)
}
