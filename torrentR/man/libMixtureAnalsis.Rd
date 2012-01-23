\name{libMixtureAnalysis}
\alias{libMixtureAnalysis}
\title{
  Analysis of library read mixtures
}
\description{
  Estimates the proportion of non-clonal wells among the library reads, using test fragment reads to estimate
  the typical locations of zero-mers and one-mers.
}
\usage{
  libMixtureAnalysis(
    dataDir,
    libKeySeq,
    plotDir = NA,
    setName = NA,
    nSample = 10000,
    densityBandwidth = 0.05,
    keySnrThreshold = 3,
    minLibNumber=5000,
    plotType = c("none", "png", "bitmap"),
    ret = TRUE
  )
}
\arguments{
  \item{dataDir}{
    The directory in which the 1.wells and bfmask.bin files are located.
  }
  \item{libKeySeq}{
    The key sequence for library fragments
  }
  \item{plotDir}{
    Directory in which plots will be created.  Optional.  If used, caller should ensure that it exists first.
  }
  \item{setName}{
    The name that will be used to refer to this dataset in plot titles.  Optional, default value is the base name of the data directory.
  }
  \item{nSample}{
    The maximum number of library and test fragment beads that will be sampled when assessing mixtures.
  }
  \item{densityBandwidth}{
    The value of the "bw" parameter that will be used in calls to density().  
  }
  \item{keySnrThreshold}{
    The SNR threshold to apply to key sequences, beads with values below this will not be considered as keypass.
  }
  \item{minLibNumber}{
    Analysis will be performed only if the number of library beads is at least this large.
  }
  \item{plotType}{
    Type of plots to produce - default value is "none".  Other options are "png" and "bitmap" which will both produce
    png files, the latter option may be requied when running in situations where there is no X running.
  }
  \item{ret}{
    If this is set to TRUE the function will return summary details and the estimated signal densities which can be used
    for testing other beads for mixing.
  }
}
\author{
  Simon Cawley
}

\seealso{
  \code{\link{findMixedReads}}
}
\examples{

dataDir   <- "/data/my_analysis_directory"
libKeySeq <- "TCAG"
tfKeySeq  <- "ATCG"
tfSeq     <- "ATCGTAGCGTACATCGCGCATCTATATATCGTCAACTACGCTGAGTCGGAGACACGCAGGGATGAGATGG"

# Perform mixture analysis and store results
#result <- libMixtureAnalysis(dataDir, tfKeySeq, libKeySeq, tfSeq)
#str(result)

# Perform mixture analysis and generate summary plots
#plotDir <- "my_plots"
#system(paste("mkdir ",plotDir))
#libMixtureAnalysis(dataDir, tfKeySeq, libKeySeq, tfSeq, plotDir=plotDir, plotType="png", ret=FALSE)
}
