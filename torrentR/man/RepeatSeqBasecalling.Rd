\name{RepeatSeqBasecalling}
\alias{RepeatSeqBasecalling}
\title{
  Generates Basecalls from multiple wells files corresponding to the same chip, taking advantage of
  multiple wells file jointly.
}
\description{
  The function generates Basecalls from multiple wells files corresponding to the same chip. Uses 3
  input files per run, 1.wells, analysis.bfmask.bin, and BaseCaller.json. 
  Assume x in {1,2,3}, then naming of the files should be
  1.x.wells
  analysis.bfmask.x.bin
  BaseCaller.x.json
  }
\usage{
  RepeatSeqBasecalling <- function(
  DataFolder,
  numRuns,
  chipRegion=NA,
  unionOfReads=FALSE
  combinations=NA
  outputFolder="."
)
}
\arguments{
  \item{DataFolder}{
    Path where the input files are stored (without last slash)
  }
  \item{numRuns}{
  	Number of different runs in DataFolder. (Range of parameter x in Description)
  }
  \item{chipRegions}{
    A chip is divided in 8x8 regions, numbered 0-63. chipRegions is a vector input which
    specifies which regions should be processed. Default: NA -> all regions are processed
  }
  \item{combinations}{
    Logical vector / matrix with rows of length numRuns, indicating which runs should
    be called jointly. Default: All runs are called jointly.
    Please put the case that inolves the most runs in row 1. 
  }
  \item{unionOfReads}{
  	Logical switch that determines whether the union or intersection set of reads
  	across the runs is called.
  }
  \item{outputFolder}{
  	Folder where output files are written. Default: current directory
  }
}
\value{
  There is no return value, but 2 files are generated per row in combinations
  1) BaseCalls.x.y.z.fastq, containing called sequences in fastq format
  2) log.txt, number of called sequences, sequences that key-pass, and fail per region
}
\examples{ 
	\dontrun{
	combination <- matrix(nrow=3, ncol=3)
	combination[1, ] <- c(TRUE, TRUE, TRUE)
	combination[2, ] <- c(TRUE, FALSE, TRUE)
	combination[3, ] <- c(TRUE, FALSE, FALSE)
	
	RepeatSeqBasecalling("./", 3, c(18, 45), combinations=combination)
    }
}
\author{
  Christian Koller
}
