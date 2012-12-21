\name{writeFASTQ}
\alias{writeFASTQ}
\title{
  Writes a vetor of sequences into a .fastq file
}
\description{
  Writes a string vector of sequences into a .fastq file}
\usage{
  writeFASTQ <- function (
    fileNamePath,
    sequences,
    qualityValues = NA,
    wellRow,
    wellColumn,
    keySeq="TCAG",
    keyPassFilter=TRUE,
    appendFile=FALSE
)
}
\arguments{
  \item{fileNamePath}{
    Name base and path where the .fastq file should be saved
  }
  \item{sequences}{
  	Vector of DNA sequences
  }
  \item{qualityValues}{
    Vector of quality value strings (same length as sequences)
    default: NA which creates dummy quality strings in the fastq file
  }
  \item{wellRow}{
    Vector of row indices of wells that created the sequences.
  }
  \item{wellColumn}{
  	Vector of column indices of wells that created the sequences.
  }
  \item{keySeq}{
    The known key sequence at the start of the read. Default: "TCAG"
  }
  \item{keyPassFilter}{
    If TRUE (default) sequences whose first called bases do not correspond to the
    keySeq are not written to the fastq file.
  }
  \item{appendFile}{
  	If TRUE, the function does not overwrite an existing file but appends the
  	information to the end of the file. Default: FALSE
  }
}
\value{
  The return value is a vector with three elements.
  1) number of sequences at the input
  2) number of sequences that had the correct key sequence at the beginning
  3) number of sequences that failed key-pass or were too short (length 0 after removal of key)
}
\examples{ 
	\dontrun{		
	Seq = c("TCAGACGGTAAGCTAGGTTAGCTTTAATCGGCGTTA", "TCAGGTATTACAGGTAGCTGATTAAAGCTCGCTAGCTAGGGATCCA")
	logVec <- writeFASTQ("MyFastq", Seq, NA, c(0,1), c(1,3))
}
}
\author{
  Christian Koller
}
