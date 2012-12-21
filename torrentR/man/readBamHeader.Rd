\name{readBamHeader}
\alias{readBamHeader}
\title{
  Read header section of a BAM file
}
\description{
  Read BAM file header
}
\usage{
  read(
    bamFile
  )
}
\arguments{
  \item{bamFile}{
    Name of the BAM file to read
  }
}
\value{
    \item{ReadGroup}{
      A list of character vectors, each vector has one entry for each read group.
      Entries are an empty string where not specified in the BAM.  Each vector
      corresponds to one of the RG tags as defined in the SAM format specification
      defined at http://samtools.sourceforge.net/SAM1.pdf
    }
    \item{Sequence}{
      A list of character vectors, each vector has one entry for each reference sequence.
      Entries are an empty string where not specified.  Each vector
      corresponds to one of the SQ tags as defined in the SAM format specification
      defined at http://samtools.sourceforge.net/SAM1.pdf
    }
}
\author{
  Simon Cawley
}
\seealso{
  \code{\link{readIonBam}}
}
