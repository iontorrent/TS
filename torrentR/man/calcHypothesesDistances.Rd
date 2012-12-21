\name{calcHypothesesDistances}
\alias{calcHypothesesDistances}
\title{
  Calculate the squared distances betweeen hypotheses base sequences.
}
\description{
  The function calculates the squared distances betweeen the predicted ionograms of 
  hypothesis base sequences to normalized observed sequences. Bases before startFlow and 
  after the end of the hypothesis are filled in by the solver based on the signal.
  Only flows are taken into where the predictions of hypotheses i, i in {1,...,N},
  differ from the predictions of hypothesis 0 by 0.05 or more.
}
\usage{
 calcDistances <- function(
  signal,
  cf,
  ie,
  dr,
  flowOrder,
  hypotheses,
  startFlow = 0,
  normalize = 0,
  verbose = 0
)
}
\arguments{
  \item{signal}{
    The vector of (key) normalized measured signal values.
    Adaptive normalization is performed for every hypothesis sequence and the
    resulting normalized values are returned in field "Normalized"
  }
  \item{flowOrder}{
    The flow cycle - for example "TACG".
  }
  \item{cf,ie,dr}{
    Estimates for cf, ie and dr.
  }
  \item{hypotheses}{
    String Vector containing the base space hypotheses.
  }
  \item{startFlow}{
    Flow which corresponds to the start of the hypotheses. Flows up to startflow
    are solved and a corresponding base prefix is attached to the hypotheses.
  }
  \item{normalize}{
  	If normalize>0, the signal values are adaptively normalized w.r.t the hypothesis prediction.ß
  }
  \item{verbose}{
  	If verbose>0, the functions prints out some text.
  }
}
\value{
  The return value is a list with the following elements.
  \item{DistanceObserved}{
    Squared distance of the normalized signal and the prediction for the hypotheses.
  }
  \item{DistanceHypotheses}{
    Squared distance of the prediction for hypotheses zero and i.
  }
  \item{Predictions}{
    Matrix of prediced signal values given the base seqences.
    Rows: different hypotheses Columns: Predicted signal for flows
  }
  \item{Normalized}{
  	Normalized measurements.
  	- Equal to input signal if normalize==0
  	- Adaptively normalized w.r.t hypothesis prediction if normaize>0
  	Rows: different hypotheses Columns: Normalized flow signals
  }  
  
}
\seealso{
  torrentR package
}
\examples{
\dontrun{
	key   <-"TCAG"
	mySeq <- paste(key, "GGCGCCAGGCGTTGAAGATACGCAGCGGGGCAAGCTATCCCCAAGGCTTCGG", sep="")
	Hyp1 <- "GGCGCCAGGCGTTGAAGATACGCAGCGGGGCAAGCTATCC"
	Hyp2 <- "GGCGCCAGGCGTTGAAGATACGCAGCGGGCAAGCTATCC"
	Hyp3 <- "GGCGCCAGGCGTTGAAGATACGCAGCGGGGGCAAGCTATCC"
	flow  <- "TACGTACGTCTGAGCATCGATCGATGTACAGC"
	startFlow <- 7
	cf    <- 0.01
	ie    <- 0.005
	dr    <- 0.001
	nflow <- 100
	signal <- SimulateCAFIE(mySeq,flow,cf,ie,dr,nflow)
	vals <- calcHypothesesDistances(signal$sig,cf,ie,dr,flow,c(Hyp1, Hyp2, Hyp3), startFlow,1)
}
}
\author{
  CK
}
