readTSV <- function(inFile,what=NULL,sep="\t",nlines=0,header=NULL,charFields=NULL) {
  if(is.null(header)) {
    header <- scan(inFile,nlines=1,what=character(),quiet=TRUE,sep=sep)
    headerSkip <- 1
  } else {
    headerSkip <- 0
  }
  if(is.null(what)) {
    what <- rep(list(numeric()),length(header))
    if(!is.null(charFields))
      what[charFields] <- rep(list(character()),length(charFields))
  }
  body <- scan(inFile,skip=headerSkip,what=what,quiet=TRUE,sep=sep,nlines=nlines)
  names(body) <- header
  return(body)
}
