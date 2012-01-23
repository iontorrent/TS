readTfStats <- function(analysisDir,filename="TFTracking.txt") {
  tfStatFile <- paste(analysisDir,.Platform$file.sep,filename,sep="")
  if(file.exists(tfStatFile)) {
    if(0==file.access(tfStatFile,4)) {
      tfStats <- scan(tfStatFile,what=c(rep(list("numeric"),2),list("character")),sep=",",comment.char="#",quiet=TRUE)
      tfStats[[3]] <- gsub(" ","_",tfStats[[3]])
      tfStats <- data.frame(col=as.numeric(tfStats[[2]]),row=as.numeric(tfStats[[1]]),tfSeq=tfStats[[3]],stringsAsFactors=FALSE)
      return(tfStats)
    } else {
      warning(sprintf("TF tracking file %s exists but is not readable\n",tfStatFile))
      return(NULL)
    }
  } else {
    warning(sprintf("TF tracking file %s does not exist\n",tfStatFile))
    return(NULL)
  }
}
