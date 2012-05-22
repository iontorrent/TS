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
    # Check for TF.sam.parsed - the new TFTracking.txt
    tfSamParsedFile <- paste(analysisDir,.Platform$file.sep,"TF.sam.parsed",sep="")
    if(file.exists(tfSamParsedFile)) {
      if(0==file.access(tfSamParsedFile,4)) {
        result <- readSamParsed(tfSamParsedFile,fields=c("name","tName","q7Len","q10Len","q17Len"))
        names(result)[3] <- "tfSeq"
        return(data.frame(result,stringsAsFactors=FALSE))
      } else {
        warning(sprintf("TF tracking file %s exists but is not readable\n",tfSamParsedFile))
        return(NULL)
      }
    } else {
      warning(sprintf("Did not find either of the possible TF tracking files:\n%s\n%s\n",tfStatFile,tfSamParsedFile))
      return(NULL)
    }
  }
}
