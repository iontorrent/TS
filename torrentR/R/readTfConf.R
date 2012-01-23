readTfConf <- function(analysisDir,filename="DefaultTFs.conf",defaultDir="/opt/ion/config") {

  # Set the search path for the TF file
  searchPath <- paste(analysisDir)
  if(length(Sys.getenv("HOME") > 0))
    searchPath <- c(searchPath,Sys.getenv("HOME"))
  searchPath <- c(searchPath,getwd())
  searchPath <- c(searchPath,defaultDir)

  # Look for the TF file
  tfFile <- NA
  for(dir in searchPath) {
    fileName <- paste(dir,.Platform$file.sep,filename,sep="")
    if(file.exists(fileName)) {
      if(0==file.access(fileName,4)) {
        tfFile <- fileName
        break;
      } else {
        warning(sprintf("TF file %s exists but is not readable, skipping\n",fileName))
      }
    }
  }
  if(!(is.na(tfFile))) {
    zz <- scan(tfFile,what=rep(list("character"),3),sep=",",quiet=TRUE)
    tf <- data.frame(tf=zz[[1]],keySeq=zz[[2]],tfSeq=zz[[3]],stringsAsFactors=FALSE)
    tf$tf <- gsub(" ","_",tf$tf)
    return(tf)
  } else {
    return(NULL)
  }
}
