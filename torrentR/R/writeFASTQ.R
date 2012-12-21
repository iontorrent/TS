############################################################################
# Christian Koller; Sept 18 2012
# function writeFASTQ writes DNA Sequences to a .fastq file.
#
# Arguments     :
# fileNamePath  : Name base and path where the .fastq file should be saved
# sequences     : Vector of DNA sequences
# qualityValues : Vector of quality value strings
#                 default: NA which creates dummy quality strings
# wellRow       : Row indices of wells
# wellColumn    : Column indices of wells
# keySeq        : common key sequence at the start of the sequences
#                 default: "TCAG"
# keyPassFilter : If TRUE (default) the function does not write reads where the 
#                 first bases do not correspond to the key sequence and removes
#                 the key before writing the ones that pass.
############################################################################


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
{
  if (length(wellRow) != length(sequences)) {
    print("Vectors <wellRow> and <sequences> need to be of the same length. Nothing written.")
  }
  else if (length(wellColumn) != length(sequences)) {
    print("Vectors <wellColumn> and <sequences> need to be of the same length. Nothing written.")
  }
  else {
    fileName <- paste(fileNamePath, ".fastq", sep="")
    RunName  <- basename(fileNamePath)
      
    if (!(nchar(keySeq)>0))
      keyPassFilter <- FALSE
      
    keyPass <- 0
    keyFail <- 0
    nothingCalled <- 0
    
    # Creating file and overwriting old one
    if (appendFile == FALSE)
      cat("", file=fileName, sep="")
    
    for (i in 1:length(sequences)) {
      
      # Check for Keypass
      if ((keyPassFilter==FALSE) && (nchar(sequences[i])>0)) {
        keyPass <- keyPass + 1;
        if (is.na(qualityValues)) {
          ThisQualityString <- paste(rep(")",nchar(sequences[i])), sep="")
        }
        else {
          ThisQualityString <- qualityValues[i]
        }
        cat("@", RunName, ":", wellRow[i], ":", wellColumn[i], "\n", file=fileName, sep="", append=TRUE)
        cat(sequences[i], "\n", file=fileName, sep="", append=TRUE)
        cat("+\n", ThisQualityString, "\n", file=fileName, sep="", append=TRUE)
      }
      else if ((substr(sequences[i], 1,nchar(keySeq))==keySeq) && (nchar(sequences[i])>nchar(keySeq))) {
        keyPass <- keyPass + 1;
        if (is.na(qualityValues)) {
          ThisQualityString <- paste(rep(")",(nchar(sequences[i])-nchar(keySeq))), sep="")
        }
        else {
          ThisQualityString <- qualityValues[i]
        }
        cat("@", RunName, ":", wellRow[i], ":", wellColumn[i], "\n", file=fileName, sep="", append=TRUE)
        cat(substr(sequences[i], (nchar(keySeq)+1), nchar(sequences[i])), "\n", file=fileName, sep="", append=TRUE)
        cat("+\n", ThisQualityString, "\n", file=fileName, sep="", append=TRUE)
      }
      else {
        keyFail <- keyFail +1
        #print(paste("Length of failed sequence:", nchar(sequences[i])))
      }
    }
    if (keyPassFilter) {
      feedb <- "on"
    } else {
      feedb <- "off"
    }
    print(paste("Written", keyPass, "Sequences to", fileName, "with key-pass filter", feedb))
    print(paste(keyFail, "Sequences failed key pass or were too short."))
    logVec <- c(length(sequences), keyPass, keyFail)
    return (logVec)
  }
}



