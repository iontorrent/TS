findDatDir <- function(
  analysisDir,
  paramFile = "processParameters.txt",
  paramName = "dataDirectory"
) {
  command <- sprintf("grep \"^%s = \" %s/%s | head -1 | cut -f3 -d \" \"",paramName,analysisDir,paramFile)
  datDir <- system(command,intern=TRUE)

  if(length(datDir)==0) {
    warning("Unable to determine datDir for %s\n",analysisDir)
    return(NULL)
  } else {
    return(datDir)
  }
}
