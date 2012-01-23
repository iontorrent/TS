readDatList <- function(datDir) {
  command <- sprintf("ls %s",datDir)
  files <- system(command,intern=TRUE)
  isDat <- grepl("^acq_\\d{4}\\.dat$",files)
  datFiles <- sort(files[isDat])
  datFlows <- 1+as.numeric(gsub("\\.dat$","",gsub("^acq_","",datFiles)))
  return(list(datFiles=datFiles,datFlows=datFlows))
}
