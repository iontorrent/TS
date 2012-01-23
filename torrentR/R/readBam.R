readBam <- function(bamFile) {

  data <- .Call("readBam",
               bamFile,
               PACKAGE="torrentR"
  )

  if(any(names(data)=="name")) {
      rowColString <- data$name
      data <- data[-which(names(data)=="name")]
      data <- c(rowColStringToRowCol(rowColString),data)
  }
  data
}
