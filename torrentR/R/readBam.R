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

readBamWithLocationFilter <- function(bamFile, seq_loc,seq_range) {

  data <- .Call("readBamWithLocationFilter",
               bamFile, seq_loc, seq_range,
               PACKAGE="torrentR"
  )

  if(any(names(data)=="name")) {
      rowColString <- data$name
      data <- data[-which(names(data)=="name")]
      data <- c(rowColStringToRowCol(rowColString),data)
  }
  data
}

readBamWithSpatialFilter <- function(bamFile, col_min, col_max, row_min,row_max) {

  data <- .Call("readBamWithSpatialFilter",
               bamFile, col_min,col_max,row_min,row_max,
               PACKAGE="torrentR"
  )

  if(any(names(data)=="name")) {
      rowColString <- data$name
      data <- data[-which(names(data)=="name")]
      data <- c(rowColStringToRowCol(rowColString),data)
  }
  data
}
