rowColStringToRowCol <- function(rowColString) {
  if(grepl(":",rowColString[1])) {
    # New-style IDs
    rowcol <- gsub("^[^:]+:","",rowColString)
    col <- as.numeric(gsub(".+:","",rowcol))
    row <- as.numeric(gsub(":.+","",rowcol))
  } else {
    # Old-style IDs
    col <- as.numeric(gsub(".+\\|c","",rowColString))
    row <- as.numeric(gsub("r","",gsub("\\|.*","",rowColString)))
  }
  return(list(col=col,row=row))
}
