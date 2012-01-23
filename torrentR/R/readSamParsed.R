readSamParsed <- function(samFile,fields=c("name","q10Len"),nlines=0) {
  knownFieldTypes <- list(
    "name"       = character(),
    "strand"     = numeric(),
    "tStart"     = numeric(),
    "tLen"       = numeric(),
    "qLen"       = numeric(),
    "match"      = numeric(),
    "percent.id" = numeric(),
    "q10Errs"    = numeric(),
    "homErrs"    = numeric(),
    "mmErrs"     = numeric(),
    "indelErrs"  = numeric(),
    "qDNA.a"     = character(),
    "match.a"    = character(),
    "tDNA.a"     = character(),
    "tName"      = character(),
    "start.a"    = numeric(),
    "q7Len"      = numeric(),
    "q10Len"     = numeric(),
    "q17Len"     = numeric(),
    "q20Len"     = numeric(),
    "q47Len"     = numeric()
  )
  header <- scan(samFile,nlines=1,what=character(),quiet=TRUE)
  if(!all(is.element(fields,header))) {
    stop(sprintf("Failed to find fields \"%s\" in %s\n",paste(fields[which(!is.element(fields,header))],collapse="\",\""),samFile))
  } else {
    if(!all(is.element(fields,names(knownFieldTypes))))
      warning(sprintf("Using type \"character\" for unrecognized field types \"%s\"\n",paste(fields[which(!is.element(fields,names(knownFieldTypes)))],collapse="\",\"")))
    fieldTypes <- rep(list(character()),length(header))
    names(fieldTypes) <- header
    knownFieldTypes <- knownFieldTypes[is.element(names(knownFieldTypes),header)]
    fieldTypes[names(knownFieldTypes)] <- knownFieldTypes
    data <- scan(samFile,what=fieldTypes,nlines=nlines,skip=1,sep="\t",quiet=TRUE)[fields]
    if(any(names(data)=="name")) {
      rowColString <- data$name
      data <- data[-which(names(data)=="name")]
      data <- c(rowColStringToRowCol(rowColString),data)
    }
    return(data)
  }
}
