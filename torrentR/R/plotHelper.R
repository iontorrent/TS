plotHelper <- function(plotFile,plotType=c("png","bitmap"),height,width) {
  plotType <- match.arg(plotType)
  if(plotType == "png")
    png(plotFile,height=height,width=width)
  else 
    bitmap(plotFile,type="png16m",units="px",height=height,width=width)
}
