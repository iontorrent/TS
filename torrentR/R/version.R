torrentR_version <- function() {
  packageDescription("torrentR")$Version
}

plot_version <- function(
  side=1,
  line=4,
  adj=1,
  ...
) {
  mtext(sprintf("torrentR_v%s",torrentR_version()),side=side,line=line,adj=adj,...)
}
