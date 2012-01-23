formImageMatrix <- function(x,y,z,maxX,maxY) {
  ret <- rep(NA,maxX*maxY)
  ret[1+(y*maxX+x)] <- z
  return(matrix(ret,maxX,maxY))
}
