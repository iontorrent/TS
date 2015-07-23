# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
# This script handles creating an NxN plot of N paired correlation functions.
# NOTE: Checks for sum(values) == 0 is used to prevent plot failures
# The optional 5th argument may be the name of a text table for output of presented r-values

options(warn=1)

args <- commandArgs(trailingOnly=TRUE)

nFileIn  <- ifelse(is.na(args[1]),"corplot.xls",args[1])
nFileOut <- ifelse(is.na(args[2]),"corplot.png",args[2])
nBarcode <- ifelse(is.na(args[3]),0,args[3])
title    <- ifelse(is.na(args[4]),"",args[4])
rValuOut <- ifelse(is.na(args[5]),"",args[5])

scalePicSize <- 128
minPicSize <- 3 * scalePicSize
maxPicSize <- 24 * scalePicSize

if( !file.exists(nFileIn) ) {
  write(sprintf("ERROR: Could not locate input file %s\n",nFileIn),stderr())
  q(status=1)
}

#col_bkgd = "#F5F5F5"
#col_plot = "#2D4782"
#col_frame = "#DBDBDB"
col_bkgd = "#E5E5E5"
col_plot = "#4580B6"
col_frame = "#CCCCCC"
col_title = "#999999"
col_line = "#D6D6D6"
col_grid = "#FFFFFF"
col_fitline = "goldenrod"

point_type = 19 ; # big spot
cor_cex = 2.3 ; # r2 & p-val text size

# read in matrix file and check expected format
bcrmat <- read.table(nFileIn, header=TRUE, sep="\t", as.is=TRUE, comment.char="")
ncols = ncol(bcrmat)
if( ncols < 2 ) {
  write(sprintf("ERROR: Expected at least 2 columns of data, including row ID field from bcmatrix file %s\n",nFileIn),stderr())
  q(status=1)
}
nrows = nrow(bcrmat)
if( nrows < 1 ) {
  write(sprintf("ERROR: Expected at least 1 row of data plus header line bcmatrix file %s\n",nFileIn),stderr())
  q(status=1)
}

# globals for collecting r-value matrix
r_value <- 0
nsams = ncols-1
r_matrx <- matrix( nrow=nsams, ncol=nsams )

# define panels to be output in pairs matrix

panel.density <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr=c(usr[1:2],0,1.05), lwd=2 )
  if( sum(x) == 0 ) {
    par(usr = c(0, 1, 0, 1))
    text(0.5, 0.5, "NA", cex=cor_cex )
  } else {
    d <- density(x,na.rm=TRUE)
    dm <- max(d$y)
    if( dm > 0 ) d$y <- d$y/max(d$y)
    lines(d,col=col_plot)
  }
}

panel.abline <- function(x, y, ...) {
  usr <- par("usr")
  points(x,y,col=col_plot,cex=0.7,pch=point_type)
  if( sum(x) > 0 && sum(y) > 0 ) {
    reg <- coef(lm(y ~ x))
    abline(coef=reg,col=col_fitline)
    slope <- sprintf("s=%.2f",reg[2])
    text( 0.85*usr[2]+0.15*usr[1], 0.95*usr[3]+0.05*usr[4], slope, cex=1.1)
  }
}

panel.cor <- function(x, y, digits=2, ...) {
  usr <- par("usr")
  par(usr = c(0, 1, 0, 1))
  if( sum(x) > 0 && sum(y) > 0 ) {
    r <- abs(cor(x,y,method="pearson"))
    txt <- sprintf("r = %.3f",r)
  } else {
    r <- 0
    txt <- "r = NA"
  }
  text(0.5, 0.5, txt, cex=cor_cex )
  r_value <<- r
}

# Define function to draw matrix of defined panels: Based on standard pairs() function but customized to get over limitations

corpairs <- function(x, labels, panel = points, ..., lower.panel = panel, upper.panel = panel, diag.panel = NULL, text.panel = textPanel, 
            label.pos = 0.5 + has.diag/3, cex.labels = NULL, font.labels = 1, row1attop = TRUE, gap = 1) {
  textPanel <- function(x = 0.5, y = 0.5, txt, cex, font) text(x, y, txt, cex = cex, font = font)
  localAxis <- function(side, x, y, xpd, bg, col = NULL, main, oma, ...) {
    if (side%%2 == 1) Axis(x, side = side, xpd = NA, lwd=0, lwd.ticks=1, ...)
    else Axis(y, side = side, xpd = NA, lwd=0, lwd.ticks=1, ...)
  }
  localPlot <- function(..., main, oma, font.main, cex.main) plot(...)
  localLowerPanel <- function(..., main, oma, font.main, cex.main) lower.panel(axes=FALSE,...)
  localUpperPanel <- function(..., main, oma, font.main, cex.main) upper.panel(...)
  localDiagPanel <- function(..., main, oma, font.main, cex.main) diag.panel(...)
  dots <- list(...)
  nmdots <- names(dots)
  if (!is.matrix(x)) {
    x <- as.data.frame(x)
    for (i in seq_along(names(x))) {
      if (is.factor(x[[i]]) || is.logical(x[[i]])) 
        x[[i]] <- as.numeric(x[[i]])
      if (!is.numeric(unclass(x[[i]]))) 
        stop("non-numeric argument to 'pairs'")
    }
  }
  else if (!is.numeric(x)) 
    stop("non-numeric argument to 'pairs'")
  panel <- match.fun(panel)
  if ((has.lower <- !is.null(lower.panel)) && !missing(lower.panel)) 
    lower.panel <- match.fun(lower.panel)
  if ((has.upper <- !is.null(upper.panel)) && !missing(upper.panel)) 
    upper.panel <- match.fun(upper.panel)
  if ((has.diag <- !is.null(diag.panel)) && !missing(diag.panel)) 
    diag.panel <- match.fun(diag.panel)
  if (row1attop) {
    tmp <- lower.panel
    lower.panel <- upper.panel
    upper.panel <- tmp
    tmp <- has.lower
    has.lower <- has.upper
    has.upper <- tmp
  }
  nc <- ncol(x)
  if (nc < 1) 
    stop("Must be at least one column in the argument to 'corpairs'")
  has.labs <- TRUE
  if (missing(labels)) {
    labels <- colnames(x)
    if (is.null(labels)) 
      labels <- paste("var", 1L:nc)
  }
  else if (is.null(labels)) 
    has.labs <- FALSE
  oma <- if ("oma" %in% nmdots) dots$oma else NULL
  main <- if ("main" %in% nmdots) dots$main else NULL
  if (is.null(oma)) {
    oma <- c(2, 2, 2, 2)
    if (!is.null(main)) oma[3L] <- 4
  }
  opar <- par(mfrow = c(nc, nc), mar = rep.int(gap/2, 4), oma = oma)
  on.exit(par(opar))
  dev.hold()
  on.exit(dev.flush(), add = TRUE)
  txtback = 0
  for (i in if (row1attop) 1L:nc else nc:1L) for (j in 1L:nc) {
    localPlot(x[, j], x[, i], xlab = "", ylab = "", axes = FALSE, type = "n", ...)
    if (txtback == 0) txtback = strheight("Q\n",cex=1.1)
    if (i == j || (i < j && has.lower) || (i > j && has.upper)) {
      mfg <- par("mfg")
      usr <- par("usr")
      # draw x-axis / labels
      if (i == nc) localAxis(1, x[, j], x[, i], ...)
      if (i == 1 ) {
        rect(usr[1],usr[4],usr[2],usr[4]+txtback,border=NA,col=col_frame,xpd=NA)
        mtext(labels[j],3,0,padj=-0.3,cex=1)
      }
      # draw y-axis / labels
      if (j == nc & i != 1) localAxis(4, x[, j], x[, i], ...)
      if (j == 1 && nc > 1 ) {
        rect(usr[1]-txtback,usr[3],usr[2],usr[4],border=NA,col=col_frame,xpd=NA)
        mtext(labels[i],2,0,padj=-0.3,cex=1)
      }
      #if (j == nc && nc > 1 ) text(usr[2],labels="y barcode",pos=4)
      rect(usr[1],usr[3],usr[2],usr[4],border=NA,col=col_bkgd)
      grid(col=col_grid,lty="solid")
      if (i == j) {
        if (has.diag) {
          localDiagPanel(as.vector(x[, i]), ...)
        } else if (has.labs) {
          par(usr = c(0, 1, 0, 1))
          if (is.null(cex.labels)) {
            l.wid <- strwidth(labels, "user")
            cex.labels <- max(0.8, min(2, 0.9/max(l.wid)))
          }
          text.panel(0.5, label.pos, labels[i], cex = cex.labels, font = font.labels)
        }
        r_matrx[i,i] <<- 1
      }
      else if (i < j) {
        # Something wrong here: calling the localLowerPanel() actually ends up calling panel.upper
        localLowerPanel(as.vector(x[, j]), as.vector(x[, i]), ...)
        r_matrx[i,j] <<- r_value
        r_matrx[j,i] <<- r_value
      } else {
        # Something wrong here: calling the upperLowerPanel() actually ends up calling panel.lower
        localUpperPanel(as.vector(x[, j]), as.vector(x[, i]), ...)
      }
      if (any(par("mfg") != mfg)) 
        stop("the 'panel' function made a new plot")
    }
    else par(new = FALSE)
  }
  if (!is.null(main)) {
    font.main <- if ("font.main" %in% nmdots) dots$font.main else par("font.main")
    cex.main <- if ("cex.main" %in% nmdots) dots$cex.main else par("cex.main")
    mtext(main, 3, 2, TRUE, 0.5, cex = cex.main, font = font.main, col=col_title)
  }
  invisible(NULL)
}

# Remove first column and any (annotation) columns after barcode data and take log2(x+1) of counts
bcrrep <- bcrmat[,-1,drop=FALSE]
if( nBarcode > 0 ) {
  bcrrep <- bcrrep[,1:nBarcode,drop=FALSE]
}
bcrrep <- log2(bcrrep+1)

# Create otuput window based on size of NxN matrix
bcdim <- if( is.null(ncol(bcrrep)) ) 1 else ncol(bcrrep)
sz = bcdim * scalePicSize
if ( sz < minPicSize ) sz = minPicSize
if ( sz > maxPicSize ) sz = maxPicSize
png(nFileOut,width=sz,height=sz)

if( title == "" ) {
  title = if( bcdim < 2 ) "log2 density plot" else "log2 pair correlation plots"
}

# Call the corpairs() functions with input matrix and custom styling...
corpairs( bcrrep, main=title, lower.panel=panel.abline, diag.panel=panel.density, upper.panel=panel.cor, gap=0.6, cex.axis=1 )

# Output optional r-values table
if( rValuOut != "" ) {
  bcnames <- colnames(bcrrep)
  colnames(r_matrx) <- bcnames
  rownames(r_matrx) <- bcnames
  write.table( r_matrx, file=rValuOut, sep="\t", quote=FALSE )
}

q()
