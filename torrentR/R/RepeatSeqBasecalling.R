##########################################################################################
# Christian Koller, 21. Sept 2012
#
# assumptions: wells, bfmask, and json of multiple runs available in Data Folder
#              Naming: 1.x.wells
#                      analysis.bfmask.x.bin
#                      BaseCaller.x.json
# Arguments    :
# DataFolder   : Path to data files
# numRuns      : Number of different runs in the folder (Range of x in above naming)
# chipRegion   : chip is divided in 8x8 regions. Vector input with values of 0-63 specifies which
#                regions should be processed. Default: NA all regions are processed
# combinations : Logical vector / matrix with rows of length numRuns, indicating which runs should
#                be called jointly. Example: c(TRUE FALSE TRUE) calls runs 1 and 3 jointly
#                Default: All runs are called jointly.
# unionOfReads : TRUE  call wells where at least one run is not filtered out
#                FALSE (default) call only wells that were not filtered in all runs.
# outputFolder : Folder where files are written
##########################################################################################

RepeatSeqBasecalling <- function(
  DataFolder,
  numRuns,
  chipRegion=NA,
  unionOfReads=FALSE,
  combinations=NA,
  outputFolder = "."
)
{
    #loading required libraries
    library(rjson)
    #library(torrentR)
    #library(testPackage)
    #source("writeFASTQ.R")
    
    # Removing last slash from path if here is one
    DataFolder <- dirname(paste(DataFolder, "/dummy", sep=""))
    outputFolder <- dirname(paste(outputFolder, "/dummy", sep=""))
    
    # If chipRegion is not specified, whole chip is being analyzed
    if (is.na(chipRegion))
      chipRegion <- 0:63

    # create combinations logical matrix; by default combine all reads
    if (is.na(combinations))
      combinations <- rep(TRUE, numRuns)
    if (!is.matrix(combinations))
      combinations <- matrix(combinations, nrow=1)
    
    # Load the Basic information about the runs
    flowOrders <- vector(length=numRuns)
    numFlows <- vector(length=numRuns)
    Phasing <- matrix(nrow=numRuns, ncol=2)
    
    RunJSON <- list()
    RunBasics <- list()
    Wells <- list()
    flowOrder <- vector(length=numRuns)
    numFlows <- vector(length=numRuns)
    nIndex <- 1:numRuns
    # To compare only reads that are called by all the runs, chcek which ones are called
    activeRuns <- vector(length=numRuns)
    
    for (i in 1:numRuns){
      # Load basic run information from wells file
      wellPath <- paste(DataFolder, "/1.", i, ".wells", sep="")
      bfFile <- paste("analysis.bfmask.", i, ".bin", sep="")
      RunBasics[[i]] <- readWells(wellPath, bfMaskFileName=bfFile, row=0, col=0)
      flowOrder[i] <- RunBasics[[i]]$flowOrder
      numFlows[i] <- RunBasics[[i]]$nFlow
      # Load Basecaller.json file
      RunJSON[[i]] <- fromJSON(file=paste(DataFolder, "/BaseCaller.", i, ".json", sep=""))
      activeRuns[i] <- any(combinations[ ,i])
    }
    libIndex <- nIndex[activeRuns]
      
    nRegionsRow <- 8
    nRegionsCol <- 8
    region_size_row <- ceiling(RunBasics[[1]]$nRow / nRegionsRow) # y
    region_size_col <- ceiling(RunBasics[[1]]$nCol / nRegionsCol) # x
    
    # xxx add combinations ------------------
    for (c in 1:nrow(combinations)) {
      
      ptm <-proc.time()
      LogNrSeqCalled <- matrix(nrow=(max(chipRegion)+1), ncol=3)
      combination <- combinations[c, ]
      NrReads <- sum(combination)
      nIndex <- 1:numRuns
      nIndex <- nIndex[combination]
      print(paste( "Calling Bases for run combination: ", paste(nIndex, sep="", collapse=" ")))
    
      # Process each region individually
      for (Reg in chipRegion){
        
        print(paste("Calling bases in region", Reg))
        # Compute region boundaries - numbering starts at zero
        cMin <- floor(Reg / nRegionsCol) * region_size_col
        cMax <- cMin + region_size_col - 1
        rMin <- floor(Reg %% nRegionsRow) * region_size_row
        rMax <- rMin + region_size_row - 1
        
        # Load region specific data, phasing and raw wells for all runs considered
        for (i in libIndex) {
          Phasing[i,1] <- RunJSON[[i]]$Phasing$CFbyRegion[(Reg+1)]
          Phasing[i,2] <- RunJSON[[i]]$Phasing$IEbyRegion[(Reg+1)]
          
          wellPath <- paste(DataFolder, "/1.", i, ".wells", sep="")
          bfFile <- paste("analysis.bfmask.", i, ".bin", sep="") 
          Wells[[i]] <- readWells(wellPath, bfMaskFileName=bfFile, colMin=cMin, colMax=cMax, rowMin=rMin, rowMax=rMax)
        }
        
        # Combine wells files into structure that can be read by multiread Treephaser
        signal <- matrix(nrow=(NrReads*(Wells[[nIndex[1]]]$nLoaded)), ncol=max(numFlows[combination]))
        active_until <- vector(mode="integer", length=(NrReads*(Wells[[nIndex[1]]]$nLoaded)) )
        maskVector <- vector(mode="logical", length=Wells[[nIndex[1]]]$nLoaded)
        iRead <- 1
        
        for (i in 1:Wells[[nIndex[1]]]$nLoaded) {
            
            # Switch whether union or intersection set of reads should be called
            if (unionOfReads) {
              readExcluded <- rep(TRUE, numRuns)
            }
            else
              readExcluded <- rep(FALSE, numRuns)
            
            for (j in libIndex) {
              readExcluded[j] <- (Wells[[j]]$mask$lib[i]==0) || (Wells[[j]]$mask$pinned[i]==1) || (Wells[[j]]$mask$ignore[i]==1) || (Wells[[j]]$mask$washout[i]==1)
              # See if readWells function (old) only returns limited number mask fields
              allMaskFields <- "exclude" %in% names(Wells[[j]]$mask)
              if (allMaskFields)
                readExcluded[j] <- (readExcluded[j] || (Wells[[j]]$mask$exclude[i]==1) || (Wells[[j]]$mask$keypass[i]==1) || (Wells[[j]]$mask$filteredBadKey[i]==1) || (Wells[[j]]$mask$filteredShort[i]==1) || (Wells[[j]]$mask$filteredBadPPF[i]==1) || (Wells[[j]]$mask$filteredBadResidual[i]==1))
            }
            
            if (unionOfReads) {
              callThisWell <- any(readExcluded==FALSE)
            }
            else
              callThisWell <- all(readExcluded==FALSE)
                                    
            if (callThisWell) {
                maskVector[i] <- TRUE
                for (j in nIndex) {
                    signal[iRead, 1:numFlows[j]] <- Wells[[j]]$signal[i, ]
                    
                    if (readExcluded[j] == FALSE) {
                        active_until[iRead] <- Wells[[j]]$nFlow
                    } else {
                        active_until[iRead] <- 0
                    }
                    iRead <- iRead + 1
                }
            }
            else
                maskVector[i] <- FALSE
        }
                
        # -------------------
        # Call Bases if there are any to call
        if (iRead > 1) {
          
          signal <- signal[1:(iRead-1), ]
          active_until <- active_until[1:(iRead-1)]
            
          BaseCalls <- multreePhaser(signal, active_until, numFlows[combination], flowOrder[combination], keySeq="TCAG", Phasing[combination, ], basecaller="treephaser-swan")
            
               
          # ------------------- XXX
          # write Sequnces to fastq file
          output_file_base <- paste(outputFolder, "/BaseCalls.", paste(nIndex, sep="", collapse="."), sep="")

          
          # Adjust file name convention to Nils'
          #if (length(nIndex)==1) {
          #  tempstr <- ".reseq"
          #} else {
          #  tempstr <- paste(".", (c-1), sep="")
          #}
          #output_file_base <- paste(DataFolder, "/", basename(DataFolder), ".", paste((libIndex-1), sep="", collapse="."), tempstr, sep="")
        
          if (Reg==chipRegion[1]) {
            appendF = FALSE
          }
          else {
            appendF = TRUE
          }
            
          logVec <- writeFASTQ(output_file_base, BaseCalls$seq, NA, Wells[[nIndex[1]]]$row[maskVector], Wells[[nIndex[1]]]$col[maskVector], keyPassFilter=TRUE, appendFile=appendF)
          LogNrSeqCalled[(Reg+1), ] <- logVec
          
          print(proc.time() - ptm)
        }
      }
      # Save number of sequences being called
      write.table(LogNrSeqCalled, paste(output_file_base, ".log.txt", sep=""), row.names=FALSE, col.names=FALSE)
    }
    
}