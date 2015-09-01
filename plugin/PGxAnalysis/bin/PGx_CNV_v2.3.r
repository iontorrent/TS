# Last update: 05/27/2015
# Author: Shann-Ching Chen
# PGx_CNV_v2.4.r inputs (4 mandatory parameters):  args[1]: CovPATH, args[2]: OutDIR, args[3]: EM_CNV_v2.4.r, args[4]: gc_bedfile
# PGx_CNV_v2.4.r output1: CNV vcfs outputed in OutDIR
# PGx_CNV_v2.4.r output2: Summary Report LogFile: number of barcodes, QC, NOCALLs, etc

THRES.COV = 100
THRES.Uniformity = 80
THRES.E9SKEW = 2
THRES.ExonPHRED = 15  
THRES.GenePHRED = 50
THRES.normampMIN = 20
THRES.CN0 = -2.8
THRES.CN4 = 3
THRES.EMMin = 7
OutputPDF = FALSE
THRES.AmpliconCV = 0.2
THRES.NormAmplicons = 20
THRES.DepthCV = 0.5
ErrorLog = ""
BCFILE_MIN_SIZE = 50000

exon9_amp = c("SP_47.1.274977")
gene_amps = c("SP_47.2.92998","SP_47.2.120524","CYP2D6_cnv_s1_1.2.136070","CYP2D6_cnv_44.3.296145","CYP2D6_cnv_44.3.77543","CYP2D6_cnv_44.3.101897","CYP2D6_cnv_44.4.64739","CYP2D6_cnv_44.3.195567","SP_816.151728")
CNV_amplicons = c(exon9_amp, gene_amps)

args = commandArgs(TRUE)

if(length(args) < 1) {
    cat("No arguments supplied. Usage:\n")
    helpstr = "Rscript PGx_CNV_v2.4.r CoverageAnalysisPATH EM_Rscript
    Example :
    Rscript PGx_CNV_v2.4.r CoverageAnalysisPATH CNV_vcf EM_CNV_v2.4.r"
    cat (helpstr)
    ##supply default values

    quit(status=1)
} else {
    CovPATH = args[1]
    cat("PATH for coverage analysis is ",CovPATH,"\n")
    if (!file.exists(CovPATH)){    stop("Coverage analysis results do not exist");    }
    CNTruth = ""; 
    if(length(args) == 1) {
        # default OutDIR = "CNV_vcf"
        OutDIR = paste(getwd(), .Platform$file.sep, "CNV_vcf", sep="")
        CNTruth = "";
	    SUB_IDX = NULL;
    } else {
        SUB_IDX = NULL;
        OutDIR = args[2];
	    EM_Rscript = args[3];
	    if(length(args)>=4) {		   gc_bedfile = args[4]		}
    }
    cat("Output directory for CNV vcfs is ",OutDIR,"\n")
    if (!file.exists(OutDIR)){    
        dir.create(OutDIR)
    } else {
        cat(OutDIR," already exists\n")
    }
}
LogFile = paste(OutDIR,.Platform$file.sep,"CNV",as.character(Sys.Date()),".log",sep="")

# read in bc_summary.xls file
BC_FileName1 = list.files(CovPATH,pattern="*bc_summary.xls")
if(length(BC_FileName1) != 1) {
    stop("Coverage file bc_summary.xls does not exist")
} else {
    BCSummary = read.table(paste(CovPATH,.Platform$file.sep,BC_FileName1,sep=""), sep="\t", header=T)
	if(length(SUB_IDX) > 0) {
        BCSummary = BCSummary[SUB_IDX,]
    }
}

# read in bcmatrix.xls file
BC_FileName2 = list.files(CovPATH,pattern="*bcmatrix.xls")
if(length(BC_FileName2) != 1) {
    stop("Coverage file bcmatrix.xls does not exist")
} else {
    BCMatrix = read.csv(paste(CovPATH,.Platform$file.sep,BC_FileName2,sep=""), sep="\t", na.strings=0)	
	BCMatrix[is.na(BCMatrix)] <- 0
	
    if(length(SUB_IDX) > 0) {
        BCMatrix = BCMatrix[,c(1,2,SUB_IDX+2)]
    }	
}

BCSummary_E9 = BCSummary
BCSummary_E9$E9_fwd_reads = ""
BCSummary_E9$E9_rev_reads = ""
BCSummary_E9$E9_Reads_ADJ = ""
BCSummary_E9$E9_SKEW = ""

if(file.exists(EM_Rscript)) {
    source(EM_Rscript)
} else {
    stop(paste("The code for CNV inference does not exist.  Please reinstall the PGx plugin.\n",sep=""))
}

# check if BCMatrix contains all the 10 necessary CYP2D6 amplicons
CYP2D6ind = match(CNV_amplicons, BCMatrix[,2])
if(any(is.na(CYP2D6ind))) {
    stop("The Coverage Analysis file does not contain all necessary CYP2D6 amplicons.  Please check your Target region bed file.")
}

AmpliconCovPath = paste(CovPATH, .Platform$file.sep, BCSummary$Barcode.ID,.Platform$file.sep, sep="")
for(a_idx in 1:length(AmpliconCovPath)) {
        BarcodeICov = read.table(paste(AmpliconCovPath[a_idx],.Platform$file.sep,list.files(AmpliconCovPath[a_idx],pattern="*amplicon.cov.xls")[1],sep=""), sep="\t", header=T)
	e9_idx = which(BarcodeICov$region_id == exon9_amp)
	BCSummary_E9$E9_fwd_reads[a_idx] = BarcodeICov$fwd_reads[e9_idx]
	BCSummary_E9$E9_rev_reads[a_idx] = BarcodeICov$rev_reads[e9_idx]
	BCSummary_E9$E9_Reads_ADJ[a_idx] = max(BarcodeICov$fwd_reads[e9_idx],BarcodeICov$rev_reads[e9_idx])*2
	if(BarcodeICov$fwd_reads[e9_idx] == 0) {
    	BCSummary_E9$E9_SKEW[a_idx] = 0
	} else if (BarcodeICov$rev_reads[e9_idx] == 0) {
	    BCSummary_E9$E9_SKEW[a_idx] = 0
	} else {
	    if( BarcodeICov$fwd_reads[e9_idx]>BarcodeICov$rev_reads[e9_idx] ) {
        	BCSummary_E9$E9_SKEW[a_idx] = BarcodeICov$fwd_reads[e9_idx] / BarcodeICov$rev_reads[e9_idx]		
		} else {
        	BCSummary_E9$E9_SKEW[a_idx] = BarcodeICov$rev_reads[e9_idx]	/ BarcodeICov$fwd_reads[e9_idx]		
		}
	}
}
BCSummary_E9$E9_fwd_reads = as.numeric(BCSummary_E9$E9_fwd_reads)
BCSummary_E9$E9_rev_reads = as.numeric(BCSummary_E9$E9_rev_reads)
BCSummary_E9$E9_SKEW = format(as.numeric(BCSummary_E9$E9_SKEW), digits=5)

if (file.exists(CNTruth)) {
    cn.truth = read.csv(CNTruth, sep="\t")
    if(length(SUB_IDX) > 0) {
        cn.truth = cn.truth[SUB_IDX,]
    }
}	

if(dim(BCSummary)[1] != dim(BCMatrix)[2]-2) {
    stop(paste("Number of barcodes in ",BC_FileName2," and ",BC_FileName1," are not the same.  Please rerun Coverage analysis plugin\n",sep=""))
} else {
        RunDir = strsplit(CovPATH,"plugin_out")[[1]][1]
        non_informativeBarcodes = NULL
        for(b_idx in 1:dim(BCSummary)[1]) {
            bamfilepath = list.files(path = RunDir, pattern = paste(BCSummary[b_idx,1],".*.bam$",sep=""), full.names=T)
            if(length(bamfilepath) == 0) {
                non_informativeBarcodes = c(non_informativeBarcodes, b_idx)
            } else {
                if(file.info(bamfilepath)$size < BCFILE_MIN_SIZE) {
                    non_informativeBarcodes = c(non_informativeBarcodes, b_idx)
                }
            }
        }
        non_informativeBarcodes = which(BCSummary$Mean.Depth < 10)
	if( length(non_informativeBarcodes)>0 ){
	    if( length(non_informativeBarcodes) == dim(BCSummary)[1] ) {
		   stop(paste("All bamfiles has file size < ",BCFILE_MIN_SIZE," bytes.  This run has very low coverage and CNV analysis will not be performed.",sep=""))
	    }
   	    BCSummary = BCSummary[-non_informativeBarcodes,]
	    BCMatrix = BCMatrix[,-(non_informativeBarcodes+2)]		
	    BCSummary_E9 = BCSummary_E9[-non_informativeBarcodes,]
	    if (file.exists(CNTruth)) { cn.truth = cn.truth[-non_informativeBarcodes,] }
	}
	if(dim(BCSummary)[1] >= 16) {
		exon.CN3.add = 1
	} else {
		exon.CN3.add = 0.7	
	}

        Run.DepthCV = sd(BCSummary$Mean.Depth) / mean(BCSummary$Mean.Depth)
	print(paste("Run.DepthCV =",format(Run.DepthCV, digits=4),sep=""))

	if(Run.DepthCV>THRES.DepthCV) {
		ErrorLog = paste(ErrorLog, "## Warning: Coefficient of variation (CV) of Read Depth is ",Run.DepthCV,">", THRES.AmpliconCV, ".  Reads are unbalanced and CNV may be incorrect.\n",sep="")		
	}

}

Reads = BCMatrix[,-c(1,2)]
#Amplicon.Mean = rowMeans(Reads)
#Amplicon.SD = apply(Reads, 1, sd)
#sample.CV = apply(Reads, 2, sd)/colMeans(Reads)
Ratio = Reads / t(replicate(nrow(Reads), colSums(Reads)))
amplicon.CV = apply(Ratio, 1, sd)/rowMeans(Ratio)
BCMatrix$amplicon.CV = amplicon.CV

# to avoid numerical error for division and log2, fill in d with minimal 1.1
# select norm_amp as "non-2D6 autosomal amplicons for normalization"
bedfilepath = list.files(path = paste(CovPATH, .Platform$file.sep, "local_beds", sep="") , pattern = "*.bed", full.names=T)
if (!file.exists(bedfilepath)){    
    bedfilepath = gc_bedfile
} 
if (!file.exists(bedfilepath)){
    stop("Error: problem allocating target bed file")
}

GCbed = read.table(bedfilepath, sep="\t", skip=1)	
colnames(GCbed) = c("chr","start","end","Amplicon","tmp","gc")
# hg19  chr22:42522445-42528527
# hg20  chr22:42126347-42132697

CYP2D6start=42126347
CYP2D6end=42528527

cyp2d6_idx = intersect( which(GCbed$chr == "chr22"), intersect(which(GCbed$start > CYP2D6start), which(GCbed$end < CYP2D6end)) )
ex_idx = c(which(GCbed$chr == "chrX"), which(GCbed$chr == "chrY"), cyp2d6_idx)
ex_amp = GCbed[ex_idx,]
# print(ex_amp)
norm_amp <- setdiff(BCMatrix[,2], ex_amp$Amplicon)
if(length(norm_amp) < THRES.normampMIN) {
    stop(paste("The number of amplicons for coverage normalization is less than ",THRES.normampMIN,".  Please use correct target bed region files for coverage analysis.\n",sep=""))
} else {
    norm_idx0 <- match(norm_amp, BCMatrix[,2])
    print(paste("number of non-CYP2D6 autosomal amplicons:",length(norm_idx0),sep=""))

    CV_idx <- which(BCMatrix$amplicon.CV < THRES.AmpliconCV)
    norm_idx = intersect(norm_idx0, CV_idx)
	# prompt warning if number of normalized amplicons < THRES.AmpliconCV is less than THRES.NormAmplicons
	if(length(norm_idx) < THRES.NormAmplicons) {
		ErrorLog = paste(ErrorLog, "## Warning: Number of amplicons with CV < ", THRES.AmpliconCV, " is ",length(norm_idx),".  CNV may not be stable.\n",sep="")	
    	CV_idx = order(BCMatrix$amplicon.CV)[1:THRES.NormAmplicons]  # top 20 amplicons with lowest CV
		norm_idx = intersect(norm_idx0, CV_idx)
	} else {
		print(paste("number of normalized amplicons (CV<",THRES.AmpliconCV,"):",length(norm_idx),sep=""))
	}	
	NAmpNorm = length(norm_amp)
	NAmp0.2 = length(intersect(norm_idx0, which(BCMatrix$amplicon.CV < 0.2)))
	NAmp0.5 = length(intersect(norm_idx0, intersect(which(BCMatrix$amplicon.CV >= 0.2),which(BCMatrix$amplicon.CV < 0.5))))
	NAmp1.0 = length(intersect(norm_idx0, intersect(which(BCMatrix$amplicon.CV >= 0.5),which(BCMatrix$amplicon.CV < 1))))	
	NAmpgt1 = length(intersect(norm_idx0, which(BCMatrix$amplicon.CV >= 1)))			
}

Metric.SampleN = dim(BCSummary)[1]
Metric.MeanDepth = BCSummary$Mean.Depth
Metric.Uniformity = as.numeric(sub("%","",(BCSummary$Uniformity)))

d <- BCMatrix[,-c(1,2)]
# this command get rid of BCMatrix$amplicon.CV column and after 
d <- d[,1:Metric.SampleN]
d[d < 2] <- 1.1

temp <- apply(d[norm_idx,], 2, median)
temp <- t(replicate(nrow(d), temp))

# normalize the read counts by median of all; will check if a subset is more robust...
d <- d / temp  
nn <- log2(d)
exon9_idx = which(BCMatrix[,2] == exon9_amp)

# identify CN=0 samples, before assigning to NOCALL
E9_CN0_idx = which(nn[exon9_idx,] < THRES.CN0)
if(any(E9_CN0_idx)) {
    E9_CN0_Barcode = colnames(nn)[E9_CN0_idx]
} else {
    E9_CN0_Barcode = NULL
}


# use 9 amplicons for gene-level CNV
geneamp_idx = match(gene_amps, BCMatrix[,2])
GN_CN0_idx = c()
for(g_idx in 1:length(geneamp_idx)) {
	GN_CN0_idx = union(GN_CN0_idx, which(nn[geneamp_idx[g_idx],] < -5))
}
if(any(GN_CN0_idx)) {
    GN_CN0_Barcode = colnames(nn)[GN_CN0_idx]
} else {
    GN_CN0_Barcode = NULL
}


NOCALL_idx1 = which(as.numeric(BCSummary_E9$E9_SKEW) > THRES.E9SKEW)
NOCALL_idx2 = which(Metric.MeanDepth < THRES.COV)
NOCALL_idx3 = which(Metric.Uniformity < THRES.Uniformity)

# NOCALL_IDX = union(NOCALL_idx1, union(NOCALL_idx2, NOCALL_idx3))
NOCALL_IDX = union(NOCALL_idx2, NOCALL_idx3)

# what if all samples pass QC?
E9_NOCALL_IDX = setdiff(NOCALL_IDX, E9_CN0_idx)
if(length(E9_NOCALL_IDX) == 0) {
    E9_NOCALL_IDX = E9_NOCALL_Barcode = NULL
} else {
    E9_NOCALL_Barcode = as.character(BCSummary[E9_NOCALL_IDX,1])
}

BCSummary_E9$RunQC = "PASS"
BCSummary_E9$RunQC[NOCALL_IDX] = "NOCALL"
BCSummary_E9$E9FR = "."
BCSummary_E9$FR = "."


for(f_idx in 1:Metric.SampleN) {
    if(f_idx %in% NOCALL_idx1) {   # only update E9 FR
		if(BCSummary_E9$E9FR[f_idx] ==".") {			BCSummary_E9$E9FR[f_idx] = paste("Exon9SKEW=",BCSummary_E9$E9_SKEW[f_idx],">",THRES.E9SKEW,sep="")
		} else { 		    BCSummary_E9$E9FR[f_idx] = paste(BCSummary_E9$E9FR[f_idx], ",Exon9SKEW=",BCSummary_E9$E9_SKEW[f_idx],">",THRES.E9SKEW,sep="")		}
	}			
    if(f_idx %in% NOCALL_idx2) {
		if(BCSummary_E9$FR[f_idx] ==".") {				BCSummary_E9$FR[f_idx] = paste("Depth=",Metric.MeanDepth[f_idx],"<",THRES.COV,sep="")
		} else {		    BCSummary_E9$FR[f_idx] = paste(BCSummary_E9$FR[f_idx], ",Depth=",Metric.MeanDepth[f_idx],"<",THRES.COV,sep="")			}			
		if(BCSummary_E9$E9FR[f_idx] ==".") {			BCSummary_E9$E9FR[f_idx] = paste("Depth=",Metric.MeanDepth[f_idx],"<",THRES.COV,sep="")
		} else { 		    BCSummary_E9$E9FR[f_idx] = paste(BCSummary_E9$E9FR[f_idx], ",Depth=",Metric.MeanDepth[f_idx],"<",THRES.COV,sep="")		}		
	}
    if(f_idx %in% NOCALL_idx3) {   
		if(BCSummary_E9$FR[f_idx] ==".") {				BCSummary_E9$FR[f_idx] = paste("Uniformity=",Metric.Uniformity[f_idx],"<",THRES.Uniformity,sep="")
		} else {		    BCSummary_E9$FR[f_idx] = paste(BCSummary_E9$FR[f_idx], ",Uniformity=",Metric.Uniformity[f_idx],"<",THRES.Uniformity,sep="")			}			
		if(BCSummary_E9$E9FR[f_idx] ==".") {			BCSummary_E9$E9FR[f_idx] = paste("Uniformity=",Metric.Uniformity[f_idx],"<",THRES.Uniformity,sep="")
		} else { 		    BCSummary_E9$E9FR[f_idx] = paste(BCSummary_E9$E9FR[f_idx], ",Uniformity=",Metric.Uniformity[f_idx],"<",THRES.Uniformity,sep="")			}				
	}		
}

CNV_EMResult_withExon = BCSummary
E9_EM_IDX = setdiff(1:Metric.SampleN, c(NOCALL_IDX,E9_CN0_idx))
if(length(E9_EM_IDX) < THRES.EMMin) {
    ErrorLog = paste(ErrorLog, "##", length(E9_EM_IDX)," out of ",Metric.SampleN," samples potentially with CN=1,2 or 3 have passed QC.\n",sep="")
    ErrorLog = paste(ErrorLog, "##CNV algorithm didn't execute with less than ",THRES.EMMin," samples.\n",sep="")
    E9RunCNV = FALSE
	CNV_EMResult_withExon$CN.EXON9 = "."
	CNV_EMResult_withExon$Exon9Confidence = 0
	CNV_EMResult_withExon$Exon9NOCALL = "NOCALL"
	CNV_EMResult_withExon$Exon9CNV = "NOCALL"		
	
	for(f_idx in 1:Metric.SampleN) {
		if(f_idx %in% E9_EM_IDX) {
			if(BCSummary_E9$E9FR[f_idx] ==".") {			
				BCSummary_E9$E9FR[f_idx] = paste("InsufficientBarcodes=",length(E9_EM_IDX),"<",THRES.EMMin+1,sep="")
			} else { 		    
				BCSummary_E9$E9FR[f_idx] = paste(BCSummary_E9$E9FR[f_idx], ",InsufficientBarcodes=",length(E9_EM_IDX),"<",THRES.EMMin+1,sep="")		
			}		
		}
 	}
} else {
    ErrorLog = ""
    E9RunCNV = TRUE    
    # exclude the samples with either NOCALL_IDX or CN0/CN4
    E9_nn_sub = nn[,E9_EM_IDX]
    ExonOut = exon9CNV(E9_nn_sub, exon9_idx)

    CN.EXON9 = ExonOut$call
    ExonOut$Phred = as.numeric(format(ExonOut$Phred, digits=4))
    Exon9Confidence = ExonOut$Phred
    Exon9NOCALL = ifelse(ExonOut$Phred > THRES.ExonPHRED, "PASS", "NOCALL")
    Exon9CNV = ifelse(ExonOut$Phred > THRES.ExonPHRED, CN.EXON9, "NOCALL")

	CNV_EMResult_withExon = cbind(BCSummary[E9_EM_IDX,], CN.EXON9, Exon9Confidence, Exon9NOCALL, Exon9CNV)	
	
	idx = which(Exon9NOCALL == "NOCALL")	
	if(length(idx)>0) {
	    for(i in 1:length(idx)) {
			if(BCSummary_E9$E9FR[E9_EM_IDX[idx[i]]] ==".") {				BCSummary_E9$E9FR[E9_EM_IDX[idx[i]]] = paste("EXON9CONF=",ExonOut$Phred[idx[i]],"<",THRES.ExonPHRED,sep="")
		    } else {		    BCSummary_E9$E9FR[E9_EM_IDX[idx[i]]] = paste(BCSummary_E9$E9FR[E9_EM_IDX[idx[i]]], "EXON9CONF=",ExonOut$Phred[idx[i]],"<",THRES.ExonPHRED,sep="")			}
		}		
	}
	Exon9Col <- rep("grey",length(E9_EM_IDX))	
}

CNV_EMResult_withGene = BCSummary
GN_EM_IDX = setdiff(1:Metric.SampleN, c(NOCALL_IDX,GN_CN0_idx))
if(length(GN_EM_IDX) < THRES.EMMin) {
    ErrorLog = paste(ErrorLog, "##", length(GN_EM_IDX)," out of ",Metric.SampleN," samples potentially with CN=1,2 or 3 have passed QC.\n",sep="")
    ErrorLog = paste(ErrorLog, "##CNV algorithm didn't execute with less than ",THRES.EMMin," samples.\n",sep="")
    GNRunCNV = FALSE
	CNV_EMResult_withGene$CN.GENE = "."
	CNV_EMResult_withGene$GeneConfidence = 0
	CNV_EMResult_withGene$GeneNOCALL = "NOCALL"
	CNV_EMResult_withGene$GeneCNV = "NOCALL"			
	
	for(f_idx in 1:Metric.SampleN) {
		if(f_idx %in% GN_EM_IDX) {
			if(BCSummary_E9$FR[f_idx] ==".") {			
				BCSummary_E9$FR[f_idx] = paste("InsufficientBarcodes=",length(GN_EM_IDX),"<",THRES.EMMin+1,sep="")
			} else { 		    
				BCSummary_E9$FR[f_idx] = paste(BCSummary_E9$FR[f_idx], ",InsufficientBarcodes=",length(GN_EM_IDX),"<",THRES.EMMin+1,sep="")		
			}		
		}
 	}
} else {
    ErrorLog = ""
    GNRunCNV = TRUE    
    # exclude the samples with either NOCALL_IDX or CN0/CN4
    GN_nn_sub = nn[,GN_EM_IDX]
    GeneOut = geneCNV(GN_nn_sub, geneamp_idx)	
	
    CN.GENE = GeneOut$call
    GeneOut$Phred = as.numeric(format(GeneOut$Phred, digits=4))
    GeneConfidence = GeneOut$Phred
    GeneNOCALL = ifelse(GeneOut$Phred > THRES.GenePHRED, "PASS", "NOCALL")
    GeneCNV = ifelse(GeneOut$Phred > THRES.GenePHRED, CN.GENE, "NOCALL")
	
	CNV_EMResult_withGene = cbind(BCSummary[GN_EM_IDX,], CN.GENE, GeneConfidence, GeneNOCALL, GeneCNV)
	
	idx = which(GeneNOCALL == "NOCALL")
	if(length(idx)>0) {
	    for(i in 1:length(idx)) {
			if(BCSummary_E9$FR[GN_EM_IDX[idx[i]]] ==".") {				BCSummary_E9$FR[GN_EM_IDX[idx[i]]] = paste("GENECONF=",GeneOut$Phred[idx[i]],"<",THRES.GenePHRED,sep="")
		    } else {		    BCSummary_E9$FR[GN_EM_IDX[idx[i]]] = paste(BCSummary_E9$FR[GN_EM_IDX[idx[i]]], "GENECONF=",GeneOut$Phred[idx[i]],"<",THRES.GenePHRED,sep="")		}
		}		
	}
	GeneCol <- rep("grey",length(GN_EM_IDX))	
}

CNV_EMResult = merge(BCSummary, CNV_EMResult_withExon, all.x=TRUE)
CNV_EMResult = merge(CNV_EMResult, CNV_EMResult_withGene, all.x=TRUE)
CNV_EMResult = CNV_EMResult[match(BCSummary[,1], CNV_EMResult[,1]),]

for(c_idx in 1:dim(CNV_EMResult)[2]) {
	CNV_EMResult[,c_idx] = as.character(CNV_EMResult[,c_idx])
}

# if skewed, in NOCALL_idx1, still can call CN0 & CN4
if(any(E9_CN0_idx)) {
    CNV_EMResult$CN.EXON9[E9_CN0_idx] = 0
    CNV_EMResult$Exon9Confidence[E9_CN0_idx] = 100
    CNV_EMResult$Exon9NOCALL[E9_CN0_idx] = "PASS"
    CNV_EMResult$Exon9CNV[E9_CN0_idx] = 0
}
	
if(any(GN_CN0_idx)) {
    CNV_EMResult$CN.GENE[GN_CN0_idx] = 0
    CNV_EMResult$GeneConfidence[GN_CN0_idx] = 100
    CNV_EMResult$GeneNOCALL[GN_CN0_idx] = "PASS"
    CNV_EMResult$GeneCNV[GN_CN0_idx] = 0
}

if(E9RunCNV & GNRunCNV) {
	#CN4_idx = as.numeric(which(colSums(rbind(ExonOut$exon.CN4.idx, GeneOut$gene.CN4.idx)) > 8))
	AmpExon = cbind(as.character(CNV_EMResult_withExon[,1]), ExonOut$exon.CN4.idx)	
	AmpsGene = cbind(as.character(CNV_EMResult_withGene[,1]), t(GeneOut$gene.CN4.idx))
	intresectBarcode = intersect(AmpExon[,1], AmpsGene[,1])
	AmpGeneExon = cbind(AmpExon[match(intresectBarcode, AmpExon[,1]),], AmpsGene[match(intresectBarcode, AmpsGene[,1]),-1])
	idx = which(rowSums(AmpGeneExon[,-1] == "TRUE") > 8)
	CN4_Barcodes = AmpGeneExon[idx,1]
	CN4_idx = match(CN4_Barcodes, CNV_EMResult[,1])	
	
	if(length(CN4_idx) > 0) {		
		CNV_EMResult$CN.EXON9[CN4_idx] = 4
		CNV_EMResult$Exon9Confidence[CN4_idx] = 50
		CNV_EMResult$Exon9NOCALL[CN4_idx] = "PASS"
		CNV_EMResult$Exon9CNV[CN4_idx] = 4

		CNV_EMResult$CN.GENE[CN4_idx] = 4
		CNV_EMResult$GeneConfidence[CN4_idx] = 50
		CNV_EMResult$GeneNOCALL[CN4_idx] = "PASS"
		CNV_EMResult$GeneCNV[CN4_idx] = 4
	}
} else {
	CN4_idx=c()
}		
			

#CNV_EMResult$FR = ifelse(is.na(CNV_EMResult$FR), BCSummary_E9$FR, CNV_EMResult$FR)
#CNV_EMResult$E9FR = ifelse(is.na(CNV_EMResult$E9FR), BCSummary_E9$E9FR, CNV_EMResult$E9FR)

CNV_EMResult$FR = BCSummary_E9$FR
CNV_EMResult$E9FR = BCSummary_E9$E9FR

#Report = merge(BCSummary, CNV_EMResult, all.x=TRUE)
tmp = CNV_EMResult[,which(colnames(CNV_EMResult) == "CN.EXON9"):dim(CNV_EMResult)[2]]
idx = match(c("FR","E9FR"),colnames(tmp))
tmp = tmp[,c(idx, setdiff(1:dim(tmp)[2], idx))]
Report = cbind(BCSummary,tmp)

for(col_idx in 1:dim(Report)[2]) {
    Report[,col_idx] = as.character(Report[,col_idx])
    Report[which(is.na(Report[,col_idx])),col_idx] = ""
}

for(f_idx in 1:Metric.SampleN) {	
    if(f_idx %in% NOCALL_IDX) {
		if(f_idx %in% E9_CN0_idx) {
			Report$E9FR[f_idx] = ".";  
		} else {
			Report$CN.EXON9[f_idx] = ".";   
			Report$Exon9Confidence[f_idx] = 0;   
			Report$Exon9NOCALL[f_idx] = "NOCALL";   
			Report$Exon9CNV[f_idx] = "NOCALL";   
		}
		if(f_idx %in% GN_CN0_idx) {
			Report$FR[f_idx] = ".";  		
		} else {
			Report$CN.GENE[f_idx] = ".";   
			Report$GeneConfidence[f_idx] = 0;   
			Report$GeneNOCALL[f_idx] = "NOCALL";   
			Report$GeneCNV[f_idx] = "NOCALL";   
		}		
		if(file.exists(CNTruth)) {
			Report$exon.CN[f_idx] = cn.truth_beforeEM$exon.CN[f_idx]
			Report$gene.CN[f_idx] = cn.truth_beforeEM$gene.CN[f_idx]				
		}
		
	}
    if(f_idx %in% NOCALL_idx1) {
		if(f_idx %in% E9_CN0_idx) {
			Report$E9FR[f_idx] = ".";  
		} else {
			Report$CN.EXON9[f_idx] = ".";   
			Report$Exon9Confidence[f_idx] = 0;   
			Report$Exon9NOCALL[f_idx] = "NOCALL";   
			Report$Exon9CNV[f_idx] = ".";   
		}
	}
}

Report$ExonQUAL  = ifelse(Report$Exon9NOCALL == "PASS", Report$Exon9Confidence, 0)
Report$GeneQUAL  = ifelse(Report$GeneNOCALL == "PASS", Report$GeneConfidence, 0)
	
CNV_vcf_HEADER = "##ALT=<ID=CNV,Description=\"Copy number variable region\">\n"
CNV_vcf_HEADER = paste(CNV_vcf_HEADER,"##INFO=<ID=FUNC,Number=.,Type=String,Description=\"Functional Annotations\">\n",sep="")
CNV_vcf_HEADER = paste(CNV_vcf_HEADER,"##INFO=<ID=CONFIDENCE,Number=1,Type=Float,Description=\"Log Likelihood Ratio of the observed ploidy to the expected ploidy\">\n",sep="")
CNV_vcf_HEADER = paste(CNV_vcf_HEADER,"##FORMAT=<ID=CN,Number=1,Type=Float,Description=\"Copy number genotype for imprecise events\">\n",sep="")
CNV_vcf_HEADER = paste(CNV_vcf_HEADER,"##FORMAT=<ID=CYP2D6Exon9Conversion,Number=1,Type=Integer,Description=\"CYP2D6 genotype with exon 9 conversion event\">\n",sep="")
CNV_vcf_HEADER = paste(CNV_vcf_HEADER,"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample",sep="")

tab = "\t"
Exon9Prefix = "chr22\t42522509\tCYP2D6Exon9\tA\t<CNV>\t"
Exon9Mid = "\tPRECISE=FALSE;SVTYPE=CNV;END=42522670;LEN=161;CONFIDENCE="
Exon9Postfix = ";FUNC=[{'gene':'CYP2D6'},{'exon':'9'}]\tGT:GQ:CN\t./.:0:"

GenePrefix = "chr22\t42523660\tCYP2D6Gene\tA\t<CNV>\t"
GeneMid = "\tPRECISE=FALSE;SVTYPE=CNV;END=42526808;LEN=3148;CONFIDENCE="
GenePostfix = ";FUNC=[{'gene':'CYP2D6'}]\tGT:GQ:CN\t./.:0:"



if(GNRunCNV) {
    cat(Metric.SampleN," vcf files created under",OutDIR,"\n")

	for(f_idx in 1:Metric.SampleN) {
		write.table(CNV_vcf_HEADER, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=F)	
	    if ( (Report$Exon9CNV[f_idx] < Report$GeneCNV[f_idx] ) & (Report$Exon9NOCALL[f_idx] == "PASS") & (Report$GeneNOCALL[f_idx] == "PASS") ) {		
			Exon_vcf = paste(Exon9Prefix, Report$ExonQUAL[f_idx], tab, Report$Exon9NOCALL[f_idx],sep="")
			Exon_vcf = paste(Exon_vcf, Exon9Mid, format(as.numeric(Report$Exon9Confidence[f_idx]), digits=4), ";FR=", Report$E9FR[f_idx],";CYP2D6Exon9Conversion=1", Exon9Postfix, Report$CN.EXON9[f_idx], sep="")
			write.table(Exon_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)
			
			Gene_vcf = paste(GenePrefix, Report$GeneQUAL[f_idx], tab, Report$GeneNOCALL[f_idx],sep="")
			Gene_vcf = paste(Gene_vcf, GeneMid, format(as.numeric(Report$GeneConfidence[f_idx]), digits=4), ";FR=", Report$FR[f_idx],";CYP2D6Exon9Conversion=1", GenePostfix, Report$CN.GENE[f_idx], sep="")
			write.table(Gene_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)
		} else {
			Exon_vcf = paste(Exon9Prefix, Report$ExonQUAL[f_idx], tab, Report$Exon9NOCALL[f_idx],sep="")
			Exon_vcf = paste(Exon_vcf, Exon9Mid, format(as.numeric(Report$Exon9Confidence[f_idx]), digits=4), ";FR=", Report$E9FR[f_idx], ";CYP2D6Exon9Conversion=0", Exon9Postfix, Report$CN.EXON9[f_idx], sep="")
			write.table(Exon_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)
			
			Gene_vcf = paste(GenePrefix, Report$GeneQUAL[f_idx], tab, Report$GeneNOCALL[f_idx],sep="")
			Gene_vcf = paste(Gene_vcf, GeneMid, format(as.numeric(Report$GeneConfidence[f_idx]), digits=4), ";FR=", Report$FR[f_idx], ";CYP2D6Exon9Conversion=0", GenePostfix, Report$CN.GENE[f_idx], sep="")
			write.table(Gene_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)		
		}
    }
	
    # Output Summary Report
	exclude_col = c("Exon9CNV","GeneCNV")
    write.table(Report[, -match(exclude_col, colnames(Report))], file = LogFile, quote=F, sep="\t", row.names=F, col.names=T, append=F)
} else {
    cat(Metric.SampleN," vcf files created under",OutDIR,"\n")

	for(f_idx in 1:Metric.SampleN) {
		write.table(CNV_vcf_HEADER, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=F)	

		Exon_vcf = paste(Exon9Prefix, Report$ExonQUAL[f_idx], tab, Report$Exon9NOCALL[f_idx],sep="")
		Exon_vcf = paste(Exon_vcf, Exon9Mid, format(as.numeric(Report$Exon9Confidence[f_idx]), digits=4), ";FR=", Report$E9FR[f_idx],";CYP2D6Exon9Conversion=0", Exon9Postfix, Report$CN.EXON9[f_idx], sep="")
		write.table(Exon_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)
		
		Gene_vcf = paste(GenePrefix, Report$GeneQUAL[f_idx], tab, Report$GeneNOCALL[f_idx],sep="")
		Gene_vcf = paste(Gene_vcf, GeneMid, format(as.numeric(Report$GeneConfidence[f_idx]), digits=4), ";FR=", Report$FR[f_idx],";CYP2D6Exon9Conversion=0", GenePostfix, Report$CN.GENE[f_idx], sep="")
		write.table(Gene_vcf, file = sprintf("%s/%s_CNV.vcf", OutDIR, BCSummary$Barcode.ID[f_idx]), quote=F, sep="\t", row.names=F, col.names=F, append=T)
    }
	
	exclude_col = c("Exon9CNV","GeneCNV")
    write.table(Report[, -match(exclude_col, colnames(Report))], file = LogFile, quote=F, sep="\t", row.names=F, col.names=T, append=F)
    write.table(ErrorLog, file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)   
    cat(ErrorLog)
}

write.table(paste("##Total Number of valid Samples = ", Metric.SampleN ,sep=""), file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)
write.table(paste("##Number of Samples passed QC for CNV Calling = ", Metric.SampleN-length(NOCALL_IDX),sep=""), file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)
write.table(paste("##Number of Samples did not pass QC = ", length(NOCALL_IDX) ,sep=""), file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)
write.table(paste("##Number of Samples did not pass QC metric 'Average coverage ",THRES.COV,"' = ", length(NOCALL_idx2),sep=""), file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)
write.table(paste("##Number of Samples did not pass QC metric 'Uniformity Rate ",THRES.Uniformity,"%' = ",length(NOCALL_idx3),sep=""), file = LogFile, quote=F, sep="\t", row.names=F, col.names=F, append=T)

cat("Log file", LogFile, "created ","\n\n")
