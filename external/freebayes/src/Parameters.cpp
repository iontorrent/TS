#include "Parameters.h"
#include "convert.h"

using namespace std;

void Parameters::simpleUsage(char ** argv) {
   cout
         << "usage: " << argv[0] << " -f [REFERENCE] [OPTIONS] [BAM FILES] >[OUTPUT]" << endl
         << endl
         << "Bayesian haplotype-based polymorphism discovery." << endl
         << endl
         << "citation: Erik Garrison, Gabor Marth" << endl
         << "          \"Haplotype-based variant detection from short-read sequencing\"" << endl
         << "          arXiv:1207.3907 (http://arxiv.org/abs/1207.3907)" << endl
         << endl
         << "overview:" << endl
         << endl
         << "    To call variants from aligned short-read sequencing data, supply BAM files and" << endl
         << "    a reference.  FreeBayes will provide VCF output on standard out describing SNPs," << endl
         << "    indels, and complex variants in samples in the input alignments." << endl
         << endl
         << "    By default, FreeBayes will consider variants supported by at least 2" << endl
         << "    observations in a single sample (-C) and also by at least 20% of the reads from" << endl
         << "    a single sample (-F).  These settings are suitable to low to high depth" << endl
         << "    sequencing in haploid and diploid samples, but users working with polyploid or" << endl
         << "    pooled samples may wish to adjust them depending on the characteristics of" << endl
         << "    their sequencing data." << endl
         << endl
         << "    FreeBayes is capable of calling variant haplotypes shorter than a read length" << endl
         << "    where multiple polymorphisms segregate on the same read.  The maximum distance" << endl
         << "    between polymorphisms phased in this way is determined by the" << endl
         << "    --max-complex-gap, which defaults to 3bp.  In practice, this can comfortably be" << endl
         << "    set to half the read length." << endl
         << endl
         << "    Ploidy may be set to any level (-p), but by default all samples are assumed to" << endl
         << "    be diploid.  FreeBayes can model per-sample and per-region variation in" << endl
         << "    copy-number (-A) using a copy-number variation map." << endl
         << endl
         << "parameters:" << endl
         << endl
         << "   -h --help       Complete description of options." << endl
         << endl
         << "author:   Erik Garrison <erik.garrison@bc.edu>, Marth Lab, Boston College, 2010-2012" << endl
         << "date:     " << FREEBAYES_COMPILE_DATE << endl
         << "version:  " << FREEBAYES_VERSION << endl;

}

void Parameters::usage(char** argv) {
    cout 
         << "usage: " << argv[0] << " [OPTION] ... [BAM FILE] ... " << endl
         << endl
         << "Bayesian haplotype-based polymorphism discovery." << endl
         << endl
         << "citation: Erik Garrison, Gabor Marth" << endl
         << "          \"Haplotype-based variant detection from short-read sequencing\"" << endl
         << "          arXiv:1207.3907 (http://arxiv.org/abs/1207.3907)" << endl
         << endl
         << "overview:" << endl
         << endl
         << "    To call variants from aligned short-read sequencing data, supply BAM files and" << endl
         << "    a reference.  FreeBayes will provide VCF output on standard out describing SNPs," << endl
         << "    indels, and complex variants in samples in the input alignments." << endl
         << endl
         << "    By default, FreeBayes will consider variants supported by at least 2" << endl
         << "    observations in a single sample (-C) and also by at least 20% of the reads from" << endl
         << "    a single sample (-F).  These settings are suitable to low to high depth" << endl
         << "    sequencing in haploid and diploid samples, but users working with polyploid or" << endl
         << "    pooled samples may wish to adjust them depending on the characteristics of" << endl
         << "    their sequencing data." << endl
         << endl
         << "    FreeBayes is capable of calling variant haplotypes shorter than a read length" << endl
         << "    where multiple polymorphisms segregate on the same read.  The maximum distance" << endl
         << "    between polymorphisms phased in this way is determined by the" << endl
         << "    --max-complex-gap, which defaults to 3bp.  In practice, this can comfortably be" << endl
         << "    set to half the read length." << endl
         << endl
         << "    Ploidy may be set to any level (-p), but by default all samples are assumed to" << endl
         << "    be diploid.  FreeBayes can model per-sample and per-region variation in" << endl
         << "    copy-number (-A) using a copy-number variation map." << endl
         << endl
         << "parameters:" << endl
         << endl
         << "   -h --help       Prints this help dialog." << endl
         << endl
         << "input and output:" << endl
         << endl
         << "   -b --bam FILE   Add FILE to the set of BAM files to be analyzed." << endl
         << "   -c --stdin      Read BAM input on stdin." << endl
         << "   -v --vcf FILE   Output VCF-format results to FILE." << endl
         << "   -f --fasta-reference FILE" << endl
         << "                   Use FILE as the reference sequence for analysis." << endl
         << "                   An index file (FILE.fai) will be created if none exists." << endl
         << "                   If neither --targets nor --region are specified, FreeBayes" << endl
         << "                   will analyze every position in this reference." << endl
         << "   -t --targets FILE" << endl
         << "                   Limit analysis to targets listed in the BED-format FILE." << endl
         << "   -r --region <chrom>:<start_position>..<end_position>" << endl
         << "                   Limit analysis to the specified region, 0-base coordinates," << endl
         << "                   end_position not included (same as BED format)." << endl
         << "   -s --samples FILE" << endl
         << "                   Limit analysis to samples listed (one per line) in the FILE." << endl
         << "                   By default FreeBayes will analyze all samples in its input" << endl
         << "                   BAM files." << endl
         << "   --populations FILE" << endl
         << "                   Each line of FILE should list a sample and a population which" << endl
         << "                   it is part of.  The population-based bayesian inference model" << endl
         << "                   will then be partitioned on the basis of the populations." << endl
         << "   -A --cnv-map FILE" << endl
         << "                   Read a copy number map from the BED file FILE, which has" << endl
         << "                   the format:" << endl
         << "                      reference sequence, start, end, sample name, copy number" << endl
         << "                   ... for each region in each sample which does not have the" << endl
         << "                   default copy number as set by --ploidy." << endl
         << "   -L --trace FILE  Output an algorithmic trace to FILE." << endl
         << "   --failed-alleles FILE" << endl
         << "                   Write a BED file of the analyzed positions which do not" << endl
         << "                   pass --pvar to FILE." << endl
         << "   -@ --variant-input VCF" << endl
         << "                   Use variants reported in VCF file as input to the algorithm." << endl
	 << "                   Variants in this file will be treated as putative variants" << endl
	 << "                   even if there is not enough support in the data to pass" << endl
         << "                   input filters." << endl
         << "   -l --only-use-input-alleles" << endl
         << "                   Only provide variant calls and genotype likelihoods for sites" << endl
         << "                   and alleles which are provided in the VCF input, and provide" << endl
         << "                   output in the VCF for all input alleles, not just those which" << endl
         << "                   have support in the data." << endl
	 << "   --haplotype-basis-alleles VCF" << endl
	 << "                   When specified, only variant alleles provided in this input" << endl
         << "                   VCF will be used for the construction of complex or haplotype" << endl
	 << "                   alleles." << endl
	 << "   --report-all-haplotype-alleles" << endl
	 << "                   At sites where genotypes are made over haplotype alleles," << endl
	 << "                   provide information about all alleles in output, not only" << endl
         << "                   those which are called." << endl
         << endl
         << "reporting:" << endl
         << endl
         << "   -P --pvar N     Report sites if the probability that there is a polymorphism" << endl
         << "                   at the site is greater than N.  default: 0.0001" << endl
         << "   -_ --show-reference-repeats" << endl
         << "                   Calculate and show information about reference repeats in" << endl
         << "                   the VCF output." << endl
         << endl
         << "population model:" << endl
         << endl
         << "   -T --theta N    The expected mutation rate or pairwise nucleotide diversity" << endl
         << "                   among the population under analysis.  This serves as the" << endl
         << "                   single parameter to the Ewens Sampling Formula prior model" << endl
         << "                   default: 0.001" << endl
         << "   -p --ploidy N   Sets the default ploidy for the analysis to N.  default: 2" << endl
         << "   -J --pooled     Assume that samples result from pooled sequencing." << endl
         << "                   When using this flag, set --ploidy to the number of" << endl
         << "                   alleles in each sample." << endl
         << endl
         << "reference allele:" << endl
         << endl
         << "   -Z --use-reference-allele" << endl
         << "                   This flag includes the reference allele in the analysis as" << endl
         << "                   if it is another sample from the same population." << endl
         << "   -H --diploid-reference" << endl
         << "                   If using the reference sequence as a sample (-Z)," << endl
         << "                   treat it as diploid.  default: false (reference is haploid)" << endl
         << "   --reference-quality MQ,BQ" << endl
         << "                   Assign mapping quality of MQ to the reference allele at each" << endl
         << "                   site and base quality of BQ.  default: 100,60" << endl
         << endl
         << "allele scope:" << endl
         << endl
         << "   -I --no-snps    Ignore SNP alleles." << endl
         << "   -i --no-indels  Ignore insertion and deletion alleles." << endl
         << "   -X --no-mnps    Ignore multi-nuceotide polymorphisms, MNPs." << endl
         << "   -u --no-complex Ignore complex events (composites of other classes)." << endl
         << "   -n --use-best-n-alleles N" << endl
         << "                   Evaluate only the best N SNP alleles, ranked by sum of" << endl
         << "                   supporting quality scores.  (Set to 0 to use all; default: all)" << endl
         << "   -E --max-complex-gap N" << endl
         << "                   Allow complex alleles with contiguous embedded matches of up" << endl
         << "                   to this length." << endl
         << endl
         << "indel realignment:" << endl
         << endl
         << "   -O --left-align-indels" << endl
         << "                   Left-realign and merge gaps embedded in reads. default: false" << endl
         << endl
         << "input filters:" << endl
         << endl
         << "   -4 --use-duplicate-reads" << endl
         << "                   Include duplicate-marked alignments in the analysis." << endl
         << "                   default: exclude duplicates" << endl
         << "   -m --min-mapping-quality Q" << endl
         << "                   Exclude alignments from analysis if they have a mapping" << endl
         << "                   quality less than Q.  default: 30" << endl
         << "   -q --min-base-quality Q" << endl
         << "                   Exclude alleles from analysis if their supporting base" << endl
         << "                   quality is less than Q.  default: 20" << endl
         << "   -R --min-supporting-quality MQ,BQ" << endl
         << "                   In order to consider an alternate allele, at least one supporting" << endl
         << "                   alignment must have mapping quality MQ, and one supporting " << endl
         << "                   allele must have base quality BQ. default: 0,0, unset" << endl
         << "   -Q --mismatch-base-quality-threshold Q" << endl
         << "                   Count mismatches toward --read-mismatch-limit if the base" << endl
         << "                   quality of the mismatch is >= Q.  default: 10" << endl
         << "   -U --read-mismatch-limit N" << endl
         << "                   Exclude reads with more than N mismatches where each mismatch" << endl
         << "                   has base quality >= mismatch-base-quality-threshold." << endl
         << "                   default: ~unbounded" << endl
         << "   -z --read-max-mismatch-fraction N" << endl
         << "                   Exclude reads with more than N [0,1] fraction of mismatches where" << endl
         << "                   each mismatch has base quality >= mismatch-base-quality-threshold" << endl
         << "                   default: 1.0" << endl
         << "   -$ --read-snp-limit N" << endl
         << "                   Exclude reads with more than N base mismatches, ignoring gaps" << endl
         << "                   with quality >= mismatch-base-quality-threshold." << endl
         << "                   default: ~unbounded" << endl
         << "   -e --read-indel-limit N" << endl
         << "                   Exclude reads with more than N separate gaps." << endl
         << "                   default: ~unbounded" << endl
         << "   -0 --standard-filters  Use stringent input base and mapping quality filters" << endl
         << "                   Equivalent to -m 30 -q 20 -R 0 -S 0" << endl
         << "   -x --indel-exclusion-window" << endl
         << "                   Ignore portions of alignments this many bases from a" << endl
         << "                   putative insertion or deletion allele.  default: 0" << endl
         << "   -F --min-alternate-fraction N" << endl
         << "                   Require at least this fraction of observations supporting" << endl
         << "                   an alternate allele within a single individual in the" << endl
         << "                   in order to evaluate the position.  default: 0.2" << endl
         << "   -C --min-alternate-count N" << endl
         << "                   Require at least this count of observations supporting" << endl
         << "                   an alternate allele within a single individual in order" << endl
         << "                   to evaluate the position.  default: 2" << endl
         << "   -3 --min-alternate-qsum N" << endl
         << "                   Require at least this sum of quality of observations supporting" << endl
         << "                   an alternate allele within a single individual in order" << endl
         << "                   to evaluate the position.  default: 0" << endl
         << "   -G --min-alternate-total N" << endl
         << "                   Require at least this count of observations supporting" << endl
         << "                   an alternate allele within the total population in order" << endl
         << "                   to use the allele in analysis.  default: 1" << endl
         << "   -! --min-coverage N" << endl
         << "                   Require at least this coverage to process a site.  default: 0" << endl
         << endl
         << "bayesian priors:" << endl
         << endl
         << "   -Y --no-ewens-priors" << endl
         << "                   Turns off the Ewens' Sampling Formula component of the priors." << endl
         << "   -k --no-population-priors" << endl
         << "                   Equivalent to --pooled --no-ewens-priors" << endl
         << "   -w --hwe-priors Use the probability of the combination arising under HWE given" << endl
         << "                   the allele frequency as estimated by observation frequency." << endl
         << endl
         << "observation prior expectations:" << endl
         << endl
         << "   -V --binomial-obs-priors" << endl
         << "                   Incorporate expectations about osbervations into the priors," << endl
         << "                   Uses read placement probability, strand balance probability," << endl
         << "                   and read position (5'-3') probability." << endl
         << "   -a --allele-balance-priors" << endl
         << "                   Use aggregate probability of observation balance between alleles" << endl
         << "                   as a component of the priors.  Best for observations with minimal" << endl
         << "                   inherent reference bias." << endl
         << endl
	 << "genotype likelihoods:" << endl
	 << "   --observation-bias FILE" << endl
	 << "                   Read length-dependent allele observation biases from FILE." << endl
	 << "                   The format is [length] [alignment efficiency relative to reference]" << endl
	 << "                   where the efficiency is 1 if there is no relative observation bias." << endl
	 << endl
         << "algorithmic features:" << endl
         << endl
	 << "   --report-genotype-likelihood-max" << endl
	 << "                   Report genotypes using the maximum-likelihood estimate provided" << endl
	 << "                   from genotype likelihoods." << endl
         << "   -M --site-selection-max-iterations N" << endl
         << "                   Uses hill-climbing algorithm to search posterior space for N" << endl
         << "                   iterations to determine if the site should be evaluated.  Set to 0" << endl
         << "                   to prevent use of this algorithm for site selection, and" << endl
         << "                   to a low integer for improvide site selection at a slight" << endl
         << "                   performance penalty. default: 5." << endl
         << "   -B --genotyping-max-iterations N" << endl
         << "                   Iterate no more than N times during genotyping step. default: 25." << endl
         << "   --genotyping-max-banddepth N" << endl
         << "                   Integrate no deeper than the Nth best genotype by likelihood when" << endl
         << "                   genotyping. default: 6." << endl
         << "   -W --posterior-integration-limits N,M" << endl
         << "                   Integrate all genotype combinations in our posterior space" << endl
         << "                   which include no more than N samples with their Mth best" << endl
         << "                   data likelihood. default: 1,3." << endl
         << "   -K --no-permute" << endl
         << "                   Do not scale prior probability of genotype combination given allele" << endl
         << "                   frequency by the number of permutations of included genotypes." << endl
         << "   -N --exclude-unobserved-genotypes" << endl
         << "                   Skip sample genotypings for which the sample has no supporting reads." << endl
         << "   -S --genotype-variant-threshold N" << endl
         << "                   Limit posterior integration to samples where the second-best" << endl
         << "                   genotype likelihood is no more than log(N) from the highest" << endl
         << "                   genotype likelihood for the sample.  default: ~unbounded" << endl
         << "   -j --use-mapping-quality" << endl
         << "                   Use mapping quality of alleles when calculating data likelihoods." << endl
         << "   -D --read-dependence-factor N" << endl
         << "                   Incorporate non-independence of reads by scaling successive" << endl
         << "                   observations by this factor during data likelihood" << endl
         << "                   calculations.  default: 0.9" << endl
         << "   -= --no-marginals" << endl
         << "                   Do not calculate the marginal probability of genotypes.  Saves" << endl
         << "                   time and improves scaling performance in large populations." << endl
         << endl
         << "debugging:" << endl
         << endl
         << "   -d --debug      Print debugging output." << endl
         << "   -dd             Print more verbose debugging output (requires \"make DEBUG\")" << endl
         << endl
         << endl
         << "author:   Erik Garrison <erik.garrison@bc.edu>, Marth Lab, Boston College, 2010-2012" << endl
         << "date:     " << FREEBAYES_COMPILE_DATE << endl
         << "version:  " << FREEBAYES_VERSION << endl;

}


Parameters::Parameters() {

   
    // set defaults

    // i/o parameters:
    useStdin = false;               // -c --stdin
    fasta = "";                // -f --fasta-reference
    targets = "";              // -t --targets
    samples = "";              // -s --samples
    populationsFile = "";
    cnvFile = "";
    output = "vcf";               // -v --vcf
    outputFile = "";
    traceFile = "";
    failedFile = "";
    alleleObservationBiasFile = "";

    // operation parameters
    outputAlleles = false;          //
    trace = false;                  // -L --trace
    useDuplicateReads = false;      // -E --use-duplicate-reads
    suppressOutput = false;         // -N --suppress-output
    useBestNAlleles = 0;         // -n --use-best-n-alleles
    forceRefAllele = false;         // -Z --use-reference-allele
    useRefAllele = false;           // .....
    diploidReference = false;      // -H --diploid-reference
    allowIndels = true;            // -i --no-indels
    leftAlignIndels = false;       // -O --left-align-indels
    allowMNPs = true;            // -X --no-mnps
    allowSNPs = true;          // -I --no-snps
    allowComplex = false;
    maxComplexGap = 3;
    pooled = false;                 // -J --pooled
    ewensPriors = true;
    permute = true;                // -K --permute
    useMappingQuality = false;
    obsBinomialPriors = false; // TODO
    hwePriors = false;
    alleleBalancePriors = false;
    excludeUnobservedGenotypes = false;
    excludePartiallyObservedGenotypes = false;
    genotypeVariantThreshold = 0;
    siteSelectionMaxIterations = 5;
    reportGenotypeLikelihoodMax = false;
    genotypingMaxIterations = 25;
    genotypingMaxBandDepth = 7;
    minPairedAltCount = 0;
    minAltMeanMapQ = 0;
    reportAllHaplotypeAlleles = false;
    onlyUseInputAlleles = false;
    MQR = 100;                     // -M --reference-mapping-quality
    BQR = 60;                     // -B --reference-base-quality
    ploidy = 2;                  // -p --ploidy
    MQL0 = 0;                    // -m --min-mapping-quality
    BQL0 = 0;                    // -q --min-base-quality
    MQL1 = 0;                    // -R --min-supporting-quality MQ,BQ
    BQL1 = 0;                    //
    BQL2 = 10;                    // -Q --mismatch-base-quality-threshold
    RMU = 10000000;                     // -U --read-mismatch-limit
    readMaxMismatchFraction = 1.0;    //  -z --read-max-mismatch-fraction
    readSnpLimit = 10000000;       // -$ --read-snp-limit
    readIndelLimit = 10000000;     // -e --read-indel-limit
    IDW = -1;                     // -x --indel-exclusion-window
    TH = 10e-3;              // -T --theta
    PVL = 0.0001;             // -P --pvar
    RDF = 0.9;             // -D --read-dependence-factor
    diffusionPriorScalar = 1.0;     // -V --diffusion-prior-scalar
    WB = 1;                      // -W --posterior-integration-limits
    TB = 3;
    posteriorIntegrationDepth = 0;
    calculateMarginals = true;
    minAltFraction = 0.2;  // require 20% of reads from sample to be supporting the same alternate to consider
    minIndelAltFraction = 0.2; 
    minAltCount = 2; // require 2 reads in same sample call
    minAltTotal = 1;
    minAltQSum = 0;
    minCoverage = 0;
    debuglevel = 0;
    debug = false;
    debug2 = false;

    showReferenceRepeats = false;
    processInputPositionsOnly = false;
   

}
