/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef STACKENGINE_H
#define STACKENGINE_H

#include "api/BamReader.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <list>

#include "ExtendedReadInfo.h"
#include "CrossHypotheses.h"
#include "BiasGenerator.h"
#include "SigmaGenerator.h"
#include "SkewGenerator.h"
#include "ExtendParameters.h"
#include "PosteriorInference.h"

using namespace std;

// what is the set of parameters describing a complete set of latent variables under some state
class LatentSlate{
public:
    // frequency
    PosteriorInference cur_posterior;
    // information by strand
//    PosteriorInference fwd_posterior;
//    PosteriorInference rev_posterior;
    
    // track important latent variables
    BasicBiasGenerator bias_generator;
    // track filter
    BiasChecker bias_checker;
    // and the other important set of latent variables
    StrandedSigmaGenerator sigma_generator;
    // and the third set
    BasicSkewGenerator skew_generator;
   
    bool detailed_integral;
    int max_iterations;
    int iter_done;
    vector<float> ll_at_stage;
    vector<float> start_freq_of_winner;
 
    int DEBUG;

    void FastExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency);
    void LocalExecuteInference(ShortStack &total_theory, bool update_frequency, bool update_sigma, vector<float> &start_frequency);
    void FastStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
    void DetailedStep(ShortStack &total_theory, bool update_frequency, bool update_sigma);
    void ScanStrandPosterior(ShortStack &total_theory, bool vs_ref);
    void ResetToOrigin();
    void PropagateTuningParameters(EnsembleEvalTuningParameters &my_params, int num_hyp_no_null);
    void SetAndPropagateDebug(int debug);
    LatentSlate(int debug = 0){
        max_iterations = 10;
        detailed_integral = true;
        iter_done = 0;
        DEBUG = debug;
        SetAndPropagateDebug(DEBUG);
    };
};

class HypothesisStack{
public:
    // latent variables under states of the world
    LatentSlate cur_state;
    ShortStack total_theory;
    EnsembleEvalTuningParameters my_params;
    bool try_alternatives;
    int DEBUG;
    vcf::Variant* variant; // just for debug message.

    vector<float> ll_record;
    vector<vector <float> > try_hyp_freq;

    HypothesisStack(){
        DefaultValues();
    }
    void AllocateFrequencyStarts(int num_hyp_no_null, vector<AlleleIdentity> &allele_identity_vector);
    void DefaultValues();
    void PropagateTuningParameters(int num_hyp_no_null);

    // starting to make inferences
    void RestoreFullInference();
    void SetAlternateFromMain();
    void ExecuteExtremeInferences();
    void TriangulateRestart();
    float ExecuteOneRestart(vector<float> &restart_hyp);
    void ExecuteInference();
    void InitForInference(PersistingThreadObjects &thread_objects, vector<const Alignment *>& read_stack, const InputStructures &global_context, vector<AlleleIdentity> &allele_identity_vector);

    // tool for posterior density estimation
    bool CallByIntegral(float af_cutoff_rej, float af_cutoff_gt, vector<int> &genotype_component, float &quasi_phred_quality_score, float &reject_status_quality_score, int &qual_type);

    float ReturnMaxLL();
};

class PositionInBam;

class EnsembleEval {
public:
    // Raw read information
    vector<const Alignment *> read_stack;    //!< Reads spanning the variant position
    // Raw alleles information
    vcf::Variant *         variant;                 //!< VCF record of this variant position
    vector<AlleleIdentity> allele_identity_vector;  //!< Detailed information for each candidate allele
    LocalReferenceContext  seq_context;             //!< Reference context of this variant position
    int                    multiallele_window_start;
    int                    multiallele_window_end;
    vector<string>         info_fields;
    bool                   doRealignment;
    int                    DEBUG;
    // Allele evaluation information
    HypothesisStack allele_eval;
    vector<int> diploid_choice;
    vector<bool> is_possible_polyploidy_allele;
    //@TODO: Use enumerate to represent the fd-code
    vector<vector<int> > global_flow_disruptive_matrix;
    
    EnsembleEval(vcf::Variant &candidate_variant) {
        diploid_choice.clear();
        variant = &candidate_variant;
        allele_eval.variant = variant;
        info_fields.clear();
        multiallele_window_start = -1;
        multiallele_window_end = -1;
        doRealignment = false;
        DEBUG = 0;
        read_id_.clear();
        strand_id_.clear();
        dist_to_left_.clear();
        dist_to_right_.clear();
    };

    //! @brief Set the parameters of the evaluator
    void SetAndPropagateParameters(ExtendParameters* parameters, bool use_molecular_tag, const vector<VariantSpecificParams>& variant_specific_params);
    //! @brief Generate the base space hypotheses for each read
    void SpliceAllelesIntoReads(PersistingThreadObjects &thread_objects, const InputStructures &global_context,
                                const ExtendParameters &parameters, const ReferenceReader &ref_reader);
    //! @brief Get MLLD from the evaluator
    void ScanSupportingEvidence(float &mean_ll_delta, int i_allele);
    //! @brief Calculate multi-min-allele-freq
    void MultiMinAlleleFreq(const vector<float>& multi_min_allele_freq);
    //! @brief Output additional information in the INFO field to make off-line filter possible
    void GatherInfoForOfflineFiltering(const ControlCallAndFilters &my_controls, int best_allele_index);

    //------------------------------------------------------------------
    // Functions for molecular tagging
    //! @brief Set the min family size that will be used in the evaluator.
    void SetEffectiveMinFamilySize(const ExtendParameters& parameters, const vector<VariantSpecificParams>& variant_specific_params);
    //! @brief Calculate the tag similarity for molecular tagging
    void CalculateTagSimilarity(const MolecularTagManager& mol_tag_manager, int max_alt_cov, int sample_idx);
    //! @brief Calculate the variant family histogram
    void VariantFamilySizeHistogram();
    //------------------------------------------------------------------
    // Functions of allele related (not go to the reads) are defined here
    //! @brief Setup the alleles, i.e., context investigation, allele classification, etc.
    void SetupAllAlleles(const ExtendParameters &parameters, const InputStructures &global_context,
                         const ReferenceReader &ref_reader);
    //! @brief Filter out undesired alleles
    void FilterAllAlleles(const ControlCallAndFilters& my_controls, const vector<VariantSpecificParams>& variant_specific_params);
    //! @brief Calculate the end of the look ahead window (primarily for candidate generator)
    int CalculateLookAheadEnd0(const ReferenceReader &ref_reader);
    //! @brief Split the current variant into as many callable smaller variants as possible (primarily for candidate generator))
    void SplitMyAlleleIdentityVector(list<list<int> >& allele_group, const ReferenceReader &ref_reader, int max_group_size_allowed);
    //------------------------------------------------------------------
    // Functions for filling read stack are defined here
    //! @brief Fill the read stack w/o molecular tagging
    void StackUpOneVariant(const ExtendParameters &parameters, const PositionInProgress& bam_position, int sample_index);
    //! @brief Fill the read stack w/ molecular tagging
    void StackUpOneVariantMolTag(const ExtendParameters &parameters, vector< vector< MolecularFamily> > &my_molecular_families_one_strand, int sample_index);
    //! @brief Strategic downsampling algorithm for molecular tagging
    void DoDownSamplingMolTag(const ExtendParameters &parameters, unsigned int effective_min_fam_size, vector< vector< MolecularFamily> > &my_molecular_families,
  		                      unsigned int num_reads_available, unsigned int num_func_fam, int strand_key);
    //------------------------------------------------------------------
    // Functions of hard classification of reads/families are defined here
    //! @brief Hard classification of reads
    void ApproximateHardClassifierForReads();
    //!@ brief Hard classification of families for molecular tagging
    void ApproximateHardClassifierForFamilies(); // calculate the family id for mol taging
    //------------------------------------------------------------------
    // Functions for high level evaluation results (e.g. QUAL, GT, GQ) are defined here
    //! @brief Determine the best two alleles
    int DetectBestMultiAllelePair();
    //! @brief Determine the best two alleles and output the allele_freq_estimation in the detection of best allele pair
    int DetectBestMultiAllelePair(vector<float>& allele_freq_estimation);
    //! @brief Determine possible polyploidy alleles
    void DetectPossiblePolyploidyAlleles(const vector<float>& allele_freq, const ControlCallAndFilters &my_controls, const vector<VariantSpecificParams>& variant_specific_params);
    //! @brief Get allele frequency cutoff according to the level of FD or allele type
    void ServeAfCutoff(const ControlCallAndFilters &my_controls, const vector<VariantSpecificParams>& variant_specific_params,
    		float& af_cutoff_rej, float& af_cutoff_gt);
    //! @brief Calculate QUAL, GT, GQ
    void MultiAlleleGenotype(float af_cutoff_rej, float af_cutoff_gt, vector<int> &genotype_component,
	                         float &gt_quality_score, float &reject_status_quality_score, int &qualiy_type);
    //------------------------------------------------------------------
    // Functions of flow-disruption related are defined here
    //! @brief Determine the flow-disruptiveness between hypotheses pairs for each read
    void FlowDisruptivenessInReadLevel(const InputStructures &global_context);
    //! @brief Determine the flow-disruptiveness between hypotheses pairs by looking at all reads
    void FlowDisruptivenessInReadStackLevel(float min_ratio_for_fd);

    friend void GlueOutputVariant(EnsembleEval &my_ensemble, VariantCandidate &candidate_variant, const ExtendParameters &parameters, int _best_allele_index, int sample_index); // I want to access your private members

private:
    // The following private members are the results of approximate hard classification for reads
    vector<int> read_id_;        // vector of allele ids per read, -1 = outlier, 0 = ref, >0 real allele
    vector<bool> strand_id_;     // vector of forward (true) or reverse (false) per read
    // for each variant, calculate its' position within the soft clipped read distance to left and distance to right
    vector<int> dist_to_left_;   // vector of distances from allele position to left soft clip per read
    vector<int> dist_to_right_;  // vector of distances from allele position to left soft clip per read

    // The followings are for tag similarity
    vector<vector<unsigned int> > alt_fam_indices_; // alt_fam_indices_[i_allele] stores the indices of families (of allele_eval.total_theory.my_eval_families) that support i_allele
    vector<int> tag_similar_counts_;
};

string PrintVariant(const vcf::Variant& variant); // just for debug message

void FindNodesInIsoSubGraph(const vector<vector<bool> >& connectivity_matrix, list<list<int> >& subgraph_to_nodes, bool sort_by_index);
void SplitAlleleIdentityVector(const vector<AlleleIdentity>& allele_identity_vector, list<list<int> >& allele_groups, const ReferenceReader& ref_reader, int max_group_size_allowed);
BasicFilters const * ServeBasicFilterByType(const AlleleIdentity& variant_identity, const ControlCallAndFilters& my_controls);
float FreqThresholdByType(const AlleleIdentity& variant_identity, const ControlCallAndFilters &my_controls, const VariantSpecificParams& variant_specific_params);


#endif // STACKENGINE_H
