#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <map>
#include <cmath>
#include "sam.h"
//#include "kstring.h"
#include "IonVersion.h"


using namespace std;

int min_mapping_qv = 0;

map<pair<string, long int>, pair<long int, bool> > map_target_size;
map<string, vector<pair<long int, long int> > > map_target_ranges;
void init_target_size() {
    map_target_ranges.clear();
}

//-----------------------------------------------------
string itos(long int i) {	// convert int to string
    stringstream s;
    s << i;
    return s.str();
}
//-----------------------------------------------------
static int depth;

typedef struct {
    int beg, end;
    samfile_t *in;
} tmpstruct_t;

// callback for bam_fetch()
static int fetch_func(const bam1_t *b, void *data) {
    if (b->core.qual < min_mapping_qv) {
        return 1;
    }
    bam_plbuf_t *buf = (bam_plbuf_t*)data;
    bam_plbuf_push(b, buf);
    return 0;
}
// callback for bam_plbuf_init()
static int pileup_func(uint32_t tid, uint32_t pos, int n, const bam_pileup1_t *pl, void *data) {
    tmpstruct_t *tmp = (tmpstruct_t*)data;
    if ((int)pos >= tmp->beg && (int)pos < tmp->end)  {
        //printf("%s\t%d\t%d\n", tmp->in->header->target_name[tid], pos + 1, n);
        depth = n;
    }
    return 0;
}

//--------------------------------------------------------

string vartypes[5] {"SNP","DEL","INS","MNP","OTHER"};
inline int vartype(const string ref, const string alt) {
    if(ref.length() ==  alt.length()) {
        if(ref.length() == 1) return 0;
        else return 3;
    }
    if(ref.length() <  alt.length()) return 2;
    else return 1;
}
//--------------------------------------------------------
struct VariantFeatures {
private:
    std::map<string, string> * info_key_value;
public:
    string vc_record;
    char gt_index;

    inline int get_int(string info_key) {
        if(info_key_value!=NULL) {
            std::map< string , string >::iterator key_val = info_key_value->find(info_key);
            if( key_val != info_key_value->end() && key_val->second.length() >0 && isdigit((key_val->second)[0])) return atoi(&((key_val->second)[0])); //dangerous but fast, potential bug
        }
        return -1;
    }

    inline float get_float(string info_key) {
        if(info_key_value!=NULL) {
            std::map< string , string >::iterator key_val = info_key_value->find(info_key);
            if( key_val != info_key_value->end() && key_val->second.length() >0 && (isdigit((key_val->second)[0]) || (key_val->second)[0] == '-' || (key_val->second)[0] == '+')) return atof(&((key_val->second)[0])); //dangerous but fast, potential bug
        }
        return -1;
    }


    VariantFeatures(string _vc_record, char _index) {
        vc_record = _vc_record;
        gt_index = _index;
        info_key_value=NULL;
    };
    ~VariantFeatures() {
        if(info_key_value != NULL) delete info_key_value;
    }

    void  parse_info_values() {

        int col_idx = 0;
        std::stringstream cols(vc_record);
        string vcf_col;
        while(cols.good() && getline( cols, vcf_col, '\t') && ++col_idx<6);
        if(col_idx == 6) {

            if(info_key_value != NULL) delete info_key_value;
            info_key_value = new std::map<string, string>();
            info_key_value->insert(std::pair<string, string>("QUAL", vcf_col));

            while(cols.good() && getline( cols, vcf_col, '\t') && ++col_idx<8);
            if(col_idx == 8 && !vcf_col.empty()) {
                std::stringstream features(vcf_col);
                string flag;
                while(features.good() && getline( features, flag, ';')) {
                    string::size_type pos = flag.find("=",0);
                    if(pos!=string::npos && pos < flag.length()-1) {
                        int val_idx = 0;
                        string flag_name = flag.substr(0,pos);
                        std::stringstream values(flag.substr(pos+1));
                        string value;
                        while(values.good() && getline( values, value, ',') && ++val_idx < (int)gt_index);
                        if(val_idx == (int)gt_index || val_idx == 1) info_key_value->insert(std::pair<string, string>(flag_name, value)); // if value already exists, that means that it is repeated twice in the vcf record, we keep first copy
                    }
                }
            }
        }
    };

};
//--------------------------------------------------------
struct VCFinfo {
private:
    vector<string> ref;
    vector<string> alt;
    vector<VariantFeatures> info;
    int zyg;   // -1 = not available, 0 = HOM, 1 = HET/REF, 2 = HET/NON-REF
    int DP;    //  0 = zero or not available
    long int target_end_pos;

public:

    inline void set_dp(int dp) {
        DP= dp;
    }
    inline void set_zyg(int _zyg) {
        zyg = _zyg;
    }
    inline int  get_zyg()  {
        return zyg;
    }
    inline void set_target_end_pos(long int pos) {
        target_end_pos = pos;
    }
    inline long int get_target_end_pos() {
        return target_end_pos;
    }

    void set_dp(string _info) {
        DP=0;
        if(!_info.empty()) {
            string::size_type  pos = _info.find("DP=");
            if(pos!=string::npos && _info.length()>pos+3 && isdigit(_info[pos+3]))  DP = atoi(&(_info[pos+3])); // fast, but dangerous, assumes string is represented in continuous array, potential bug
        }
    }

    int get_var_type(int idx) { //SNP=0,DEL=1,INS=2,MNP=3,OTHER=4
        if(idx>=(int)ref.size()) {
            cerr << "get_var_type(idx): requested idx is larger than allele counts" << endl;
            exit(1);
        }
        return vartype(ref.at(idx), alt.at(idx));
    }

    float get_int_flag_value(int idx,string FLAG) {
        if(idx>=(int)info.size()) {
            cerr << "get_int_flag_value(idx): requested idx is larger than allele counts" << endl;
            exit(1);
        }
        return info.at(idx).get_float(FLAG);
    }

    int alt_count() {
        return alt.size();
    }
    const string get_alt(int idx) {
        return alt.at(idx);
    }
    const string get_ref(int idx) {
        return ref.at(idx);
    }
    const string get_info(int idx) {
        return info.at(idx).vc_record;
    }
    char get_gt_index(int idx) {
        return info.at(idx).gt_index;
    }


    void  parse_info_values() {
        for(unsigned int i=0; i<info.size(); i++)
            info.at(i).parse_info_values();
    }

    const string alt_to_str() {
        stringstream s;
        if(alt.size()>0) s << alt.at(0);
        for(unsigned int i=1; i<alt.size(); i++) s << "," << alt.at(i) ;
        return s.str();
    }

    const string ref_to_str() {
        stringstream s;
        if(ref.size()>0) s << ref.at(0);
        for(unsigned int i=1; i<ref.size(); i++) s << "," << ref.at(i) ;
        return s.str();
    }

    void inline add(const string _ref, const string _alt, const string _info, const char _gt_index, const int _zyg) {
        ref.push_back(_ref);
        alt.push_back(_alt);
        info.push_back(VariantFeatures(_info, _gt_index));
        if(info.size()==1) {
            set_dp(_info);
            set_zyg(_zyg);
        }
    }

    void inline add(VCFinfo & rec) {
        int original_size = alt_count();
        for(int i=0; i<rec.alt_count(); i++) {
            bool already_added = false;
            for(int j=0; j<original_size; j++)
                if(strcmp(ref.at(j).c_str(),rec.get_ref(i).c_str())==0 && strcmp(alt.at(j).c_str(),rec.get_alt(i).c_str())==0) {
                    already_added = true;
                    break;
                }
            if(!already_added) add(rec.get_ref(i), rec.get_alt(i), rec.get_info(i), rec.get_gt_index(i), rec.get_zyg());
        }
    }

    VCFinfo(const string _ref, const string _alt, const string _info, const char _gt_index,  const int _zyg) {
        add(_ref, _alt, _info, _gt_index, _zyg);
    }

    inline int get_dp() { // returns -1 if not available
        return DP;
    }

    VCFinfo(string line, long &alleles_loaded, long &rows_missing_genotype, bool ignore_genotype = false) {
// parsing VCF format 4.0 or later, example:
//#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SampleName
//chr1	1573	.	G	A,T	2.8	PASS	ff=5	GT:AF:DP	0/2:0,0.4:1607

        int col_idx = 0;
        int GT_idx = -1;
        bool obs_ref_allele = false;
        string ref_al = "";
        string info_al = line;
        vector<int> added_alt;

        std::stringstream cols(line);
        while(cols.good()) {
            string vcf_col;
            getline( cols, vcf_col, '\t');

            switch(++col_idx) {
            case 4: {
                ref_al = vcf_col;
                break;
            }
            case 5: {
                std::stringstream ss(vcf_col);
                while(ss.good()) {
                    string tmp_alt;
                    getline( ss, tmp_alt, ',');
                    alt.push_back(tmp_alt);
                }
                break;
            }
            case 8: {
                //info_al = vcf_col;
                break;
            }
            case 9: {
                std::stringstream ss(vcf_col);
                int gt_col = -1;
                string tmp_format;
                while(ss.good()) {
                    gt_col++;
                    getline( ss, tmp_format, ':');
                    if(!ignore_genotype && strcmp(tmp_format.c_str(), "GT")==0) {
                        GT_idx = gt_col;
                        break;
                    }
                }
                break;
            }
            case 10: {
                if(GT_idx==-1) break;
                int tmp_idx = GT_idx; // for now GT_idx represents column index
                GT_idx = -1;
                std::stringstream ss(vcf_col);
                int gt_col = -1;
                string tmp_value;
                while(ss.good()) {
                    gt_col++;
                    getline( ss, tmp_value, ':');
                    if(gt_col == tmp_idx) {
                        if( tmp_value.empty()) break;
                        std::stringstream sf(tmp_value);

                        char separator = '|';
                        if(tmp_value.find("/",0) != string::npos) separator = '/';
                        else if(tmp_value.find("|",0) == string::npos) break;

                        GT_idx = 0; // now GT_idx represents allele number

                        string gt_value;
                        vector<string> alt_subset;
                        while(sf.good()) {
                            getline(sf, gt_value, separator);
                            if(gt_value.length()>0 && isdigit(gt_value[0])) {
                                if( gt_value[0]!='0') {
                                    GT_idx = atoi(gt_value.c_str());
                                    if(GT_idx-1<(int)alt.size()) {
                                        bool already_added = false;
                                        for(unsigned int i=0; i<added_alt.size(); i++) if(added_alt.at(i)==GT_idx) {
                                                already_added= true;
                                                break;
                                            }
                                        if(!already_added) {
                                            alt_subset.push_back(alt[GT_idx-1]);
                                            added_alt.push_back(GT_idx);
                                        }
                                    }
                                } else obs_ref_allele = true;
                            }
                        }
                        alt = alt_subset;
                        break;
                    }
                }
                break;
            }

            }
        }

        if((alleles_loaded = alt.size()) > 0) {
            if(GT_idx == -1) {
                rows_missing_genotype++;
                set_zyg(-1);
            } else set_zyg( (!obs_ref_allele) ? ( alleles_loaded == 1 ? 0 : 2) : 1 );
            set_dp(info_al);

            for(int i=0; i<alleles_loaded; i++) {
                ref.push_back(ref_al);
                info.push_back(VariantFeatures(info_al, (GT_idx == -1) ? (i+1) : (char)added_alt.at(i)));
            }
        }
    }

};

//--------------------------------------
class FASTA {
    std::map<string,string > contig;

public:
    int load_file(string fasta_file) {
        ifstream infile;
        infile.open(fasta_file.c_str());
        string line;

        if (!infile) {
            cerr << "Unable to read " <<  fasta_file << endl;
            exit(1);
        }

        int contigs_loaded = 0;
        string contig_name = "";
        string sequence = "";
        while (getline(infile, line)) {
            if(line[0]=='>') {
                if(contig_name!="") {
                    int pos = contig_name.find("\t");
                    if (pos != -1) {contig_name = contig_name.substr(0, pos);}
                    std::pair<std::map<string,string>::iterator,bool> ret = contig.insert ( std::pair<string, string>(contig_name, sequence));
                    if (ret.second==false) {
                        cerr << "Contig is listed twice in fasta. Abort! " << contig_name << endl;
                        exit(1);
                    }
                    contigs_loaded++;
                    cerr << contigs_loaded << " loaded " << contig_name << " length: " << sequence.length() << endl;
                }
                contig_name = line.substr(1);
                sequence="";
            } else if(line[0]!='#') {
                sequence += line;
            }
        }

        if(contig_name!="") {
            int pos = contig_name.find("\t");
            if (pos != -1) {contig_name = contig_name.substr(0, pos);}
            std::pair<std::map<string,string>::iterator,bool> ret = contig.insert ( std::pair<string, string>(contig_name, sequence));
            if (ret.second==false) {
                cerr << "Contig is listed twice in fasta. Abort! " << contig_name << endl;
                exit(1);
            }
            contigs_loaded++;
            cerr << contigs_loaded << " loaded " << contig_name << " length: " << sequence.length() << endl;
        }
        return contigs_loaded;
    }


    int size() {
        return contig.size();
    }

    bool left_align_indel(string chr, long & vcf_pos, string & ref, string & alt) {
        int prefsize = vcf_pos<70?vcf_pos:70; //caution: assuming that vcf_pos > 0, based on vcf-format
        string prefseq = "";
		try {prefseq = contig.find(chr)->second.substr(vcf_pos - prefsize, prefsize - 1);} catch(...) {prefseq = "";}
        string refseq = prefseq + ref;
        string altseq = prefseq + alt;

        int reflen = refseq.length();
        int altlen = altseq.length();
        int right_match = 0;

        while( right_match < reflen && right_match < altlen && refseq[reflen - 1 - right_match]==altseq[altlen - 1 - right_match]) {
            right_match++;
        }
        if(right_match > 0) {
            long l = reflen-right_match-ref.length();
            if (l < 0) {ref = refseq.substr(0, ref.length());} 
            else {ref = refseq.substr(reflen-right_match-ref.length(), ref.length());}
            l = altlen-right_match-alt.length();
            if (l < 0) {alt = altseq.substr(0, alt.length());}
            else {alt = altseq.substr(altlen-right_match-alt.length(), alt.length());}
            vcf_pos -= right_match;
            return true;
        }
        return false;
    }

};

//-----------------------------------------------------
class VCFList {
    std::map<string,std::map<long int,VCFinfo> > vcfrec;

public:
    void remove(VCFList* p, bool zygosity_match, bool allele_match) {
        if (p == NULL) {return;}
        for (std::map<string,std::map<long int,VCFinfo> >::iterator iter = p->vcfrec.begin(); (iter != p->vcfrec.end()); ++iter) {
            for (std::map<long int,VCFinfo>::iterator iter_pos = iter->second.begin(); (iter_pos != iter->second.end()); ++iter_pos) {
                std::map<string,std::map<long int,VCFinfo> >::iterator iter2 = vcfrec.find(iter->first);
                if (iter2 != vcfrec.end()) {
                    std::map<long int,VCFinfo>::iterator iter2_pos = iter2->second.find(iter_pos->first);
                    if (iter2_pos != iter2->second.end()) {
                        if (zygosity_match) {
                            if ((iter_pos->second.ref_to_str().find(iter2_pos->second.ref_to_str()) != string::npos) and (iter_pos->second.alt_to_str().find(iter2_pos->second.alt_to_str()) != string::npos)and (iter2_pos->second.get_zyg() == iter_pos->second.get_zyg())) {iter2->second.erase(iter2_pos);}
                        }
                        else if (allele_match) {
							cerr << iter2_pos->second.ref_to_str() << endl;
							cerr << iter_pos->second.ref_to_str() << endl;
							cerr << iter2_pos->second.alt_to_str() << endl;
							cerr << iter_pos->second.alt_to_str() << endl;
                            if ((iter_pos->second.ref_to_str().find(iter2_pos->second.ref_to_str()) != string::npos) and (iter_pos->second.alt_to_str().find(iter2_pos->second.alt_to_str()) != string::npos)) {iter2->second.erase(iter2_pos);}
                        }
                        else {
                            iter2->second.erase(iter2_pos);
                        }
                    }
                }
            }
        }
    }
	
    long calculate_target_size(int DP) {
        long target_size = 0;
        for (map<string, map<long int, VCFinfo> >::iterator chrit = vcfrec.begin(); (chrit != vcfrec.end()); ++chrit) {
            for (map<long int, VCFinfo>::iterator posit = chrit->second.begin(); (posit != chrit->second.end()); ++posit) {
                if (posit->second.get_dp() >= DP) {
                    target_size += (posit->second.get_zyg() - posit->first);
                }
            }
        }
        return target_size;
    }

private:
    void add(string chr, long genpos, VCFinfo & rec) {
        std::map< string , std::map<long int,VCFinfo> >::iterator it = vcfrec.find(chr);
        if(it == vcfrec.end()) {
            std::map<long int,VCFinfo> emptymap;
            it = vcfrec.insert(std::pair< string , std::map<long int,VCFinfo> > (chr, emptymap)).first;
        }

        std::pair<std::map<long int,VCFinfo>::iterator,bool> ret = (it->second).insert ( std::pair<long int,VCFinfo>(genpos, rec));
        if (ret.second==false) ret.first->second.add(rec);
    }

    bool add(const string chr, const long genpos, const string ref, const string alt, const string info, const char gt_index, const int zyg) {
        std::map< string , std::map<long int,VCFinfo> >::iterator it = vcfrec.find(chr);
        if(it == vcfrec.end()) {
            std::map<long int,VCFinfo> emptymap;
            it = vcfrec.insert(std::pair< string , std::map<long int,VCFinfo> > (chr, emptymap)).first;
        }

        std::map<long int,VCFinfo>::iterator ret = (it->second).find(genpos);
        if(ret == (it->second).end()) {
            (it->second).insert ( std::pair<long int,VCFinfo>(genpos, VCFinfo(ref,alt,info, gt_index, zyg)));
        } else {
            if(ret->second.get_zyg() > 9) { // this is used only for BED files to store second coordinate
                if(ret->second.get_zyg() < zyg) ret->second.set_zyg(zyg);
            } else {
                for(int i=0; i<ret->second.alt_count(); i++) {
                    if(strcmp(ref.c_str(),ret->second.get_ref(i).c_str())==0 && strcmp(alt.c_str(),ret->second.get_alt(i).c_str())==0) return false;
                }
                ret->second.add(ref, alt, info, gt_index, zyg);
            }
        }
        return true;
    }


    void add(string vcfline, long &alleles_loaded, long &rows_missing_genotype, long &het_rows, long & hom_rows, FASTA * reference = NULL, bool split_mnp = true, bool ignore_genotype = false) {
        alleles_loaded = 0;
        string::size_type pos = vcfline.find("\t",0);

        if( pos != string::npos && pos > 0 && pos < vcfline.length() - 1) {
            string chr    = vcfline.substr(0,pos);
            long   genpos = atoi(vcfline.substr(pos+1).c_str());

            // extracting genotyped/or all alleles at that position and the info
            // this eliminates non-called alleles that are mixed with genotyped alleles

            VCFinfo rec(vcfline, alleles_loaded, rows_missing_genotype, ignore_genotype);

            if(rec.get_zyg()==1) het_rows++;
            if(rec.get_zyg()==0) hom_rows++;
            for(int i=0; i<rec.alt_count(); i++) {
                string ref = rec.get_ref(i);
                string alt = rec.get_alt(i);
                long shift = 0;

                //remove common padding bases
                //TG AG  -> T A
                while(shift < (long)ref.length()-1 && shift< (long)alt.length()-1 && ref[(long)ref.length()-1-shift]==alt[(long)alt.length()-1-shift]) shift++;
                if(shift>0) {
                    ref = ref.substr(0, ref.length()-shift);
                    alt = alt.substr(0,alt.length()-shift);
                }
                //TA TG  -> A G
                shift = 0;
                while(shift < (long)ref.length()-1 && shift< (long)alt.length()-1 && ref[shift]==alt[shift]) shift++;
                if(shift>0) {
                    ref = ref.substr(shift);
                    alt = alt.substr(shift);
                }
                long adjusted_pos = genpos+shift;

                int vt = vartype(ref,alt);

                //left-align indels
                if(reference!=NULL && (vt == 1 || vt == 2))  reference->left_align_indel(chr, adjusted_pos, ref, alt);

                //split-MNPs into single SNPs, this is wrong but community still does it
                if(split_mnp && vt == 3) { // XXX
                    for(unsigned int j=0; j<ref.length(); j++) if(ref.at(j)!=alt.at(j)) add(chr, adjusted_pos + j, ref.substr(j,1), alt.substr(j,1), rec.get_info(i), rec.get_gt_index(i), rec.get_zyg());
                } else if(!add(chr, adjusted_pos, ref, alt, rec.get_info(i), rec.get_gt_index(i), rec.get_zyg())) alleles_loaded--;
            }
        }
    }


public:
    std::map<string,std::map<long int,VCFinfo> > * getList() {
        return &vcfrec;
    }
//-----------------------------------------------------
    long int load_file(string filename, FASTA * reference = NULL, bool split_mnp = true, bool ignore_genotype = false) {
        ifstream infile;
        infile.open(filename.c_str());
        string line;

        if (!infile) {
            cerr << "Unable to read " <<  filename << endl;
            exit(1);
        }

        long rows_processed = 0;
        long rows_loaded = 0;
        long alleles_loaded = 0, al;
        long rows_missing_genotype = 0;
        long het_rows = 0, hom_rows = 0;

        while (getline(infile, line)) {
            if( line[0]!='#') {
                add(line, al, rows_missing_genotype, het_rows, hom_rows, reference, split_mnp, ignore_genotype);
                if(al>0) {
                    rows_loaded++;
                    alleles_loaded+=al;
                }
                rows_processed++;
            }
        }

        infile.close();

        cerr << "# informative vcf records:" << rows_processed << "\n# records containing alternative allele:" << rows_loaded << "\n# loaded alternative alleles:" << alleles_loaded << endl;
        cerr << "# loaded vcf records with HET genotype:" << het_rows << "\n# loaded vcf records with HOM genotype:" << hom_rows << "\n# loaded vcf records with missing genotype:" <<  rows_missing_genotype << endl;
        return rows_loaded;
    }
//-----------------------------------------------------
    long int load_bed_file(string filename) {
        ifstream infile;
        infile.open(filename.c_str());
        string line;

        if (!infile) {
            cerr << "Unable to read " <<  filename << endl;
            exit(1);
        }

        long rows_processed = 0;
        long total_target_size = 0;
        init_target_size();

        while (getline(infile, line))
            if( line[0]!='#') {
                string::size_type pos = line.find("\t",0);

                if( pos != string::npos && pos > 0 && pos < line.length() - 1) {
                    string chr    = line.substr(0,pos);
                    long   start_pos = atoi(line.substr(pos+1).c_str());
                    if( (pos = line.find("\t",pos+1))  != string::npos) {
                        long end_pos = atoi(line.substr(pos+1).c_str());
                        if(add(chr, start_pos, "", "", "", 0, end_pos)) {
                            total_target_size += end_pos - start_pos;
                            map_target_ranges[chr].push_back(make_pair(start_pos, end_pos));
                        }
                    }
                    rows_processed++;
                }
            }

        infile.close();

        cerr << "# informative bed records:" << rows_processed << "\n# total target size:" << total_target_size << endl;
        return rows_processed;
    }

// Supports overlapping target segments (on fly merging)
//-----------------------------------------------------
    long get_target_vcf(VCFList * vcf_in, VCFList * vcf_out) {

        long records_on_target = 0;
        for (std::map< string , std::map<long int,VCFinfo> >::iterator bed_chrit = vcfrec.begin(); bed_chrit != vcfrec.end(); ++bed_chrit) {

            std::map< string , std::map<long int,VCFinfo> >::iterator vcf_chrit = vcf_in->getList()->find(bed_chrit->first);
            if(vcf_chrit == vcf_in->getList()->end()) continue;

            std::map<long int,VCFinfo>::iterator vcf_posit = vcf_chrit->second.begin();

            std::map<long int,VCFinfo>::iterator bed_posit = bed_chrit->second.begin();

            if(bed_posit == bed_chrit->second.end()) continue;

            long int reg_start = bed_posit->first;
            long int reg_end  = bed_posit->second.get_zyg();

            bed_posit++;

            while (bed_posit != bed_chrit->second.end()) {
                if(  vcf_posit == vcf_chrit->second.end()) break;

                if(bed_posit->first <=  reg_end) {
                    if(bed_posit->second.get_zyg() >  reg_end ) reg_end = bed_posit->second.get_zyg();
                    bed_posit++;
                    continue;
                }

                if(  reg_end < vcf_posit->first) {
                    reg_start = bed_posit->first;
                    reg_end  = bed_posit->second.get_zyg();
                    bed_posit++;
                    continue;
                }

                while( vcf_posit != vcf_chrit->second.end() && reg_start >= vcf_posit->first) vcf_posit++;
                while( vcf_posit != vcf_chrit->second.end() && reg_end >= vcf_posit->first) {
                    if((vcf_posit->first + (long)vcf_posit->second.get_ref(0).length()-1) <= reg_end) {
                        vcf_posit->second.set_target_end_pos(reg_end - vcf_posit->first);
                        vcf_out->add(vcf_chrit->first, vcf_posit->first, vcf_posit->second);
                        records_on_target++;
                    }
                    vcf_posit++;
                }
            }

            while( vcf_posit != vcf_chrit->second.end() && reg_start >= vcf_posit->first) vcf_posit++;
            while( vcf_posit != vcf_chrit->second.end() && reg_end >= vcf_posit->first) {
                if(vcf_posit->first + (long)vcf_posit->second.get_ref(0).length()-1 <= reg_end) {
                    vcf_posit->second.set_target_end_pos(reg_end - vcf_posit->first);
                    vcf_out->add(vcf_chrit->first, vcf_posit->first, vcf_posit->second);
                    records_on_target++;
                }
                vcf_posit++;
            }
        }
        return records_on_target;
    }
//-----------------------------------------------------
    long merge_overlapping_segments(VCFList * merged_bed) {
        long total_target_size = 0;
        init_target_size();
        for (std::map< string , std::map<long int,VCFinfo> >::iterator bed_chrit = vcfrec.begin(); bed_chrit != vcfrec.end(); ++bed_chrit) {
            std::map<long int,VCFinfo>::iterator bed_posit = bed_chrit->second.begin();

            if(bed_posit == bed_chrit->second.end()) continue;

            long int reg_start = bed_posit->first;
            long int reg_end  = bed_posit->second.get_zyg();

            bed_posit++;

            while (bed_posit != bed_chrit->second.end()) {

                if(bed_posit->first <=  reg_end) {
                    if(bed_posit->second.get_zyg() >  reg_end ) reg_end = bed_posit->second.get_zyg();
                    bed_posit++;
                    continue;
                }

                merged_bed->add(bed_chrit->first,reg_start, "", "", "", 0, reg_end);
                total_target_size+= reg_end - reg_start;
                map_target_ranges[bed_chrit->first].push_back(make_pair(reg_start, reg_end));

                reg_start = bed_posit->first;
                reg_end  = bed_posit->second.get_zyg();
                bed_posit++;

            }

            merged_bed->add(bed_chrit->first,reg_start, "", "", "", 0, reg_end);
            total_target_size+= reg_end - reg_start;
            map_target_ranges[bed_chrit->first].push_back(make_pair(reg_start, reg_end));
        }

        return total_target_size;
    }
//-----------------------------------------------------
    long intersect(VCFList * in, VCFList * bed_out) {
        long total_target_size = 0;
        init_target_size();
        for (std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit) {
            std::map< string , std::map<long int,VCFinfo> >::iterator in_chrit = in->getList()->find(chrit->first);
            if(in_chrit != in->getList()->end())
                for (std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(), in_posit = in_chrit->second.begin(); posit != chrit->second.end() && in_posit != in_chrit->second.end(); ) {
                    if(posit->first > in_posit->second.get_zyg()) {
                        in_posit++;
                        continue;
                    }
                    if(in_posit->first > posit->second.get_zyg()) {
                        posit++;
                        continue;
                    }
                    long reg_start = in_posit->first < posit->first ? posit->first : in_posit->first;
                    long reg_end = posit->second.get_zyg() < in_posit->second.get_zyg() ? posit->second.get_zyg() : in_posit->second.get_zyg();
                    total_target_size += reg_end - reg_start;
                    map_target_ranges[in_chrit->first].push_back(make_pair(reg_start, reg_end));
                    bed_out->add(chrit->first, reg_start, "", "", "", 0, reg_end);
                    if(posit->second.get_zyg() == reg_end) posit++;
                    else in_posit++;
                }
        }
        return total_target_size;
    }
//-----------------------------------------------------
    void parse_info_values() {
        for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit)
            for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++) posit->second.parse_info_values();
    }

//-----------------------------------------------------
    int * get_var_counts(string basename = "", int DP = -1) {
        ofstream output;
        bool outfile = (basename!="");
        if(outfile)output.open(basename.c_str());
        if(!output) {
            cerr << "Error writing output file: " << basename << endl;
        }

        int * counts = new int[5] {0,0,0,0,0};
        for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit)
            for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++)
                for(int i=0; i< posit->second.alt_count(); i++) {
                    int vartype = posit->second.get_var_type(i);
                    if(DP==-1 || posit->second.get_dp()>=DP) counts[vartype]++;
                    if(output) output  << chrit->first << "\t" << posit->first << "\t*"<< vartypes[vartype] <<"*\t" << posit->second.get_ref(i) << "\t" << posit->second.get_alt(i)<< "\t"<< posit->second.get_dp() << "\t.\t" << posit->second.get_info(i) << endl;
                }

        if(outfile) output.close();

        return counts;
    }
//-----------------------------------------------------
    int ** get_var_counts(string FLAG, float _min, float _step, int nbins, string FLAG2="", char cmp='>', float _val=0) {
        float _max = _min + nbins*_step;

        int ** counts = new int*[2];
        counts[0] = new int[nbins+3];
        counts[1] = new int[nbins+3];

        for(int i=0; i<nbins+3; i++) counts[0][i] = counts[1][i] = 0;

        for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit)
            for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++)
                for(int i=0; i < posit->second.alt_count(); i++) {
                    float flag_value = posit->second.get_int_flag_value(i,FLAG);
                    int vartype = posit->second.get_var_type(i) > 0;

                    if(flag_value == -1 || (!FLAG2.empty() && ((cmp=='<' && posit->second.get_int_flag_value(i,FLAG2)<_val) || (cmp=='>' && posit->second.get_int_flag_value(i,FLAG2)>_val)))) counts[vartype][nbins+2]++;
                    else if(flag_value<_min) counts[vartype][0]++;
                    else if(flag_value >= _max) counts[vartype][nbins+1]++;
                    else
                        counts[vartype][1+(int)((flag_value-_min)/_step)]++;
                }

        return counts;
    }
//-----------------------------------------------------

    void positional_allele_match(VCFList * a, VCFList * match, VCFList * unmatch1, VCFList * unmatch2, bool require_allele_match, bool require_zygosity_match) {
        for (std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit) {
            std::map< string , std::map<long int,VCFinfo> >::iterator achrit = a->getList()->find(chrit->first);
            if(achrit != a->getList()->end()) {
                std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(), aposit = achrit->second.begin();
                while(posit != chrit->second.end() && aposit != achrit->second.end()) {
                    if(posit->first < aposit->first) {
                        unmatch1->add(chrit->first,posit->first, posit->second);
                        posit++;
                        continue;
                    } else if(posit->first > aposit->first) {
                        unmatch2->add(achrit->first,aposit->first, aposit->second);
                        aposit++;
                        continue;
                    } else {

                        if(!require_allele_match) {
                            if(!require_zygosity_match || (aposit->second.get_zyg() != -1 && posit->second.get_zyg() != -1 && aposit->second.get_zyg() == posit->second.get_zyg() ))
                                match->add(achrit->first,aposit->first, aposit->second);
                            else {
                                unmatch1->add(chrit->first,posit->first, posit->second);
                                unmatch2->add(achrit->first,aposit->first, aposit->second);
                            }
                        } else {
                            // allele matching
                            int n_match_alleles = 0;
                            for(int i=0; i < posit->second.alt_count(); i++) {
                                bool has_match = false;
                                for(int ai=0; ai < aposit->second.alt_count(); ai++)
                                    if(strcmp(aposit->second.get_alt(ai).c_str(),posit->second.get_alt(i).c_str()) == 0) {
                                        if(!require_zygosity_match || (aposit->second.get_zyg() != -1 && posit->second.get_zyg() != -1 && aposit->second.get_zyg() == posit->second.get_zyg() ))
                                            match->add(achrit->first,aposit->first, aposit->second.get_ref(ai), aposit->second.get_alt(ai), aposit->second.get_info(ai), aposit->second.get_gt_index(ai), aposit->second.get_zyg());
                                        else {
                                            unmatch1->add(chrit->first,posit->first, posit->second.get_ref(i), posit->second.get_alt(i), posit->second.get_info(i), posit->second.get_gt_index(i), posit->second.get_zyg());
                                            unmatch2->add(achrit->first,aposit->first, aposit->second.get_ref(ai), aposit->second.get_alt(ai), aposit->second.get_info(ai), aposit->second.get_gt_index(ai), aposit->second.get_zyg());
                                        }
                                        has_match = true;
                                        n_match_alleles++;
                                        // break;
                                    }
                                if(!has_match) unmatch1->add(chrit->first,posit->first, posit->second.get_ref(i), posit->second.get_alt(i), posit->second.get_info(i), posit->second.get_gt_index(i), posit->second.get_zyg());
                            }

                            if(n_match_alleles<aposit->second.alt_count())
                                for(int ai=0; ai < aposit->second.alt_count(); ai++) {
                                    bool has_match = false;
                                    for(int i=0; i < posit->second.alt_count(); i++)
                                        if(strcmp(aposit->second.get_alt(ai).c_str(),posit->second.get_alt(i).c_str()) == 0) {
                                            has_match = true;
                                            break;
                                        }
                                    if(!has_match) unmatch2->add(achrit->first,aposit->first, aposit->second.get_ref(ai), aposit->second.get_alt(ai), aposit->second.get_info(ai), aposit->second.get_gt_index(ai), aposit->second.get_zyg());
                                }

                        }

                        ++posit;
                        ++aposit;
                    }
                }
                while(posit != chrit->second.end())  {
                    unmatch1->add(chrit->first,posit->first, posit->second);
                    posit++;
                }
                while(aposit != achrit->second.end())  {
                    unmatch2->add(achrit->first,aposit->first, aposit->second);
                    aposit++;
                }
            } else {
                for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++) {
                    unmatch1->add(chrit->first,posit->first, posit->second);
				}
            }
        }

        for (std::map< string , std::map<long int,VCFinfo> >::iterator achrit = a->getList()->begin(); achrit != a->getList()->end(); ++achrit) {
            std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.find(achrit->first);
            if(chrit == vcfrec.end()) {
                for(std::map<long int,VCFinfo>::iterator aposit = achrit->second.begin(); aposit != achrit->second.end(); aposit++)
                    unmatch2->add(achrit->first,aposit->first, aposit->second);
            }
        }
    }
//-----------------------------------------------------

    void match(VCFList * a, VCFList * tp, VCFList * fn, VCFList * fp, FASTA * reference, bool require_allele_match, bool require_zygosity_match) {
        positional_allele_match( a, tp, fn, fp, require_allele_match, require_zygosity_match);
        // add haplotype, block, hp-error matchings
    }

//-----------------------------------------------------

    void set_positional_info_from_bam(const char * fname) {
        if (strlen(fname) == 0) {
            return;
        }
        tmpstruct_t tmp;

        tmp.beg = 0;
        tmp.end = 0x7fffffff;
        tmp.in = samopen(fname, "rb", 0);
        if (tmp.in == 0) {
            fprintf(stderr, "Fail to open BAM file %s\n", fname);
            return;
        }

        int ref;
        bam_index_t *idx=NULL;
        bam_plbuf_t *buf=NULL;
        idx = bam_index_load(fname); // load BAM index
        if (idx == 0) {
            fprintf(stderr, "BAM indexing file is not available.\n");
            return;
        }


        for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit)
            for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++)
                for(int i=0; i< posit->second.alt_count(); i++) {
                    depth = -1;
                    string pos_id = chrit->first + ":" + itos(posit->first)+ "-" + itos(posit->first);
                    bam_parse_region(tmp.in->header, pos_id.c_str(), &ref,
                                     &tmp.beg, &tmp.end); // parse the region
                    if (ref < 0) {
                        cerr <<"Invalid region " << pos_id << endl;
                        return;
                    }
                    buf = bam_plbuf_init(pileup_func, &tmp); // initialize pileup
                    bam_fetch(tmp.in->x.bam, idx, ref, tmp.beg, tmp.end, buf, fetch_func);
                    bam_plbuf_push(0, buf); // finalize pileup
                    posit->second.set_dp(depth);
                }
        bam_index_destroy(idx);
        bam_plbuf_destroy(buf);

        samclose(tmp.in);
    }

//-----------------------------------------------------

    void set_dp_to_zero() {

        for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit)
            for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++)
                posit->second.set_dp(0);
    }

};

//-----------------------------------------------------------
void print_stats(VCFList * target, VCFList * tp, VCFList * fp, VCFList * fn, string out_dir,
                 string out_pref, string out_format, string json_file = "", string DP = "",
                 long TARGET_SIZE = -1, string full_version_string="0.0")
{
    if(tp==NULL) return;

    std::map<int, int> map_fp_counts;
    for (int index = 0; (index < 100); ++index) {map_fp_counts[index] = 0;}
    if (fp != NULL) {
        for (std::map<string,std::map<long int,VCFinfo> >::iterator iter1 = fp->getList()->begin(); (iter1 != fp->getList()->end()); ++iter1) {
            for (std::map<long int,VCFinfo>::iterator iter2 = iter1->second.begin(); (iter2 != iter1->second.end()); ++iter2) {
                if (iter2->second.get_target_end_pos() < 100) {
                    map_fp_counts[iter2->second.get_target_end_pos()]++;
                }
            }
        }
    }
    std::map<int, int> map_fn_counts;
    for (int index = 0; (index < 100); ++index) {map_fn_counts[index] = 0;}
    if (fn != NULL) {
        for (std::map<string,std::map<long int,VCFinfo> >::iterator iter1 = fn->getList()->begin(); (iter1 != fn->getList()->end()); ++iter1) {
            for (std::map<long int,VCFinfo>::iterator iter2 = iter1->second.begin(); (iter2 != iter1->second.end()); ++iter2) {
                if (iter2->second.get_target_end_pos() < 100) {
                    map_fn_counts[iter2->second.get_target_end_pos()]++;
                }
            }
        }
    }
    ofstream fout;
    fout.open((out_dir + "/distance_from_end.txt").c_str());
	fout << "#distance_from_target_end\tfp\tfn" << endl;
	for (int index = 0; (index < 100); ++index) {fout << (index * -1) << "\t" << map_fp_counts[index] << "\t" << map_fn_counts[index] << endl;}
	fout.close();

    bool text = (strcmp(out_format.c_str(),"text") == 0);
    bool html = (strcmp(out_format.c_str(),"html") == 0);

    vector<int> cov;
    cov.push_back(0);

    if(!DP.empty()) {

        std::stringstream ss(DP);
        string tmp_value;
        while(ss.good()) {
            getline( ss, tmp_value, ',');
            try {
                int c = atoi(tmp_value.c_str());
                if(c>0) cov.push_back(c);
            } catch(...) {}
        }
    }

    ofstream output;
    bool outfile = (json_file!="");
    if(outfile) {
        output.open(json_file.c_str());
        output << "{" << endl;
    }

    for(unsigned int j=0; j<cov.size(); j++) {

        cout << "Cov>=" << cov[j] <<"x:SNP,DEL,INS,MNP,SNP+MNP,INDEL,ALL" << (html?"<BR>\n":"\n");

        int * counts_tp = (j==0) ? tp->get_var_counts(out_pref+"_tp.vcf") : tp->get_var_counts("",cov[j]);
        if(j==0) {
            if(html)  cout << "<a href="<<  "_tp.vcf" << ">TP:</a>";
            else   cout << "("<< out_pref << "_tp.vcf" << ")TP:";
        } else cout << "TP:";

        for(int i=0; i<4; i++) cout <<  counts_tp[i] << ",";
        cout << (counts_tp[0] + counts_tp[3]) << ",";
        cout << (counts_tp[1] + counts_tp[2]) << ",";
        cout << (counts_tp[1] + counts_tp[0] + counts_tp[2] + counts_tp[3] ) << (html?"<BR>\n":"\n");
        if(outfile) {
            output << "\"TP-SNP-" << cov[j] << "\":"<< counts_tp[0] << "," << endl;
            output << "\"TP-DEL-" << cov[j] << "\":"<< counts_tp[1] << "," <<  endl;
            output << "\"TP-INS-" << cov[j] << "\":"<< counts_tp[2] << "," <<  endl;
            output << "\"TP-MNP-" << cov[j] << "\":"<< counts_tp[3] << "," <<  endl;
            output << "\"TP-SNPMNP-" << cov[j] << "\":"<< (counts_tp[0] + counts_tp[3]) << "," <<  endl;
            output << "\"TP-INDEL-" << cov[j] << "\":"<< (counts_tp[1] + counts_tp[2]) << "," <<  endl;
            output << "\"TP-ALL-" << cov[j] << "\":"<< (counts_tp[1] + counts_tp[0] + counts_tp[2] + counts_tp[3]) << "," <<  endl;
        }

        if( fn != NULL && fp != NULL) {

            int * counts_fn = (j==0) ? fn->get_var_counts(out_pref+"_fn.vcf") : fn->get_var_counts("",cov[j]);

            if(j==0) {
                if(html)  cout << "<a href="<<  "_fn.vcf" << ">FN:</a>";
                else   cout << "("<< out_pref << "_fn.vcf" << ")FN:";
            } else cout << "FN:";

            for(int i=0; i<4; i++) cout << counts_fn[i] << ",";
            cout << (counts_fn[0] + counts_fn[3]) << ",";
            cout << (counts_fn[1] + counts_fn[2]) << ",";
            cout << (counts_fn[1] + counts_fn[0] + counts_fn[2] + counts_fn[3] ) << (html?"<BR>\n":"\n");

            if(outfile) {
                output << "\"FN-SNP-" << cov[j] << "\":"<< counts_fn[0] << "," <<  endl;
                output << "\"FN-DEL-" << cov[j] << "\":"<< counts_fn[1] << "," <<  endl;
                output << "\"FN-INS-" << cov[j] << "\":"<< counts_fn[2] << "," <<  endl;
                output << "\"FN-MNP-" << cov[j] << "\":"<< counts_fn[3] << "," <<  endl;
                output << "\"FN-SNPMNP-" << cov[j] << "\":"<< (counts_fn[0] + counts_fn[3]) << "," <<  endl;
                output << "\"FN-INDEL-" << cov[j] << "\":"<< (counts_fn[1] + counts_fn[2]) << "," <<  endl;
                output << "\"FN-ALL-" << cov[j] << "\":"<< (counts_fn[1] + counts_fn[0] + counts_fn[2] + counts_fn[3]) << "," << endl;
            }


            int * counts_fp = (j==0) ? fp->get_var_counts(out_pref+"_fp.vcf") : fp->get_var_counts("",cov[j]);
            if(j==0) {
                if(html)  cout << "<a href="<< "_fp.vcf" << ">FP:</a>";
                else   cout << "("<< out_pref << "_fp.vcf" << ")FP:";
            } else cout << "FP:";

            for(int i=0; i<4; i++) cout << counts_fp[i] << ",";
            cout << (counts_fp[0] + counts_fp[3]) << ",";
            cout << (counts_fp[1] + counts_fp[2]) << ",";
            cout << (counts_fp[1] + counts_fp[0] + counts_fp[2] + counts_fp[3] ) << (html?"<BR>\n":"\n");

            if(outfile) {
                output << "\"FP-SNP-" << cov[j] << "\":"<< counts_fp[0] << "," <<  endl;
                output << "\"FP-DEL-" << cov[j] << "\":"<< counts_fp[1] << "," <<  endl;
                output << "\"FP-INS-" << cov[j] << "\":"<< counts_fp[2] << "," <<  endl;
                output << "\"FP-MNP-" << cov[j] << "\":"<< counts_fp[3] << "," <<  endl;
                output << "\"FP-SNPMNP-" << cov[j] << "\":"<< (counts_fp[0] + counts_fp[3]) << "," <<  endl;
                output << "\"FP-INDEL-" << cov[j] << "\":"<< (counts_fp[1] + counts_fp[2]) << "," <<  endl;
                output << "\"FP-ALL-" << cov[j] << "\":"<< (counts_fp[1] + counts_fp[0] + counts_fp[2] + counts_fp[3]) << "," <<  endl;
            }
            int target_size = 0;
            if (target != NULL) {
                target_size = ((j==0) ? target->calculate_target_size(0) : target->calculate_target_size(cov[j]));
            }
            float snp_sens = (10000*(counts_tp[0] + counts_tp[3])/(counts_tp[0] + counts_tp[3] + counts_fn[0] + counts_fn[3]+0.0001))/100.00;
            float indel_sens = (10000*(counts_tp[1] + counts_tp[2])/(counts_tp[1] + counts_tp[2] + counts_fn[1] + counts_fn[2]+0.0001))/100.00;
            float total_sens = (10000*(counts_tp[0] + counts_tp[3]+ counts_tp[1] + counts_tp[2])/(counts_tp[0] + counts_tp[3] + counts_tp[1] + counts_tp[2] + counts_fn[0] + counts_fn[3] + counts_fn[1] + counts_fn[2]+0.0001))/100.00;
            float snp_ppv = (10000*(counts_tp[0] + counts_tp[3])/(counts_tp[0] + counts_tp[3] + counts_fp[0] + counts_fp[3]+0.0001))/100.00;
            float indel_ppv = (10000*(counts_tp[1] + counts_tp[2])/(counts_tp[1] + counts_tp[2] + counts_fp[1] + counts_fp[2]+0.0001))/100.00;
            float total_ppv = (10000*(counts_tp[0] + counts_tp[3]+ counts_tp[1] + counts_tp[2])/(counts_tp[0] + counts_tp[3] + counts_tp[1] + counts_tp[2] + counts_fp[0] + counts_fp[3] + counts_fp[1] + counts_fp[2]+0.0001))/100.00;
            float snp_fpr = ((1.0e+8)*(counts_fp[0] + counts_fp[3])/(target_size+0.0001))/100.00;
            float indel_fpr = ((1.0e+8)*(counts_fp[1] + counts_fp[2])/(target_size+0.0001))/100.00;
            float total_fpr = ((1.0e+8)*(counts_fp[0] + counts_fp[3]+counts_fp[1] + counts_fp[2])/(target_size+0.0001))/100.00;


            cout << "SENS%:"<<"-,-,-,-,-," << snp_sens << "," << indel_sens << "," << total_sens << (html?"<BR>\n":"\n");
            cout << "PPV%:"<<"-,-,-,-,-," << snp_ppv << "," << indel_ppv << "," << total_ppv << (html?"<BR>\n":"\n");
            if (target_size>0) cout << "FP/Mb:"<<"-,-,-,-,-," << snp_fpr << "," << indel_fpr << "," << total_fpr << (html?"<BR>\n":"\n");
            if ((target != NULL) and (cov.size() > 1)) {
                cout << "COVERAGE:" << ((j==0) ? target->calculate_target_size(0) : target->calculate_target_size(cov[j])) << (html?"<BR>\n":"\n");
            }

            if(outfile) {
                output << "\"SENS-SNPMNP" << cov[j] << "\":"<< snp_sens << "," <<  endl;
                output << "\"SENS-INDEL" << cov[j] << "\":"<< indel_sens << "," <<  endl;
                output << "\"SENS-ALL" << cov[j] << "\":"<< total_sens << "," <<  endl;
                output << "\"PPV-SNPMNP" << cov[j] << "\":"<< snp_ppv << "," <<  endl;
                output << "\"PPV-INDEL" << cov[j] << "\":"<< indel_ppv << "," <<  endl;
                output << "\"PPV-ALL" << cov[j] << "\":"<< total_ppv << "," <<  endl;
                if(TARGET_SIZE>0) {
                    output << "\"FPR-SNPMNP" << cov[j] << "\":"<< snp_fpr << "," <<  endl;
                    output << "\"FPR-INDEL" << cov[j] << "\":"<< indel_fpr << "," <<  endl;
                    output << "\"FPR-ALL" << cov[j] << "\":"<< total_fpr << "," <<  endl;
                }
            }

            delete counts_fp;
            delete counts_fn;
        }

        delete counts_tp;
    }

    if(outfile) {
      output << "\"vcfcomp-version\":"<< "\"" << full_version_string << "\""<< "\n}" << endl;
      output.close();
    }

}

void optimizer(VCFList * _tp,VCFList * _tp_filter,VCFList * _fp, VCFList * _fp_filter, string FLAG="QUAL", float _min = 15, float _step = 5, int nbins = 10, string FLAG2="", char cmp_operator='<', float _val = 0) {

// RBI - filter_unusual_predictions
// MLLD - data_qual_string
// STB - xx_strand_bias
// STBP - xx_strand_bias_pval

    /*
    ofstream output;
    bool outfile = (basename!="");
    if(outfile)output.open(basename.c_str());
    if(!output) { cerr << "Error writing output file: " << basename << endl; }
    */

    int ** tp = FLAG2.empty() ? _tp->get_var_counts(FLAG, _min, _step, nbins): _tp->get_var_counts(FLAG, _min, _step, nbins, FLAG2, cmp_operator, _val);
    int ** fn = FLAG2.empty() ? _tp_filter->get_var_counts(FLAG, _min, _step, nbins) : _tp_filter->get_var_counts(FLAG, _min, _step, nbins, FLAG2, cmp_operator, _val);
    int ** fp = FLAG2.empty() ? _fp->get_var_counts(FLAG, _min, _step, nbins) : _fp->get_var_counts(FLAG, _min, _step, nbins, FLAG2, cmp_operator, _val);
    int ** tn = FLAG2.empty() ? _fp_filter->get_var_counts(FLAG, _min, _step, nbins) : _fp_filter->get_var_counts(FLAG, _min, _step, nbins, FLAG2, cmp_operator, _val);

    //if(outfile) {
    if(!FLAG2.empty()) cout << "fixing " << FLAG2 << (char)cmp_operator << _val << endl;
    cout << "thresholding " << FLAG << " min="<< _min << " step=" << _step << " nbins=" << nbins << endl;

    cout << "FLAG" << "\t" << FLAG << "\t"  << "true-snp" << "\t" << "true-indel" << "\t" << "fp-snp" << "\t" << "fp-indel" << "\t" << "tn-snp" << "\t" << "tn-indel"  << endl;
    int i=0;
    cout << FLAG << "<<\t" << _min << "\t" << (tp[0][i] + fn[0][i]) << "\t" << (tp[1][i] + fn[1][i]) << "\t" << fp[0][i] << "\t" << fp[1][i] << "\t" << tn[0][i] << "\t" << tn[1][i]  << endl;
    for(i=1; i<=nbins+1; i++) cout << FLAG << ">=\t" << (((float)(i-1)*_step + _min)) << "\t" << (tp[0][i] + fn[0][i]) << "\t" << (tp[1][i] + fn[1][i]) << "\t" << fp[0][i] << "\t" << fp[1][i] << "\t" << tn[0][i] << "\t" << tn[1][i]  << endl;
    cout << FLAG << "\t" <<  "N/A\t" << (tp[0][i] + fn[0][i]) << "\t" << (tp[1][i] + fn[1][i]) << "\t" << fp[0][i] << "\t" << fp[1][i] << "\t" << tn[0][i] << "\t" << tn[1][i]  << endl;

    //output.close();
    //}
}

//--------------------------------------------------
void print_usage(int ext_code, string full_version_string) {
    cerr << "*****************************************************************" << endl;
    cerr << "VCF comparison tool v." << full_version_string                     << endl;
    cerr << "Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved"  << endl;
    cerr << "The tool generates variant matching report in form of TP,FP,FN  "  << endl;
    cerr << "printed to stdout, and creates corresponding files, e.g., _tp.vcf" << endl;
    cerr << "column 3 and 6 in the output files are for var type and coverage." << endl;
    cerr <<"******************************************************************" << endl;
    cerr << "Usage: vcfcomp --main-vcf filename [--option value]" << endl << endl;
    cerr << "[Options]:" << endl;
    cerr << "--main-vcf   filename     this option is required. filename points to a file with variants in VCF format (requires at minimum first 5 VCF columns,"  << endl;
    cerr << "                          # columns definitions are not required). if record is missing GT value, then all alternative alleles at that position are" << endl;
    cerr << "                          evaluated disregarding zygosity. If record contains INFO field then key=value pairs are extracted for custom reports. If"<< endl;
    cerr << "                          other optional vcf files are not provided, then metrics related to this file only, as well intersection with target(s) are" << endl;
    cerr << "                          are reported in TP category in (_tp.vcf)."     << endl;
    cerr << "--truth-vcf  filename     filename with variants in VCF format (requires at minimum first 5 VCF columns). if genotype GT values are missing, then"   << endl;
    cerr << "                          all alternative alleles at the same position are evaluated disregarding zygosity. variants from main-vcf are compared to"  << endl;
    cerr << "                          variants in truth-vcf. matching variants are reported as TP in (_tp.vcf). variants in main-vcf, but not in truth-vcf are"  << endl;
    cerr << "                          reported as FP in (_fp.vcf). variants in truth-vcf, but not in main-vcf are reported as FN in (_fn.vcf)." << endl;
    cerr << "--bg-vcf     filename     filename with variants in VCF format (requires at minimum first 5 VCF columns). if genotype GT values are missing, then"   << endl;
    cerr << "                          all alternative alleles at the same position are evaluated. this file is intended for exclusion of known back-ground variants"<< endl;
    cerr << "                          or systematic errors from variant matching step. If trut-vcf is not provided then bg-vcf is used instead."  << endl;
    cerr << "--filter-vcf filename     filename with variants in VCF format (requires at minimum first 5 VCF columns). if genotype GT values are missing, then"   << endl;
    cerr << "                          all alternative alleles at the same position are evaluated. it should include candidate variants filtered by variant caller."<< endl;
    cerr << "                          it is in parameter optimization report."  << endl;
    cerr << "--target     bedfile      bedfile is a file with a set of target regions in BED format. limit analysis to variants with anchor base in the target region." << endl;
    cerr << "                          In case of DEL or block-substitution entier block should be inside the target. this option can be used multiple times. vcf files" << endl;
    cerr << "                          are normalized then intersected with all provided target files." << endl;
    cerr << "--mt,--mb    [a,z,p]      variant matching criteria in addition to genomic position when comparing main-vcf with truth-vcf (--mt), and bg-vcf (--mb)." << endl;
    cerr << "                          value (a) requires alternative allele match, (z) requires allele + zygosity match, (p) position only. Default '--mt a', and '--mb p'." << endl;
    cerr << "--opref      prefix       preffix is appended to each output file, i.e., prefix_tp.vcf, prefix_fp.vcf, prefix_fn.vcf." << endl;
    cerr << "--odir       path         output directory. default value '.'" << endl;
    cerr << "--oformat    format       format can be text or html. analysis numbers are printed to standard output in convenient representation form." << endl;
    cerr << "--ojson      filename     outputs stats into the filename in a json KEY:VALUE format"<< endl;
    cerr << "--main-bam   filename     filename of the indexed BAM file. using this option may significantly impact the runtime. this file is used to recalculate" << endl;
    cerr << "                          certain metrics for variant positions in case they are requested by analysis, but not provided in INFO field of main-vcf,"  << endl;
    cerr << "                          or values from INFO field require to be recalculated (e.g., depth of coverage metric 'DP')." << endl;
    cerr << "--min_mapping_qv          The minimum mapping quality to use in calculating coverage." << endl;
    cerr << "--depth      cov_list     additional coverage depth thresholds used in the report. multiple thresholds should be comma separated. for TP,FP reports" << endl;
    cerr << "                          DP is extracted from main-vcf INFO field if available. for FN reports DP is recalulated from the main-bam file if provided." << endl;
    cerr << "                          missing DP equals to DP=0. example: '-dp 20,30,50'. analysis for coverage >=0 is always reported and stored in the output files." << endl;
    cerr << "--fasta      reference    filename with the reference sequence in FASTA format. If provided, then variant maching is done by normalizing variants to" << endl;
    cerr << "                          the reference. that enables matching of differently represented equivalent variants. usually variants representation is already" << endl;
    cerr << "                          reference normalized, thus this option may not affect the results, however if used it will significantly increase the runtime." << endl;
    cerr << "--ignore-gt [m,t,b,f]     ignore GT value when it is specified in (m)ain-vcf, (t)ruth-vcf, (b)g-vcf, (f)ilter-vcf. all listed alleles are loaded, zygosity" << endl;
    cerr << "                          is ignored." << endl;
    cerr << "                          intended to include into comparison alleles with empty ./. or reference 0/0 GT values. example: --ignore-gt mt." << endl;
    cerr << "--mnp       [split,keep]  splits MNPs into single SNPs or keeps them unsplit. default '--mnp split'." << endl;
    cerr << "--optimze   expression    use only with --filter-vcf. groups variants into bins defined by expression \"FLAG1,min,step,nbins:FLAG2,<|>,value\". FLAG2 is optional." << endl;
    cerr << "                          variants are grouped into nbins+2 where first and last bins are for values with FLAG <min or >=(min+step*nbins). FLAG value is extrated" << endl;
    cerr << "                          from vcf records, and it goes into N/A bin if missing. Most common optimization FLAGS are QUAL, MLLD, RBI, VARB, STB, STBP, FSAF, FSAR." << endl;
    cerr << "                          FLAG2 is used to provide a single value. Example: \"MLLD,5,1,10:QUAL,>,15\" or \"VARB,0.1,0.01,20\" . Outputs table to std output. Multiple" << endl;
    cerr << "                          expressions can be provided by separating them with ';', e.g. \"MLLD,5,1,10:QUAL,>,15;VARB,0.1,0.01,20\" ." << endl;
    cerr << "in progress:              combining vars, haplotype sequence match, approx var match up to hp-error, handling block-substitutions" << endl;
    cerr << endl;

    exit(ext_code);
}
//--------------------------------------------------
int main(int argc, char *argv[]) {

    string full_version_string = IonVersion::GetVersion() + "." +IonVersion::GetRelease() +
                               " (" + IonVersion::GetGitHash() + ") (" + IonVersion::GetBuildNum() + ")";

// Set default values
    string bg_vcf_file = "";
    string truth_vcf_file = "";
    string main_vcf_file = "";
    string filter_vcf_file = "";
    string ref_file  = "";
    string out_pref = "";
    string out_dir = ".";
    string out_format = "text"; // {text,html,json,cvs}
    string DP = "";
    string main_bam_file = "";
    string ignore_genotype = "f";
    string bg_match_criteria = "";
    string truth_match_criteria = "a";
    string optimize = "";
    string map_quality = "4";

    vector<string> targets;
    bool split_mnp = true;
    string json_file ="";

// Read input options
    int acnt = 1;
    while(acnt<argc) {
        if(strcmp(argv[acnt], "--bg-vcf")==0 )    {
            bg_vcf_file = (++acnt) < argc ? argv[acnt] : bg_vcf_file;
        } else if(strcmp(argv[acnt], "--truth-vcf")==0 ) {
            truth_vcf_file = (++acnt) < argc ? argv[acnt] :  truth_vcf_file;
        } else if(strcmp(argv[acnt], "--main-vcf")==0 )  {
            main_vcf_file = (++acnt) < argc ? argv[acnt] : main_vcf_file;
        } else if(strcmp(argv[acnt], "--filter-vcf")==0 ) {
            filter_vcf_file = (++acnt) < argc ? argv[acnt] : filter_vcf_file;
        } else if(strcmp(argv[acnt], "--target")==0 )    {
            if(++acnt < argc) targets.push_back(argv[acnt]);
        } else if(strcmp(argv[acnt], "--opref")==0 )     {
            out_pref = (++acnt) < argc ? argv[acnt] : out_pref;
        } else if(strcmp(argv[acnt], "--odir")==0 )      {
            out_dir = (++acnt) < argc ? argv[acnt] : out_dir;
        } else if(strcmp(argv[acnt], "--optimize")==0 )  {
            optimize = (++acnt) < argc ? argv[acnt] : optimize;
        } else if(strcmp(argv[acnt], "--oformat")==0 )   {
            out_format = (++acnt) < argc ? argv[acnt] : out_format;
        } else if(strcmp(argv[acnt], "--main-bam")==0 )  {
            main_bam_file = (++acnt) < argc ? argv[acnt] : main_bam_file;
        } else if(strcmp(argv[acnt], "--min-mapping-qv")==0 )     {
            map_quality = (++acnt) < argc ? argv[acnt] : map_quality;
        } else if(strcmp(argv[acnt], "--depth")==0 )     {
            DP = (++acnt) < argc ? argv[acnt] : DP;
        } else if(strcmp(argv[acnt], "--ignore-gt")==0 ) {
            ignore_genotype = (++acnt) < argc ? argv[acnt] : ignore_genotype;
        } else if(strcmp(argv[acnt], "--mt")==0 )        {
            truth_match_criteria = (++acnt) < argc ? argv[acnt] : truth_match_criteria;
        } else if(strcmp(argv[acnt], "--mb")==0 )        {
            bg_match_criteria = (++acnt) < argc ? argv[acnt] : bg_match_criteria;
        } else if(strcmp(argv[acnt], "--fasta")==0 )     {
            ref_file = (++acnt) < argc ? argv[acnt] : ref_file;
        } else if(strcmp(argv[acnt], "--mnp")==0 )       {
            if(++acnt < argc && strcmp(argv[acnt], "keep") == 0 ) split_mnp = false;
        } else if(strcmp(argv[acnt], "--ojson")==0 )     {
            json_file = (++acnt) < argc ? argv[acnt] : json_file;
        } else {
            cerr << endl << "\nUnknown parameter " << argv[acnt] << endl;
            print_usage(1, full_version_string);
        }
        ++acnt;
    }

    if (!map_quality.empty()) {
        min_mapping_qv = atoi(map_quality.c_str());
    }

    out_pref = out_dir + "/" + out_pref;

    cerr << "vcfcomp version:"<< full_version_string << endl;

    VCFList * varBg = NULL, * varTruth = NULL, * varVC = NULL, * varFilter = NULL;
    FASTA * reference = NULL;

    if(!ref_file.empty()) {
        cerr << "\nloading reference ... from " << ref_file << endl;
        reference = new FASTA();
        int ncnt = reference->load_file(ref_file);
        cerr << "loaded " << ncnt << " reference contigs." << endl;
    }

    if(reference==NULL || reference->size()==0) {
        cerr << "\nMissing reference data." << endl;
        cerr << "Reference normalization is not performed." <<endl;
    }


    if(main_vcf_file.empty()) {
        cerr << "\nMissing main vcf file name!" <<endl;
        print_usage(1, full_version_string);
    } else {
        cerr << "\nLoading main vcf variants: " << main_vcf_file << endl;
        varVC = new VCFList();
        varVC->load_file(main_vcf_file, reference, split_mnp, !ignore_genotype.empty() && ignore_genotype.find("m")!=string::npos);
    }

    if(!truth_vcf_file.empty()) {
        cerr << "\nLoading truth variants: " << truth_vcf_file << endl;
        varTruth = new VCFList();
        varTruth->load_file(truth_vcf_file, reference, split_mnp, !ignore_genotype.empty() && ignore_genotype.find("t")!=string::npos);
    }


    if(!bg_vcf_file.empty()) {
        cerr << "\nLoading background variants: " << bg_vcf_file << endl;
        varBg = new VCFList();
        varBg->load_file(bg_vcf_file, reference, split_mnp, !ignore_genotype.empty() && ignore_genotype.find("b")!=string::npos);
    }

    if(!filter_vcf_file.empty()) {
        cerr << "\nLoading filtered-candidate variants: " << filter_vcf_file << endl;
        varFilter = new VCFList();
        varFilter->load_file(filter_vcf_file, reference, split_mnp, !ignore_genotype.empty() && ignore_genotype.find("f")!=string::npos);
    }

    long TARGET_SIZE = -1;

    VCFList * target = NULL;
    if(targets.size()>0) { // intersect variant list with target regions

        for(unsigned int i=0; i<targets.size(); i++) {
            cerr << "\nloading target file ... " << targets.at(i) << endl;
            VCFList * tmp_target = new VCFList();
            VCFList * merged_target = new VCFList();
            tmp_target->load_bed_file(targets.at(i));

            //merging target is required for correct target size calculation and intersection of multiple BEDs
            //get_target_vcf supports BEDs with overlapping contigs

            cerr << "merging overlapping targets ... " << targets.at(i) << endl;
            TARGET_SIZE = tmp_target->merge_overlapping_segments(merged_target);
            cerr << "merged size: "<< TARGET_SIZE << endl;

            delete tmp_target;
            if(i==0) target = merged_target;
            else {
                tmp_target = new VCFList();
                cerr << "intersect previous target with " << targets.at(i) << endl;
                TARGET_SIZE = target->intersect(merged_target,tmp_target);
                cerr << "resulting target size: " << TARGET_SIZE << endl;
                delete target;
                delete merged_target;
                target = tmp_target;
            }
        }
        target->set_positional_info_from_bam(main_bam_file.c_str());

        cerr << "extract on-target vcf records ... " << endl;
        if(varVC!=NULL) {
            VCFList * _varVC = new VCFList();
            long vcrecords = target->get_target_vcf(varVC, _varVC);
            cerr << "main-vcf records on target .. " << vcrecords << endl;
            delete varVC;
            varVC = _varVC;
        }
        if(varTruth!=NULL) {
            VCFList * _varTruth = new VCFList();
            long trrecords = target->get_target_vcf(varTruth, _varTruth);
            cerr << "truth-vcf records on target .. " << trrecords << endl;
            delete varTruth;
            varTruth = _varTruth;
        }
        if(varBg!=NULL) {
            VCFList * _varBg = new VCFList();
            long bgrecords = target->get_target_vcf(varBg, _varBg);
            cerr << "bg-vcf records on target .. " << bgrecords << endl;
            delete varBg;
            varBg = _varBg;
        }

        if(varFilter!=NULL) {
            VCFList * _varFilter = new VCFList();
            long bgrecords = target->get_target_vcf(varFilter, _varFilter);
            cerr << "filter-vcf records on target .. " << bgrecords << endl;
            delete varFilter;
            varFilter = _varFilter;
        }

        //if(target != NULL) delete target;

    } // finished target intersection

	// remove Bg from Truth

    bool zygosity_match = !bg_match_criteria.empty() && bg_match_criteria.find("z")!=string::npos;
    bool allele_match = zygosity_match || (!bg_match_criteria.empty() && bg_match_criteria.find("a")!=string::npos);
	//if ((zygosity_match) and (truth_match_criteria.find("p")!=string::npos)) {zygosity_match = false;}
	cerr << "zygosity_match = " << zygosity_match << endl;
	cerr << "allele_match = " << allele_match << endl;
	varTruth->remove(varBg, zygosity_match, allele_match);
	
    VCFList *tp = NULL, *fp = NULL, *fn = NULL;

    if(!bg_vcf_file.empty()) {
        bool require_zygosity_match = !bg_match_criteria.empty() && bg_match_criteria.find("z")!=string::npos;
        bool require_allele_match = require_zygosity_match || (!bg_match_criteria.empty() && bg_match_criteria.find("a")!=string::npos);

        tp = new VCFList();
        fp = new VCFList();
        fn = new VCFList();
        varBg->match(varVC, tp, fn, fp, reference, require_allele_match, require_zygosity_match);
    }

    if(!truth_vcf_file.empty()) {
        bool require_zygosity_match = !truth_match_criteria.empty() && truth_match_criteria.find("z")!=string::npos;
        bool require_allele_match = require_zygosity_match || (!truth_match_criteria.empty() && truth_match_criteria.find("a")!=string::npos);
        if(fp!=NULL) {
            if(varVC!=NULL) delete varVC;
            varVC = new VCFList(*fp);
			delete fp;
        }
        if(tp!=NULL) delete tp;
        if(fn!=NULL) delete fn;
        tp = new VCFList();
        fn = new VCFList();
        fp = new VCFList();

        varTruth->match(varVC, tp, fn, fp, reference, require_allele_match, require_zygosity_match);
    }

    if(!main_bam_file.empty()) {
        cerr << "loading positional info from BAM file " << main_bam_file << endl;
        fn->set_positional_info_from_bam(main_bam_file.c_str());
    } else if(fn!=NULL) fn->set_dp_to_zero();

    bool html = (strcmp(out_format.c_str(),"html") == 0);

    cerr << endl;
    cout << "vcfcomp version:"<< full_version_string << (html?"<BR>\n":"\n");
    if(TARGET_SIZE>0) cout << "Final Target Size (bp):"<< TARGET_SIZE << (html?"<BR>\n":"\n");
    cout << "\nReport for main-vcf\n**********************************************************" << endl;
    print_stats(target, tp == NULL ? varVC : tp , fp, fn, out_dir, out_pref, out_format, json_file, DP, TARGET_SIZE, full_version_string );

    // inspecting filtered file

    VCFList *tp_filter = NULL, *fp_filter = NULL, *fn_filter = NULL;

    if(varFilter!=NULL) {

        if(!bg_vcf_file.empty()) {
            bool require_zygosity_match = !bg_match_criteria.empty() && bg_match_criteria.find("z")!=string::npos;
            bool require_allele_match = require_zygosity_match || (!bg_match_criteria.empty() && bg_match_criteria.find("a")!=string::npos);

            tp_filter = new VCFList();
            fp_filter = new VCFList();
            fn_filter = new VCFList();
            varBg->match(varFilter, tp_filter, fn_filter, fp_filter, reference, require_allele_match, require_zygosity_match);
        }

        if(!truth_vcf_file.empty()) {
            bool require_zygosity_match = !truth_match_criteria.empty() && truth_match_criteria.find("z")!=string::npos;
            bool require_allele_match = require_zygosity_match || (!truth_match_criteria.empty() && truth_match_criteria.find("a")!=string::npos);

            if(fp_filter!=NULL) {
                if(varFilter!=NULL) delete varFilter;
                varFilter = fp_filter;
            }
            if(tp_filter!=NULL) delete tp_filter;
            if(fn_filter!=NULL) delete fn_filter;
            tp_filter = new VCFList();
            fn_filter = new VCFList();
            fp_filter = new VCFList();

            varTruth->match(varFilter, tp_filter, fn_filter, fp_filter, reference, require_allele_match, require_zygosity_match);
        }

        if(!main_bam_file.empty()) {
            cerr << "loading positional info from BAM file " << main_bam_file << endl;
            fn_filter->set_positional_info_from_bam(main_bam_file.c_str());
        } else if(fn_filter!=NULL) fn_filter->set_dp_to_zero();

        cout << "\nReport for filter-vcf\n**********************************************************" << endl;
        print_stats(target, tp_filter == NULL ? varFilter : tp_filter , fp_filter, fn_filter, out_dir, out_pref + "_filtered", out_format, "", DP, TARGET_SIZE, full_version_string );

    }

    if(target != NULL) delete target;
    if(varBg!=NULL) delete varBg;
    if(varTruth!=NULL) delete varTruth;
    if(varVC!=NULL) delete varVC;
    if(varFilter!=NULL) delete varFilter;
    if(fn!=NULL) delete fn;
    if(fn_filter!=NULL) delete fn_filter;


    if(tp_filter!=NULL && !optimize.empty()) {
        cout << "\nRunning parameter optimization\n**********************************************************" << endl;
        cerr << "Parsing info values..." << endl;
        tp->parse_info_values();
        tp_filter->parse_info_values();
        fp->parse_info_values();
        fp_filter->parse_info_values();
        cerr << "Optimizing ...." << endl;

        std::stringstream ss(optimize);
        string expres;
        while(ss.good()) {
            getline( ss, expres, ';');
            if(!expres.empty()) {
                std::stringstream aa(expres);
                string first = "", second = "";
                if(aa.good()) getline( aa, first, ':');
                if(aa.good()) getline( aa, second, ':');

                string FLAG="",FLAG2="";
                float _min = 15.0, _step = 5.0, _value = 0.0;
                int   _nbins = 0;
                char comp_operator='<';

                if(!first.empty()) {
                    std::stringstream bb(first);

                    if(bb.good()) try {
                            string tmp;
                            getline(bb, FLAG, ',');
                            getline(bb, tmp, ',');
                            _min = atof(tmp.c_str());
                            getline(bb, tmp, ',');
                            _step = atof(tmp.c_str());
                            getline(bb, tmp, ',');
                            _nbins = atoi(tmp.c_str());
                        } catch(...) {
                            FLAG="";
                        }
                }

                if(!second.empty()) {
                    std::stringstream bb(first);

                    if(bb.good()) try {
                            string tmp;
                            getline(bb, FLAG2, ',');
                            getline(bb, tmp, ',');
                            comp_operator = tmp[0]!='<'?'>':'<';
                            getline(bb, tmp, ',');
                            _value = atof(tmp.c_str());
                        } catch(...) {
                            FLAG2="";
                        }
                }

                if(!FLAG.empty()) {
                    if(!FLAG2.empty()) optimizer(tp,tp_filter,fp,fp_filter, FLAG, _min, _step, _nbins, FLAG2, comp_operator, _value);
                    else optimizer(tp,tp_filter,fp,fp_filter, FLAG, _min, _step, _nbins);
                }
            }
        }
    }

    return 0;
    // -- free-up memory, not necessary since program terminates here
    if(reference!=NULL) delete reference;
    if(tp!=NULL) delete tp;
    if(fp!=NULL) delete fp;
    if(tp_filter!=NULL) delete tp_filter;
    if(fp_filter!=NULL) delete fp_filter;



}

//-----------------------------------------------------
