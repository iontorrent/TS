/* Copyright (C) 2016 Thermo Fisher Scientific, All Rights Reserved */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h> 
#include <sstream>
#include <cstdlib> 
#include <stdio.h>
#include <vector>
#include <climits>
#include <map>
#include <cmath>
#include <algorithm>
#include "IonVersion.h"
#include "api/BamReader.h"
#include "api/BamWriter.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"

using namespace std;
using namespace BamTools;

//-----------------------------------------------------
// converts int to string
inline string  itos(int i){ stringstream s; s << i; return s.str();}

// converts int to string
inline string  ctos(char * c, int len){ stringstream s; for(int i=0;i<len;i++) s << (char)c[i]; return s.str();}
//-----------------------------------------------------

string vartypes[5]{"del","snp","ins","mnp","complex"};
string acgt="ACGT";
inline int vartype(const string ref, const string alt)
{
  if(ref.length() ==  alt.length())
   {
    if(ref.length() == 1) return 0; else return 3;
   }
   if(ref.length() <  alt.length()) return 2;
   else return 1;
}
// reverse complements base/color sequence
//-----------------------------------------------------
string reverse(string s){
	string x=s;
	for(int i=(int)s.length()-1;i>=0;i--)
		{
		if(s[i]=='.') x[s.length()-1-i] = '.';	else
		if(s[i]=='A') x[s.length()-1-i] = 'T';	else
		if(s[i]=='C') x[s.length()-1-i] = 'G';	else
		if(s[i]=='G') x[s.length()-1-i] = 'C';	else
		if(s[i]=='T') x[s.length()-1-i] = 'A';	else
		if(s[i]=='N') x[s.length()-1-i] = 'N';
		}
	return x;
}
//--------------------------------------------------------
struct PileUp
{
 private:
      std::map< string , int > * alt_set;
 public:	
      PileUp() { alt_set = new std::map<string, int>(); }
	 ~PileUp() { if(alt_set != NULL) delete alt_set; alt_set=NULL;}
 
 inline void add(string alt){
			std::map< string , int >::iterator alt_key = alt_set->find(alt);
			if( alt_key != alt_set->end() ) alt_key->second++;
			else alt_set->insert(std::pair<string, int>(alt, 1));
      }
	  
 inline int get_coverage(){
			long coverage = 0;
			for (std::map< string , int >::iterator alt_key = alt_set->begin(); alt_key != alt_set->end(); ++alt_key) if(alt_key->first[0]!='+' && alt_key->first[0]!='-') coverage += alt_key->second;
			return coverage;
			}			
 
 inline int get_plus_insert_coverage(){
			long coverage = 0;
			for (std::map< string , int >::iterator alt_key = alt_set->begin(); alt_key != alt_set->end(); ++alt_key) coverage += alt_key->second;
			return coverage;
			}				
			
 inline int get_allele_read_counts(string allele){
			std::map< string , int >::iterator alt_key = alt_set->find(allele);
			if( alt_key != alt_set->end() ) return alt_key->second;
			return 0;
			}			

string consensus_call(float min_allele_freq, int & alt_counts)
{
  const double total_cov = get_coverage();
  const unsigned min_count = ceil(total_cov*min_allele_freq);
  for (std::map< string , int >::iterator alt_key = alt_set->begin(); alt_key != alt_set->end(); ++alt_key) if((unsigned int)alt_key->second >= min_count) {alt_counts=alt_key->second; return alt_key->first;}
  return "";
}	  

inline std::map< string , int > * get_alt_set(){return alt_set;}	

};
//--------------------------------------------------------
struct VariantFeatures
{
 private:
          std::map<string, string> * info_key_value;
 public:          
          string vc_record;
          char gt_index;

inline int get_int(string info_key){
            if(info_key_value!=NULL)
            {
              std::map< string , string >::iterator key_val = info_key_value->find(info_key);
              if( key_val != info_key_value->end() && key_val->second.length() >0 && isdigit((key_val->second)[0])) return atoi(&((key_val->second)[0])); //dangerous but fast, potential bug
            }
             return -1;  
            }
            
inline float get_float(string info_key){
            if(info_key_value!=NULL)
            {
              std::map< string , string >::iterator key_val = info_key_value->find(info_key);
              if( key_val != info_key_value->end() && key_val->second.length() >0 && (isdigit((key_val->second)[0]) || (key_val->second)[0] == '-' || (key_val->second)[0] == '+')) return atof(&((key_val->second)[0])); //dangerous but fast, potential bug
            }
             return -1;  
            }     

inline string get_str(string info_key){
            if(info_key_value!=NULL)
            {
              std::map< string , string >::iterator key_val = info_key_value->find(info_key);
              if( key_val != info_key_value->end() && key_val->second.length() >0) return key_val->second;
            }
             return "";  
            }			


 VariantFeatures(string _vc_record, char _index){vc_record = _vc_record; gt_index = _index; info_key_value=NULL;};
 ~VariantFeatures(){/*if(info_key_value != NULL) delete info_key_value; info_key_value = NULL;*/}

 void  parse_info_values(){

   int col_idx = 0;
   std::stringstream cols(vc_record);
   string vcf_col;
   while(cols.good() && getline( cols, vcf_col, '\t') && ++col_idx<6);
   if(col_idx == 6 && info_key_value==NULL)
   {
   
    info_key_value = new std::map<string, string>();
    info_key_value->insert(std::pair<string, string>("QUAL", vcf_col)); 
    
   while(cols.good() && getline( cols, vcf_col, '\t') && ++col_idx<8);
   if(col_idx == 8 && !vcf_col.empty())
   {
    std::stringstream features(vcf_col);
    string flag;
    while(features.good() && getline( features, flag, ';'))
    {
     string::size_type pos = flag.find("=",0);
     if(pos!=string::npos && pos < flag.length()-1)
      {
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
 
 void  parse_amplicon_names(){
   int col_idx = 0;
   int num_cols=0;
   std::stringstream cols(vc_record);
   string vcf_col;
   while(cols.good() && getline( cols, vcf_col, '\t'))
   {
   col_idx++;
   if(col_idx == 4)
   {
    if(info_key_value != NULL) delete info_key_value;
    info_key_value = new std::map<string, string>();
    info_key_value->insert(std::pair<string, string>("AMP_NAME", vcf_col)); 
   }
   }
   if(col_idx > 4 && !vcf_col.empty())
   {
    std::stringstream features(vcf_col);
    string flag;
    while(features.good() && getline( features, flag, ';'))
    {
     string::size_type pos = flag.find("=",0);
     if(pos!=string::npos && pos < flag.length()-1)
      {
       int val_idx = 0;
       string flag_name = flag.substr(0,pos);
       std::stringstream values(flag.substr(pos+1));
       string value;
       while(values.good() && getline( values, value, ',') && ++val_idx < (int)gt_index);
       if(val_idx == (int)gt_index || val_idx == 1) info_key_value->insert(std::pair<string, string>(flag_name, value)); // if value already exists, that means that it is repeated twice in the vcf record, we keep first copy
      }
    }
   }     
 };
 
};
//--------------------------------------------------------
class FASTA
{
  public:
  
  std::vector<string > name;
  std::vector<string > contig;
  
  int load_file(string fasta_file)
   {
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
  while (getline(infile, line)) 
  {
   if(line[0]=='>')
   {
    if(contig_name!="") 
    {
     name.push_back ( contig_name);		
     contig.push_back ( sequence);
     contigs_loaded++;
     cerr << contigs_loaded << " loaded " << contig_name << " length: " << sequence.length() << endl;
	 //if(contigs_loaded>1) return contigs_loaded;
    }
    contig_name = line.substr(1);
    sequence="";
   }
   else if(line[0]!='#'){
   sequence += line;
   }
   }
   
   if(contig_name!="") 
    {
      name.push_back ( contig_name);
      contig.push_back ( sequence);
     contigs_loaded++;
     cerr << contigs_loaded << " loaded " << contig_name << " length: " << sequence.length() << endl;
    }
  
  return contigs_loaded;
}


int size()
{
 return (int)contig.size();
}

int32_t get_contig_idx(string chr)
{
	for(unsigned int i=0;i<name.size(); i++)
	{
		if(name[i].compare(chr)==0) return i;
	}
cerr << "incorrect contig name in vcf file"	 << endl;
return -1;	
}

string get_ref_segment(string chr, long start_pos, long end_pos)
{
	int32_t idx = get_contig_idx(chr);
	if(start_pos<0 || start_pos>(long)contig[idx].length() || end_pos<0 || end_pos>(long)contig[idx].length() || start_pos>=end_pos)
	{
	 cerr << "Fatal error: target segment outside of reference: " << chr << ":" << start_pos << "-" << end_pos << endl; 
	 exit(2);
	}
	return contig[idx].substr(start_pos,end_pos-start_pos);
}

};
//--------------------------------------------------------
struct VCFinfo
{
private:
  vector<string> ref;
  vector<string> alt; 
  vector<VariantFeatures> info;
  int zyg;   // -1 = not available, 0 = HOM, 1 = HET/REF, 2 = HET/NON-REF
  int DP;    //  0 = zero or not available
  
public:  

 inline void set_dp(int dp) { DP= dp;}
 inline void set_zyg(int _zyg) { zyg = _zyg; }
 inline int  get_zyg()  {  return zyg; }
  
 void set_dp(string _info) 
{
 DP=0;
 if(!_info.empty())
 {
  string::size_type  pos = _info.find("DP=");
  if(pos!=string::npos && _info.length()>pos+3 && isdigit(_info[pos+3]))  DP = atoi(&(_info[pos+3])); // fast, but dangerous, assumes string is represented in continuous array, potential bug
 }  
} 
  
int get_var_type(int idx) //SNP=0,DEL=1,INS=2,MNP=3,OTHER=4
{
  if(idx>=(int)ref.size()) { cerr << "get_var_type(idx): requested idx is larger than allele counts" << endl; exit(1); }
  return vartype(ref.at(idx), alt.at(idx));
}

float get_int_flag_value(int idx,string FLAG)
{
  if(idx>=(int)info.size()) { cerr << "get_int_flag_value(idx): requested idx is larger than allele counts" << endl; exit(1); }
  return info.at(idx).get_float(FLAG); 
}

string get_str_flag_value(int idx,string FLAG)
{
  if(idx>=(int)info.size()) { cerr << "get_str_flag_value(idx): requested idx is larger than allele counts" << endl; exit(1); }
  return info.at(idx).get_str(FLAG); 
}

int alt_count() { return alt.size();}
string get_alt(int idx){ return alt.at(idx);}
string get_ref(int idx){ return ref.at(idx);}
string get_info(int idx){ return info.at(idx).vc_record;}
char get_gt_index(int idx){ return info.at(idx).gt_index;}


void  parse_info_values()
{
 for(unsigned int i=0;i<info.size();i++)
  info.at(i).parse_info_values();
}

void  parse_amplicon_names()
{
 for(unsigned int i=0;i<info.size();i++)
  info.at(i).parse_amplicon_names();
}

const string alt_to_str()
{
 stringstream s;
 if(alt.size()>0) s << alt.at(0);
 for(unsigned int i=1;i<alt.size();i++) s << "," << alt.at(i) ;
 return s.str();
}

const string ref_to_str()
{
 stringstream s;
 if(ref.size()>0) s << ref.at(0);
 for(unsigned int i=1;i<ref.size();i++) s << "," << ref.at(i) ;
 return s.str();
}

void inline add(const string _ref, const string _alt, const string _info, const char _gt_index, const int _zyg){
 ref.push_back(_ref);
 alt.push_back(_alt);
 info.push_back(VariantFeatures(_info, _gt_index));
 if(info.size()==1){
 set_dp(_info);
 set_zyg(_zyg);
 }
}

void inline add(VCFinfo & rec){
 int original_size = alt_count();
 for(int i=0;i<rec.alt_count();i++)
 {
  bool already_added = false;
  for(int j=0;j<original_size;j++)
    if(strcmp(ref.at(j).c_str(),rec.get_ref(i).c_str())==0 && strcmp(alt.at(j).c_str(),rec.get_alt(i).c_str())==0) 
       {
        already_added = true;
        break;        
       }
 if(!already_added) add(rec.get_ref(i), rec.get_alt(i), rec.get_info(i), rec.get_gt_index(i), rec.get_zyg());
 }
}

VCFinfo(const string _ref, const string _alt, const string _info, const char _gt_index,  const int _zyg)
{
 add(_ref, _alt, _info, _gt_index, _zyg);
}

inline int get_dp() // returns -1 if not available
{
  return DP;
}

VCFinfo(string line, long &alleles_loaded, long &rows_missing_genotype, bool ignore_genotype = false)
  {
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
   while(cols.good())
   {
   string vcf_col;
   getline( cols, vcf_col, '\t');

   switch(++col_idx)
    {
     case 4: {
               ref_al = vcf_col; 
               break;
              } 
     case 5: {
               std::stringstream ss(vcf_col);
               while(ss.good())
               {
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
               while(ss.good())
               {
                gt_col++;
                getline( ss, tmp_format, ':');
                if(!ignore_genotype && strcmp(tmp_format.c_str(), "GT")==0) {GT_idx = gt_col; break;}
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
               while(ss.good())
               {
                gt_col++;
                getline( ss, tmp_value, ':');
                if(gt_col == tmp_idx) {
                   if( tmp_value.empty()) break; 
                   std::stringstream sf(tmp_value);
                   
                   char separator = '|';
                   if(tmp_value.find("/",0) != string::npos) separator = '/';
                   else 
                   if(tmp_value.find("|",0) == string::npos) break;

                   GT_idx = 0; // now GT_idx represents allele number
                                      
                   string gt_value;
                   vector<string> alt_subset;
                   while(sf.good())
                   {
                    getline(sf, gt_value, separator);
                    if(gt_value.length()>0 && isdigit(gt_value[0]))
                    {
                    if( gt_value[0]!='0')
                    {
                     GT_idx = atoi(gt_value.c_str());
                       if(GT_idx-1<(int)alt.size()){
                        bool already_added = false;
                        for(unsigned int i=0;i<added_alt.size();i++) if(added_alt.at(i)==GT_idx) {already_added= true; break;}
                          if(!already_added)
                            {
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

   if((alleles_loaded = (long)alt.size()) > 0)
   {
      if(GT_idx == -1) { rows_missing_genotype++; set_zyg(-1); }
      else set_zyg( (!obs_ref_allele) ? ( alleles_loaded == 1 ? 0 : 2) : 1 );   
      set_dp(info_al);
     
      for(int i=0;i<alleles_loaded;i++) 
      {
       ref.push_back(ref_al);
       info.push_back(VariantFeatures(info_al, (GT_idx == -1) ? (i+1) : (char)added_alt.at(i)));
      }
   }
 }
   
};
//-----------------------------------------------------
class VCFList
{
  std::map<string,std::map<long int,VCFinfo> > vcfrec;

 void add(string chr, long genpos, VCFinfo & rec)
  {
    std::map< string , std::map<long int,VCFinfo> >::iterator it = vcfrec.find(chr);
    if(it == vcfrec.end()) 
    {
     std::map<long int,VCFinfo> emptymap;
     it = vcfrec.insert(std::pair< string , std::map<long int,VCFinfo> > (chr, emptymap)).first;
    }
           
     std::pair<std::map<long int,VCFinfo>::iterator,bool> ret = (it->second).insert ( std::pair<long int,VCFinfo>(genpos, rec));
     if (ret.second==false) ret.first->second.add(rec);
  }
  
  bool add(const string chr, const long genpos, const string ref, const string alt, const string info, const char gt_index, const int zyg)
  {
    std::map< string , std::map<long int,VCFinfo> >::iterator it = vcfrec.find(chr);
    if(it == vcfrec.end()) 
    {
     std::map<long int,VCFinfo> emptymap;
     it = vcfrec.insert(std::pair< string , std::map<long int,VCFinfo> > (chr, emptymap)).first;
    }
     std::map<long int,VCFinfo>::iterator ret = (it->second).find(genpos);
     if(ret == (it->second).end()) 
     {
      (it->second).insert ( std::pair<long int,VCFinfo>(genpos, VCFinfo(ref,alt,info, gt_index, zyg)));
     } 
     else
     {
       if(ret->second.get_zyg() > 9) // this is used only for BED files to store second coordinate
       {if(ret->second.get_zyg() < zyg) ret->second.set_zyg(zyg);}
       else
       {
       for(int i=0;i<ret->second.alt_count();i++)
       {
        if(strcmp(ref.c_str(),ret->second.get_ref(i).c_str())==0 && strcmp(alt.c_str(),ret->second.get_alt(i).c_str())==0) return false;  
       }
       ret->second.add(ref, alt, info, gt_index, zyg);
       }
     }
     return true;
  }


 void add(string vcfline, long &alleles_loaded, long &rows_missing_genotype, long &het_rows, long & hom_rows, bool split_mnp = true, bool ignore_genotype = false)
  {
    alleles_loaded = 0;
    string::size_type pos = vcfline.find("\t",0);
        
    if( pos != string::npos && pos > 0 && pos < vcfline.length() - 1) 
      {
            string chr    = vcfline.substr(0,pos);
            long   genpos = atoi(vcfline.substr(pos+1).c_str());
            
   // extracting genotyped/or all alleles at that position and the info
   // this eliminates non-called alleles that are mixed with genotyped alleles
   
   VCFinfo rec(vcfline, alleles_loaded, rows_missing_genotype, ignore_genotype);
            
   if(rec.get_zyg()==1) het_rows++;
   if(rec.get_zyg()==0) hom_rows++;
   for(int i=0;i<rec.alt_count(); i++)
   {
    string ref = rec.get_ref(i);
    string alt = rec.get_alt(i);
    long shift = 0;
    
    //remove common padding bases
    
    //TG AG  -> T A
    while(shift < (long)ref.length()-1 && shift< (long)alt.length()-1 && ref[ref.length()-1-shift]==alt[alt.length()-1-shift]) shift++;
    if(shift>0) { ref = ref.substr(0, (long)ref.length()-shift); alt = alt.substr(0,(long)alt.length()-shift); }
    //TA TG  -> A G
    shift = 0;
    while(shift < (long)ref.length()-1 && shift< (long)alt.length()-1 && ref[shift]==alt[shift]) shift++;
    if(shift>0) { ref = ref.substr(shift); alt = alt.substr(shift);}
    long adjusted_pos = genpos+shift;
    
     int vt = vartype(ref,alt);

	 //split-MNPs into single SNPs, this is wrong but community still does it
    if(split_mnp && vt == 3){
     for(unsigned int j=0;j<ref.length();j++) if(ref.at(j)!=alt.at(j)) add(chr, adjusted_pos + j, ref.substr(j,1), alt.substr(j,1), rec.get_info(i), rec.get_gt_index(i), rec.get_zyg());
    }
      else      
    if(!add(chr, adjusted_pos, ref, alt, rec.get_info(i), rec.get_gt_index(i), rec.get_zyg())) alleles_loaded--;
   }    
   }
  }
  

public: 
std::map<string,std::map<long int,VCFinfo> > * getList()
{
 return &vcfrec;
}
//-----------------------------------------------------
long int load_file(string filename, bool split_mnp = true, bool ignore_genotype = false)
{
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
  
  while (getline(infile, line)) 
     if( line[0]!='#')
        {
         add(line, al, rows_missing_genotype, het_rows, hom_rows, split_mnp, ignore_genotype);
         if(al>0)
         {
          rows_loaded++;
          alleles_loaded+=al;
         }
         rows_processed++; 
        }  
  
  infile.close();
  
  cerr << "# informative vcf records:" << rows_processed << "\n# records containing alternative allele:" << rows_loaded << "\n# loaded alternative alleles:" << alleles_loaded << endl;
  cerr << "# loaded vcf records with HET genotype:" << het_rows << "\n# loaded vcf records with HOM genotype:" << hom_rows << "\n# loaded vcf records with missing genotype:" <<  rows_missing_genotype << endl;
  return rows_loaded;
}
//-----------------------------------------------------
long int load_bed_file(string filename)
{
  ifstream infile;
  infile.open(filename.c_str());
  string line;

  if (!infile) {
    cerr << "Unable to read " <<  filename << endl;
    exit(1);
  }
  
  long rows_processed = 0;
  long total_target_size = 0;
 
  while (getline(infile, line)) 
     if( line[0]!='#')
        {
          string::size_type pos = line.find("\t",0);
        
           if( pos != string::npos && pos > 0 && pos < line.length() - 1) 
           {
            string chr    = line.substr(0,pos);
            long   start_pos = atoi(line.substr(pos+1).c_str());
            if( (pos = line.find("\t",pos+1))  != string::npos)
            {
             long end_pos = atoi(line.substr(pos+1).c_str());
             if(add(chr, start_pos, "", "", line, 0, end_pos)) total_target_size += end_pos - start_pos;

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
long get_target_vcf(VCFList * vcf_in, VCFList * vcf_out)
{

  long records_on_target = 0;
  for (std::map< string , std::map<long int,VCFinfo> >::iterator bed_chrit = vcfrec.begin(); bed_chrit != vcfrec.end(); ++bed_chrit) 
  {
  
   std::map< string , std::map<long int,VCFinfo> >::iterator vcf_chrit = vcf_in->getList()->find(bed_chrit->first);
   if(vcf_chrit == vcf_in->getList()->end()) continue;

   std::map<long int,VCFinfo>::iterator vcf_posit = vcf_chrit->second.begin();
   
   std::map<long int,VCFinfo>::iterator bed_posit = bed_chrit->second.begin();
   
   if(bed_posit == bed_chrit->second.end()) continue;
   
   long int reg_start = bed_posit->first;
   long int reg_end  = bed_posit->second.get_zyg();
   
   bed_posit++;
          
   while (bed_posit != bed_chrit->second.end())
   {  
     if(  vcf_posit == vcf_chrit->second.end()) break;
    
     if(bed_posit->first <=  reg_end)
      {
       if(bed_posit->second.get_zyg() >  reg_end ) reg_end = bed_posit->second.get_zyg();
       bed_posit++;
       continue;
      } 
        
     if(  reg_end < vcf_posit->first) 
         { 
            reg_start = bed_posit->first;
            reg_end  = bed_posit->second.get_zyg();
           bed_posit++;      
           continue;
         }
         
    while( vcf_posit != vcf_chrit->second.end() && reg_start >= vcf_posit->first) vcf_posit++;
    while( vcf_posit != vcf_chrit->second.end() && reg_end >= vcf_posit->first) {
        if(vcf_posit->first + (long)vcf_posit->second.get_ref(0).length()-1 <= reg_end)
         { vcf_out->add(vcf_chrit->first, vcf_posit->first, vcf_posit->second);
           records_on_target++; 
         }  
        vcf_posit++;
        }       
    }
    
        while( vcf_posit != vcf_chrit->second.end() && reg_start >= vcf_posit->first) vcf_posit++;
    while( vcf_posit != vcf_chrit->second.end() && reg_end >= vcf_posit->first) {
        if(vcf_posit->first + (long)vcf_posit->second.get_ref(0).length()-1 <= reg_end)
         { vcf_out->add(vcf_chrit->first, vcf_posit->first, vcf_posit->second);
           records_on_target++; 
         }  
        vcf_posit++;
        }       
  }
return records_on_target;
}  
//-----------------------------------------------------
long merge_overlapping_segments(VCFList * merged_bed)
{
 long total_target_size = 0;
  for (std::map< string , std::map<long int,VCFinfo> >::iterator bed_chrit = vcfrec.begin(); bed_chrit != vcfrec.end(); ++bed_chrit) 
  {
   std::map<long int,VCFinfo>::iterator bed_posit = bed_chrit->second.begin();   
   
   if(bed_posit == bed_chrit->second.end()) continue;
   
   long int reg_start = bed_posit->first;
   long int reg_end  = bed_posit->second.get_zyg();
   string reg_info = bed_posit->second.get_info(0);
   
   bed_posit++;
          
   while (bed_posit != bed_chrit->second.end())
   {  

     if(bed_posit->first <=  reg_end)
      {
       if(bed_posit->second.get_zyg() >  reg_end ) reg_end = bed_posit->second.get_zyg();
       bed_posit++;
       continue;
      } 
    
    merged_bed->add(bed_chrit->first,reg_start, "", "", reg_info, 0, reg_end);
    total_target_size+= reg_end - reg_start;
        
    reg_start = bed_posit->first;
    reg_end  = bed_posit->second.get_zyg();
	reg_info = bed_posit->second.get_info(0);
    bed_posit++;      
   
    }
    
    merged_bed->add(bed_chrit->first,reg_start, "", "", reg_info, 0, reg_end);
    total_target_size+= reg_end - reg_start;
  }
  
return total_target_size;
}  
//-----------------------------------------------------
long intersect(VCFList * in, VCFList * bed_out)
{
  long total_target_size = 0;
  for (std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit) 
  {
   std::map< string , std::map<long int,VCFinfo> >::iterator in_chrit = in->getList()->find(chrit->first);
   if(in_chrit != in->getList()->end()) 
    for (std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(), in_posit = in_chrit->second.begin(); posit != chrit->second.end() && in_posit != in_chrit->second.end(); )
     {
      if(posit->first > in_posit->second.get_zyg()) { in_posit++; continue;}
      if(in_posit->first > posit->second.get_zyg()) { posit++; continue;}
      long reg_start = in_posit->first < posit->first ? posit->first : in_posit->first;
      long reg_end = posit->second.get_zyg() < in_posit->second.get_zyg() ? posit->second.get_zyg() : in_posit->second.get_zyg();
      total_target_size += reg_end - reg_start;
      bed_out->add(chrit->first, reg_start, "", "", "", 0, reg_end);
      if(posit->second.get_zyg() == reg_end) posit++; else in_posit++;
     }
  }
  return total_target_size;
}
//-----------------------------------------------------
void add_ref_seq(VCFList * bed_out, FASTA reference)
{
 for (std::map< string , std::map<long int,VCFinfo> >::iterator chrit = vcfrec.begin(); chrit != vcfrec.end(); ++chrit) 
    for (std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); ++posit)
     {
      long start_pos = posit->first;
      long end_pos = posit->second.get_zyg();
	  string info = posit->second.get_info(0);
      bed_out->add(chrit->first, start_pos, reference.get_ref_segment(chrit->first, start_pos, end_pos+500), "", info, 0, end_pos); // 500 is just a padding number, no special meaning
     }
}  

};

void load_region_reads( BamReader &reader, vector<BamAlignment> * buffer, string input_bam, int32_t refid=0, int32_t start_pos=0, int32_t end_pos=0, int min_map_qv = 0) // zero based, half-open (BED)
{   
    buffer->clear();
    	if(!reader.SetRegion(refid,start_pos+(end_pos-start_pos)/3,refid,end_pos-(end_pos-start_pos)/3))
	{
		reader.Close();
		reader.Open(input_bam);
		reader.SetRegion(refid,start_pos+(end_pos-start_pos)/3,refid,end_pos-(end_pos-start_pos)/3);
	}
	BamAlignment aread;
	while(reader.GetNextAlignment(aread)) if (aread.IsMapped() && aread.MapQuality>=min_map_qv && aread.Position <= start_pos-10 && aread.GetEndPosition() >= end_pos+10) buffer->push_back(aread);
}

inline unsigned seq_to_int(const char * seq)
{
 unsigned int_seq = 0;
 int i = -1;
 while(seq[++i]!='\0')
    {
     int_seq<<=2;
     switch( seq[i] )
     {
     case 'A': break;
     case 'C': int_seq|= 1; break;
     case 'G': int_seq|= 2; break;
     case 'T': int_seq|= 3; break;
     default:; 
     }
    } 
  return int_seq;
 } 

unsigned getIndex_preamp_tagging(BamAlignment & aread, bool one_index=true)
{
    string queryBases = aread.QueryBases;
    int qlen = queryBases.length();
    unsigned a_index = 0;
    
	if(qlen>50)
	{	
	// detection of a_index, should be very accurate
	  a_index = seq_to_int(queryBases.substr(0,10).c_str());
	  if(one_index) return a_index;
	  
	  int p1_adapter_pos = -1;
      if( aread.GetTag("ZA",p1_adapter_pos) && p1_adapter_pos-qlen < 3 ) 
	  {
	   unsigned p1_index = seq_to_int((queryBases.substr(qlen-9,3)+queryBases.substr(qlen-3)).c_str());
	   p1_index <<= 4;
	   a_index+=p1_index;
	    // here we can have p1_index verification p1_index.substr(3,3) = "AGT";
	  } else return 0;
	  
      unsigned pos = aread.Position;

      vector<CigarOp>::const_iterator cigItr = aread.CigarData.begin();
      vector<CigarOp>::const_iterator cigEnd = aread.CigarData.end();

      if(cigItr != cigEnd && cigItr->Type =='S') pos -= cigItr->Length;
      
      pos = pos % 255; 
	  a_index <<= 8;
	  a_index += pos;
	}
	return a_index;
}


string getIndex_AmpliSeq_primers(BamAlignment & aread, bool one_index=true) // one_index=false => p1_index is used instead of a_index
{
    string queryBases = aread.QueryBases;
    int qlen = queryBases.length();
	
	string a_index =  "AAAAAAAAAAAAAA"; //14bp
    string p1_index = "AAAAAAAAAAAAAA";

	if(qlen>100)
	{	
      bool IsReverseStrand = aread.IsReverseStrand();
	
	// detection of a_index, should be very accurate
	   a_index = IsReverseStrand?queryBases.substr(qlen-14):queryBases.substr(0,14);	
    
	int p1_adapter_pos = -1;
    if( aread.GetTag("ZA",p1_adapter_pos) && p1_adapter_pos-qlen < 3 ) 
	 {
	   p1_index = IsReverseStrand?queryBases.substr(0,14):queryBases.substr(qlen-14);
	 }
	}
 return one_index?a_index:p1_index;
}

	  const string a_handle_pref = "CTGTACGGT";
	  const string a_handle_suff = "GACAAGGCG";
	  bool is_correct_adaptor_trimming = true;


string getIndex_barcoded_primers(BamAlignment & aread, bool one_index=true)
{
    string queryBases = aread.IsReverseStrand() ? reverse(aread.QueryBases) : aread.QueryBases;
    int qlen = queryBases.length();
	
	string a_index = "AAAAAA"; //6bp
    string p1_index = "AAAAAA";
    string fake_p1_index = p1_index;
	
	// detection of a_index, should be very accurate
	// for incomplete IonXpress barcode
	  if(!is_correct_adaptor_trimming)
	  {
	  string a_seq = queryBases.substr(0,32);
	  string::size_type handle_pos = string::npos;
	  if((handle_pos = a_seq.find(a_handle_suff)) != string::npos && handle_pos+18 < a_seq.length()) a_index = a_seq.substr(handle_pos+9,3) + a_seq.substr(handle_pos+15,3);	
	  else
	  if((handle_pos = a_seq.find(a_handle_pref)) != string::npos && handle_pos+24 < a_seq.length()) a_index = a_seq.substr(handle_pos+18,3) + a_seq.substr(handle_pos+21,3);	
         // here we can have a_index verification a_index.substr(3,3) = "ACT";
	  }	 
       //for IonCodeTag barcodes     
	   else a_index = queryBases.substr(0,3) + queryBases.substr(6,3);
	 
     

	int p1_adapter_pos = -1;
    if( aread.GetTag("ZA",p1_adapter_pos) && p1_adapter_pos == qlen ) 
	 {
	   p1_index = queryBases.substr(qlen-9,3)+queryBases.substr(qlen-3);
	    // here we can have p1_index verification p1_index.substr(3,3) = "AGT";
	 }

	 return one_index?(a_index+fake_p1_index):(a_index+p1_index); 
}
//----------------------------------------------------
// Similar functionality to above functions, just reading tags from BAM
// Hard coded to assume 12 base Sherlock type tags

string GetTagsFromBamAlignment(BamAlignment & alignment, bool prefix_only=true)
{
  string prefix_tag = "AAAAAAAAAAAA";
  string suffix_tag = "AAAAAAAAAAAA";

  // Load Tags from Bam Alignment
  if (not alignment.GetTag("ZT", prefix_tag))
    prefix_tag = "AAAAAAAAAAAA";
  if (prefix_tag.length() < 12)
    prefix_tag += string(12-prefix_tag.length(), 'A');

  if (prefix_only == false){
    if (not alignment.GetTag("YT", suffix_tag))
      suffix_tag = "AAAAAAAAAAAA";
  }
  if (suffix_tag.length() < 12)
    suffix_tag += string(12-suffix_tag.length(), 'A');

	return (prefix_tag.substr(0,3) + prefix_tag.substr(6,3) + suffix_tag.substr(3,3) + suffix_tag.substr(9,3));
}


//----------------------------------------------------

static char base[4] = {'A','C','G','T'};

inline string int_to_seq(int len, unsigned long long int_seq)
{
 char seq[len];
 int i = -1;
 unsigned tmp = int_seq;
 while((++i)<len)
 {
  seq[len-1-i] = base[tmp&3];
  tmp>>=2;
 }
return ctos(seq,len);
}

void build_index_table( vector<BamAlignment> * buffer, unsigned char * index_counts, int tagging_method, unsigned long & total_reads_p1, bool one_index=true)
{
    string a_index, p1_index, fake_p1_index;

  for (std::vector<BamAlignment>::iterator a = buffer->begin() ; a != buffer->end(); ++a)
  {
    BamAlignment & aread = *a;
    int p1_adapter_pos = -1;
    if( aread.GetTag("ZA",p1_adapter_pos) ) total_reads_p1++;
    switch (tagging_method)
	{
	//ampliseq 
    case 1: index_counts[seq_to_int(getIndex_AmpliSeq_primers(aread, one_index).c_str())]++; break; 
	//bcprimer 
    case 2: {unsigned idx = seq_to_int(getIndex_barcoded_primers(aread, one_index).c_str()); if(index_counts[idx]<254)index_counts[idx]++; break;}
    //preamp
    case 3: {index_counts[getIndex_preamp_tagging(aread, one_index)]++; break;}
    // The tags are trimmed and stored in the BAM body
    case 4: {unsigned idx = seq_to_int(GetTagsFromBamAlignment(aread, one_index).c_str()); if(index_counts[idx]<254) index_counts[idx]++; break;}
    }
 }
}
//---------------------------------------------------------------------------------------
bool valid_haplotype(PileUp * pileup, vector<std::map< int , string > > * read_filter, int ref_idx, const int32_t start_pos, long target_size )
{
  for(unsigned int i=0;i<read_filter->size();i++)	
  {
   int max_match_number = 0;
   int match_number = 0;
   int n=0;
  for (std::map<int, string>::iterator rf = read_filter->at(i).begin(); rf != read_filter->at(i).end(); ++rf, n++)
   {
	 if(n==0) max_match_number = rf->first;
 	  else
     for (std::map<string, int>::iterator a = pileup[rf->first-start_pos-1].get_alt_set()->begin(); a != pileup[rf->first-start_pos-1].get_alt_set()->end(); ++a)
		if(a->first[0]==rf->second[0] && ( a->first[0]!='+' || (rf->second.length()>1 && a->first[1]==rf->second[1]))) match_number++;
   }	 
  if(match_number >= max_match_number) return false;
 }
 return true;    
}
//-------------------------------------------------------------------	
map<unsigned,PileUp *> * consensus_analysis(string & ref_seq, vector<BamAlignment> * buffer, unsigned char * index_counts, unsigned * global_consensus, vector<std::map< int , string > > * read_filter, int ref_idx, const int32_t start_pos, const int32_t end_pos, int tagging_method, const unsigned min_family_size = 3, bool one_index=true) // one based, half-open (BED)
{
  map<unsigned, PileUp *> * top_family = new std::map<unsigned, PileUp *>();
  int target_size = end_pos - start_pos;
  
  for (std::vector<BamAlignment>::iterator a = buffer->begin() ; a != buffer->end(); ++a)
  {
    BamAlignment & aread = *a;

    int bc_len = 14;
	int a_handle_left = (tagging_method==2) ? 18 : 0; 
	int a_handle_right = (tagging_method==2) ?  18 : 0; 
	if(aread.IsReverseStrand()) a_handle_left=0; else a_handle_right = 0;
	 
	// If the tags are in the BAM tags, they are no longer in the query sequence
	if (tagging_method ==4){
	  bc_len = 0;
	  a_handle_left = a_handle_right = 0;
	}
	
	unsigned index = 0;
	if (tagging_method==1)
	  index = seq_to_int(getIndex_AmpliSeq_primers(aread, one_index).c_str());
    else if (tagging_method==2)
      index = seq_to_int(getIndex_barcoded_primers(aread, one_index).c_str());
	else if (tagging_method==3)
	  index = getIndex_preamp_tagging(aread, one_index);
	else if (tagging_method==4)
	  index = seq_to_int(GetTagsFromBamAlignment(aread, one_index).c_str());

	PileUp * pileup = index_counts[index]>=min_family_size ? (new PileUp[target_size]) : NULL;
	
    string & queryBases = aread.QueryBases;
	int & pos = aread.Position;
	
    if(aread.RefID == ref_idx && pos>=start_pos-500 && pos<end_pos)
	{
	
	int posInRef = 0;
    int posInRead=0;
    int validSeqPos_left = bc_len + a_handle_left; // do not include barcode sequences into analysis 13bp (14bp for ampliseq, 12bp for bcprimer, 13bp for preamp)
	int validSeqPos_right = queryBases.length()-bc_len-a_handle_right; // do not include barcode sequences into analysis 13bp (14bp for ampliseq, 12bp for bcprimer, 13bp for preamp)
	int aln_shift = pos-start_pos;

     vector<CigarOp>::const_iterator cigItr = aread.CigarData.begin();
     vector<CigarOp>::const_iterator cigEnd = aread.CigarData.end();
     for ( ; cigItr != cigEnd; ++cigItr ) 
       if(cigItr->Type != 'H')
       {
        int cgl = cigItr->Length;
        switch (cigItr->Type)
        {
         case 'M': {
					if(posInRef+aln_shift < target_size && posInRef+aln_shift + cgl >= 0)
					{
					 const char * ref_sub_seq = &ref_seq[posInRef];
					 const char * aln_sub_seq = &queryBases[posInRead];
					 for(int i = 0;i<cgl;i++)
					 {
						 if(posInRef + aln_shift + i >=0 && posInRef + aln_shift + i < target_size && posInRead + i > validSeqPos_left && posInRead + i < validSeqPos_right)
						 {
					      int x = 12*(posInRef + aln_shift + i);
						  switch (aln_sub_seq[i])
						 {
							case 'C': x+=1; break;
							case 'G': x+=2; break;
							case 'T': x+=3; break;
						 }
                          global_consensus[x]++;
						  if(pileup!=NULL && index_counts[index]>=min_family_size ) pileup[posInRef + aln_shift + i].add(queryBases.substr(posInRead+i,1));
						}
					 };
					}
			 break;}
         case 'S': {
					// opportunity for long indel assembly
			 break;}
         case 'I': {
					if(pileup!=NULL && index_counts[index]>=min_family_size && posInRef + aln_shift - 1 >=0 && posInRef + aln_shift - 1 < target_size && posInRead > validSeqPos_left && posInRead + cgl < validSeqPos_right)
					{ 
 				        int x = 12*(posInRef + aln_shift - 1);
						 switch (queryBases[posInRead])
						 {
							case 'C': x+=5; break;
							case 'G': x+=6; break;
							case 'T': x+=7; break;
						 }
                        global_consensus[x]++;
						pileup[posInRef + aln_shift - 1].add("+" + queryBases.substr(posInRead,cgl));
					}
			 break;}
         case 'D': {
					if(pileup!=NULL && index_counts[index]>=min_family_size && posInRef + aln_shift -1 >=0 && posInRef + aln_shift + cgl -1 < target_size && posInRead > validSeqPos_left && posInRead < validSeqPos_right)
					{
				        int x = 12*(posInRef + aln_shift - 1);
						  switch (ref_seq[posInRef + aln_shift])
						 {
							case 'C': x+=9; break;
							case 'G': x+=10; break;
							case 'T': x+=11; break;
						 }
                        global_consensus[x]++;
						
					  pileup[posInRef + aln_shift - 1].add("-" + ref_seq.substr(posInRef + aln_shift,cgl));
				      for(int d=0;d<cgl;d++) pileup[posInRef + aln_shift + d].add("$");}
			 break;}
        }
        
		if(cigItr->Type != 'I' && cigItr->Type != 'S') posInRef+=cgl;
        if(cigItr->Type !='D' )  posInRead+=cgl;
       }
     }
	 
	 if(pileup!=NULL && valid_haplotype(pileup, read_filter, ref_idx, start_pos, target_size )) // exclude systematic multi-error reads
	 {
	 std::map<unsigned, PileUp *>::iterator ret = top_family->find(index);
     if(ret == top_family->end()) ret = top_family->insert ( std::pair<unsigned,PileUp *>(index, pileup)).first;
     else 
		{
		  for(int i=0;i<target_size;i++)
	     	  for (std::map<string, int>::iterator a = pileup[i].get_alt_set()->begin() ; a != pileup[i].get_alt_set()->end(); ++a)
  				  ret->second[i].add(a->first);	 	 
		  if(pileup!=NULL) delete [] pileup;  
		}	  
	 } else if(pileup!=NULL) delete [] pileup; 	 
  }
  return top_family; 
}
//---------------------------------------------------------------------------------------
bool load_read_filters(vector<std::map< int , string > > * read_filter, VCFinfo & rec)
{
  bool haplo_filter = true;
		std::map<string,bool> rfmap;
				for(int a=0;a<rec.alt_count();a++)
				{
			     string rf = rec.get_str_flag_value(a,"READ_FILTER");
			     
			     if(haplo_filter){
			      string pf = rec.get_str_flag_value(a,"HAPLO_FILTER");
			      if(pf.length()>0 && pf[0]=='0') haplo_filter = false;
			      }
			      
				  if(rf.length()>0){
					 std::stringstream ss(rf);
                     string rf_temp;
                     while(ss.good())
						{
                         getline( ss, rf_temp, ':');
						 rfmap.insert(std::pair<string, bool>(rf_temp, 1));
						}
				    }	
				}
	
	for (std::map< string , bool >::iterator it = rfmap.begin(); it != rfmap.end(); ++it) 
	{
	  std::stringstream ss(it->first);
	  string sign_mut;
	  string::size_type dot_pos = string::npos;
	  std::map< int , string > signature;
	     while(ss.good())
		{
         getline( ss, sign_mut, '^');
		 if(sign_mut.length()>2 && (dot_pos = sign_mut.find(".")) != string::npos && dot_pos>0 && dot_pos<sign_mut.length()-1 && isdigit(sign_mut[0]))
		 {
		  signature.insert(std::pair<int, string>(atoi(sign_mut.c_str()), sign_mut.substr(dot_pos+1)));
         }
	    }
	if(!signature.empty()) read_filter->push_back(signature);
	}
	return haplo_filter;	
}				
//---------------------------------------------------------------------------------------
void report_consensus_counts(string & ref_seq, string ref_name, map<unsigned,PileUp *> * top_family, unsigned * global_consensus,  int tagging_method, int ref_idx, const int32_t start_pos, const int32_t end_pos, ofstream & alt_fam, ofstream & global_counts, ofstream & targ_fam_sz, ofstream & var_calls, vector<long int> & hotspot_pos, vector<VCFinfo> & hotspot_rec, unsigned reads_on_target, unsigned reads_with_p1, int vc_min_fam_size, int vc_min_func_cov,  float vc_min_func_maf, int vc_min_num_fam, string trg_def, int max_var_amp, bool hs_mask_only)
{
	int target_size = end_pos-start_pos;
	const int   indx_size = (tagging_method==1?14:12);
     
	 //accuracy of consensus
	 const float min_alt_freq = 0.8;
	 
	 unsigned num_fm_sz3 = 0;
	 unsigned num_fm_sz4 = 0;
	 unsigned num_fm_sz7 = 0;
	 unsigned alt_fm_sz3 = 0;
	 unsigned alt_fm_sz4 = 0;
	 unsigned alt_fm_sz7 = 0;
	 unsigned num_exluded_fam = 0;
 	 
	 unsigned cons_sz[target_size];
	 PileUp** functional_pileup = new PileUp*[target_size];
	 
     for(int i=0; i<target_size; i++) {
		 cons_sz[i]=0;
		 for(int j=0; j<12; j++) cons_sz[i] += global_consensus[12*i+j];
		 functional_pileup[i] = new PileUp[11];
	 }

	 for (std::map<unsigned,PileUp *>::iterator family = top_family->begin(); family != top_family->end(); ++family) 
     {
      bool has_ref3 = false;
	  bool has_ref4 = false;
      bool has_ref7 = false;
      bool has_alt3 = false;
	  bool has_alt4 = false;
      bool has_alt7 = false;
      
	  int alt_count_tmp = 0;
	  int num_called_vars = 0;
      for(int i=0; i< target_size; i++)
	  if(family->second[i].get_coverage() >= vc_min_fam_size)
	   {
		 string cpf = family->second[i].consensus_call(min_alt_freq, alt_count_tmp);
		 if(cpf.length()>0 && cpf[0]!=ref_seq[i] && cpf[0]!='$') num_called_vars++;
	   } 
	
	  if(num_called_vars > max_var_amp) num_exluded_fam++;

      //rewrite counts
      for(int i=0; i< target_size; i++)
	  {
		unsigned fm_sz = family->second[i].get_coverage();
		if(fm_sz>1)
		{
		int alt_counts = 0;
	    string cpf = family->second[i].consensus_call(min_alt_freq, alt_counts);
		
		if(cpf.length()>0) {

		if(fm_sz >= (unsigned)vc_min_fam_size && num_called_vars <= max_var_amp){ functional_pileup[i][fm_sz-vc_min_fam_size>=10?9:fm_sz-vc_min_fam_size].add(cpf); functional_pileup[i][10].add(cpf);	}

	    if(cpf[0]!=ref_seq[i] && cpf[0]!='$') 
		{
		 has_alt3 = true; 
		 if(fm_sz>=4) has_alt4 = true;
		 if(fm_sz>=7) has_alt7 = true;
	     alt_fam <<  ref_name << "\t" << (start_pos+i+1) << "\t"<< alt_counts << "\t" << fm_sz << "\t"  << ref_seq[i] << "\t" << cpf << "\t" << int_to_seq(indx_size, family->first) << "\t" << cons_sz[i] << endl;
		} else  {
			has_ref3 = true; 
			if(fm_sz>=7) has_ref7 = true;
			if(fm_sz>=4) has_ref4 = true;
			}
		}	
	   }	
	 }
	 
   if(has_alt3) alt_fm_sz3++;
   if(has_alt4) alt_fm_sz4++;
   if(has_alt7) alt_fm_sz7++;
   if(has_ref3) num_fm_sz3++;
   if(has_ref4) num_fm_sz4++;
   if(has_ref7) num_fm_sz7++;
   }	
 
   targ_fam_sz << trg_def << "\t"<< reads_on_target << "\t"<< reads_with_p1 << "\t" << (num_fm_sz3) << "\t" << (num_fm_sz4) <<"\t" << (num_fm_sz7) <<"\t" << (alt_fm_sz3)<<"\t" << (alt_fm_sz4) << "\t" << (alt_fm_sz7) << "\t" << (num_exluded_fam) << endl;

    for(unsigned int i=0,h=0; i<(unsigned)target_size; i++)
	 {	 
		// implement indel caller in generic vc to add to molec calls
		unsigned int max_alt_freq = 0;
		unsigned int max_alt_idx = 0;
		if(cons_sz[i] >= 15000) //min coverage
		for(unsigned int j=0; j<4; j++)
		if(ref_seq[i]!=acgt[j] && max_alt_freq < global_consensus[12*i+j]){ max_alt_freq = global_consensus[12*i+j]; max_alt_idx=j;}
		
		float AF = cons_sz[i]>0?((double)max_alt_freq)/((double)cons_sz[i]):0;
		// min frequency
		string ALT =( AF >= vc_min_func_maf) ? acgt.substr(max_alt_idx,1) : ".";
		global_counts << ref_name << "\t" << (start_pos+i+1) << "\t.\t"  << ref_seq[i] << "\t"<< ALT <<"\t" << cons_sz[i] << "\t.\tDP=" << cons_sz[i] <<";AO="<<max_alt_freq<<";AF="<<AF<<"\tGT:A:C:G:T:+A:+C:+G:+T:-A:-C:-G:-T\t"<<(ALT[0]!='.'?"0/1:":"0/0:") << global_consensus[12*i] << ","<< global_consensus[12*i+1] << ","<< global_consensus[12*i+2] << ","<< global_consensus[12*i+3] << ",+" << global_consensus[12*i+4] << ",+"<< global_consensus[12*i+5] << ",+"<< global_consensus[12*i+6] << ",+"<< global_consensus[12*i+7] << ",-" << global_consensus[12*i+8] << ",-"<< global_consensus[12*i+9] << ",-"<< global_consensus[12*i+10] << ",-"<< global_consensus[12*i+11] << endl;

		unsigned functional_counts =	functional_pileup[i][10].get_plus_insert_coverage();
	    double LOD = ((double)vc_min_num_fam - 0.5)/((double)functional_counts);
		float output_lod = ((int)(LOD*10000))%10 >= 5 ? ((int)((LOD+0.0005)*1000))*0.001 : (((int)(LOD*1000))*10+5)*0.0001;
		if(output_lod<0.0001)output_lod=0.0001;
		if(output_lod>100)output_lod=100;
	    double min_freq = LOD * 0.6;
	    bool hs_pos = false;	
	    if(h<hotspot_pos.size() && hotspot_pos[h]==(start_pos+i+1)) {hs_pos=true; h++;} 	
	
    	if(functional_counts >= (unsigned)vc_min_func_cov)
		{
          std::map<string, int> * alt_set = functional_pileup[i][10].get_alt_set();
          bool reported = false;
		  for (std::map< string , int >::iterator alt_key = alt_set->begin(); alt_key != alt_set->end(); ++alt_key)
		 if( alt_key->first[0]!=ref_seq[i] && alt_key->first[0]!='$' && alt_key->second>=vc_min_num_fam && (double)alt_key->second/((double)functional_counts) >= vc_min_func_maf ) 
		  {
             double faf = (double)alt_key->second/((double)functional_counts);
			 int qual = 10*log(3*faf/min_freq);	 if(qual>1000) qual = 1000; if(qual<0) qual = 0;
			 
			 int j=-1;
			 switch(alt_key->first[0])
			 {
				 case 'A': j=0; break;
				 case 'C': j=1; break;
				 case 'G': j=2; break;
				 case 'T': j=3; break;
				 case '+':{
					switch(alt_key->first[1])
					{
					case 'A': j=4; break;
					case 'C': j=5; break;
					case 'G': j=6; break;
					case 'T': j=7; break;
					default: j=1;
					}
				 break;}
				 case '-':{
					switch(alt_key->first[1])
					{
					case 'A': j=8; break;
					case 'C': j=9; break;
					case 'G': j=10; break;
					case 'T': j=11; break;
					default: j=1;
					}
				 break;}
				 default: j=-1;
			 }
			 
			 if(j>=0 && (hotspot_pos.size() == 0 || hs_pos || hs_mask_only))
			 {
			 double af = ((double)global_consensus[12*i+j]/cons_sz[i]);

             string alt_allele = alt_key->first;
			 int allele_type = alt_allele[0]=='-'?0:(alt_allele[0]=='+'?2:1); // del=0,snp=1;ins=2
		     string ref_allele = (allele_type==0?(ref_seq.substr(i,1)+alt_allele.substr(1)):ref_seq.substr(i,1));
			 if(allele_type==2) alt_allele=ref_seq.substr(i,1) + alt_allele.substr(1); else  if(alt_allele[0]=='-') alt_allele = ref_seq.substr(i,1);
			 if(allele_type==1 && ref_allele.length()>1) allele_type = 3; // mnp=3
             
			if(faf >= min_freq)
			   {
			
				 bool filter_out = false;
			     // filter out black-listed alleles
				 if(hs_pos)
				 {	 
				  //hotspot_rec[h-1].parse_info_values();
				  for(int a=0;a<hotspot_rec[h-1].alt_count();a++)
				   if(strcmp(alt_allele.c_str(),hotspot_rec[h-1].get_alt(a).c_str()) == 0)
					  {
						string bstrand = hotspot_rec[h-1].get_str_flag_value(a,"BSTRAND");
						if(bstrand.length()>0 && strcmp(bstrand.c_str(),"B")==0) filter_out = true;
					  }
				 } 	  

	     		if(filter_out)var_calls << ref_name << "\t" << (start_pos+i+1) << "\t.\t"  << ref_allele << "\t" << alt_allele << "\t" << qual << "\tNOCALL\tDP=" << cons_sz[i] << ";AO=" << global_consensus[12*i+j] << ";AF=" << af << ";MDP=" << functional_counts << ";MAO=" << alt_key->second << ";MAF=" << faf << ";LOD=" << output_lod << ";TYPE=" << vartypes[allele_type] << ";FR=systematic_error\tGT:DP:AO:AF:FS\t" << "./.:" << cons_sz[i] << ":" << global_consensus[12*i+j] << ":" << af << ":";
				else var_calls << ref_name << "\t" << (start_pos+i+1) << "\t.\t"  << ref_allele << "\t" << alt_allele << "\t" << qual << "\tPASS\tDP=" << cons_sz[i] << ";AO=" << global_consensus[12*i+j] << ";AF=" << af << ";MDP=" << functional_counts << ";MAO=" << alt_key->second << ";MAF=" << faf << ";LOD=" << output_lod << ";TYPE=" << vartypes[allele_type] << "\tGT:DP:AO:AF:FS\t" << (faf>0.999?"1/1:":"0/1:") << cons_sz[i] << ":" << global_consensus[12*i+j] << ":" << af << ":";
				for(int k=0;k<9;k++) var_calls << functional_pileup[i][k].get_allele_read_counts(alt_key->first) << ",";
				var_calls << functional_pileup[i][9].get_allele_read_counts(alt_key->first) << endl;
				reported = true;
				} 

		    else if(!hs_mask_only && hotspot_pos.size() > 0 && strcmp(hotspot_rec[h-1].get_ref(0).c_str(),ref_allele.c_str()) == 0 && strcmp(hotspot_rec[h-1].get_alt(0).c_str(),alt_allele.c_str()) == 0)
			{
				var_calls << ref_name << "\t" << (start_pos+i+1) << "\t.\t"  << ref_allele << "\t" << alt_allele << "\t" << 0 << "\tNOCALL\tDP=" << cons_sz[i] << ";AO=" << global_consensus[12*i+j] << ";AF=" << af << ";MDP=" << functional_counts << ";MAO=" << alt_key->second << ";MAF=" << faf << ";LOD=" << output_lod << ";FR=MAF<"<< faf << ";TYPE=" << vartypes[allele_type] <<";HS\tGT:DP:AO:AF:FS\t" << "./.:" << cons_sz[i] << ":" << global_consensus[12*i+j] << ":" << af << ":";
				for(int k=0;k<9;k++) var_calls << functional_pileup[i][k].get_allele_read_counts(alt_key->first) << ",";
				var_calls << functional_pileup[i][9].get_allele_read_counts(alt_key->first) << endl;
				reported = true;
			}
			}
     	  } 
     	  if (hs_pos && !reported && !hs_mask_only)
		  {
			    for(int a=0;a<hotspot_rec[h-1].alt_count() && a<1 /*output just one allele, requires more work to output more alleles*/;a++)
				{	
			    int allele_type = hotspot_rec[h-1].get_ref(a).length()>hotspot_rec[h-1].get_alt(a).length()?0:(hotspot_rec[h-1].get_ref(a).length()<hotspot_rec[h-1].get_alt(a).length()?2:1); // del=0,snp=1;ins=2
				if(allele_type==1 && hotspot_rec[h-1].get_ref(a).length()>1) allele_type = 3; // mnp
				if((allele_type==0 && (hotspot_rec[h-1].get_alt(a).length()>1 || hotspot_rec[h-1].get_alt(a)[0]!=hotspot_rec[h-1].get_ref(a)[0])) || (allele_type==2 && (hotspot_rec[h-1].get_ref(a).length()>1 || hotspot_rec[h-1].get_alt(a)[0]!=hotspot_rec[h-1].get_ref(a)[0]))) allele_type = 4; // complex
				var_calls << ref_name << "\t" << (start_pos+i+1) << "\t.\t"  << hotspot_rec[h-1].get_ref(a) << "\t" << hotspot_rec[h-1].get_alt(a) << "\t" << 0 << "\tREF\tDP=" << cons_sz[i] << ";AO=.;AF=.;MDP=" << functional_counts << ";MAO=.;MAF=.;LOD=" << output_lod  << ";TYPE=" << vartypes[allele_type] << ";HS\tGT:DP:AO:AF:FS\t" << "0/0:" << cons_sz[i] << ":.:.:";
				for(int k=0;k<9;k++) var_calls << functional_pileup[i][k].get_allele_read_counts(ref_seq.substr(i,1)) << ",";
				var_calls << functional_pileup[i][9].get_allele_read_counts(ref_seq.substr(i,1)) << endl;
				}
		  }
	    }	
    }
    
	for(int i=0; i<target_size; i++) delete [] functional_pileup[i];
	delete [] functional_pileup;
}
//---------------------------------------------------------------
void report_target_family_counts(unsigned char * index_counts,int tagging_method,string & tgdef, ofstream & all_fam, unsigned * sz_dist, unsigned & n_fam3, unsigned & n_fam7)
{
 int indx_size = (tagging_method==1?14:12);
 for(unsigned index=0; index<INT_MAX/125; index++)
 if(index_counts[index]>=3)
    {
      all_fam << tgdef << "\t" << int_to_seq(indx_size, index) << "\t" << (unsigned)index_counts[index] << endl;
	  int fm_sz = index_counts[index]; if(fm_sz>30) fm_sz=30;
	  sz_dist[fm_sz]++;
	  n_fam3++;if(fm_sz>=7) n_fam7++;
	}
}


//--------------------------------------------------
void print_usage(int ext_code, string prog_version)
{
 cerr << "*****************************************************************" << endl;
 cerr << "Tagged Molecule Caller tool v." << prog_version                    << endl;
 cerr << "Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved"  << endl;
 cerr << "The tool clusters reads by families and generates variant calls."  << endl; 
 cerr <<"******************************************************************" << endl;
 cerr << "Usage: tmol --fasta a --bam b --target bed --tag-method m --tag-level x --ofamily f1 --oglobal f2 [--option value]" << endl << endl;
 cerr << "[Options]:" << endl;
 cerr << "--fasta      reference    filename with the reference sequence in FASTA format." << endl;
 cerr << "--bam        filename     filename of the indexed BAM file(s)." << endl;
 cerr << "--target     bedfile      bedfile is a file with a set of target regions in BED format. " << endl; 
 cerr << "                          limit analysis to variants with anchor base in the target region." << endl; 
 cerr << "--hotspot    vcffile      vcffile with hotspot positions in VCF format. Calls and no-calls are reported at" << endl; 
 cerr << "                          these positions only. Allele values are disregarded. BED file is still required." << endl; 
 cerr << "--truth      filename     filename with variants in VCF format (requires at minimum first 5 VCF columns)."   << endl;
 cerr << "--tag-method name         supported choices are: ampliseq, bcprimer, preamp, in-BAM. Default (in-BAM)."  << endl;
 cerr << "--tag-level  number       supported choices are: single, double. Default(double)."   << endl;
 cerr << "--oALT       filename     output filename with list of ALT families and their size for each position on target."   << endl;
 cerr << "--ofamily    filename     output filename with list of families and their size for each target region."   << endl;
 cerr << "--oglobal    filename     output filename with allele counts for each position on target regardless of family."   << endl;
 cerr << "--ostats     filename     output filename with gloabal statistics about the run."   << endl;
 cerr << "--ocalls     filename     output filename with variant calls in vcf format."   << endl;
 cerr << endl;
 cerr << "[Variant Caller Options]:" << endl;
 cerr << "--fam-size      n     minimum number of reads with the same tag required for accurate reconstruction of original molecule. Default(3)."<< endl;
 cerr << "--func-cov      m     minimum total number of functional families required to do a call. Default(3)."   << endl;
 cerr << "--func-maf      x     minimum frequency of functional families supporting a variant to the number of other families. Default(0.0005)."   << endl;
 cerr << "--min-num-fam   y     minimum number of families supporting variant required to make a call. Default(2)." << endl; 
 cerr << "--hp-max-length h     indels that are part of HP of size larger than h are filtered out. Default(5)." << endl; 
 cerr << "--min-map-qv    q     alignments with mapping quality value bellow q are filtered out. Default(4)." << endl; 
 cerr << "--max-var-amp   v     families with more than v variants per amplicon are excluded from pileup. Default(2)." << endl; 
 cerr << "--a-handle-on         flag to support TS5.0 bam files with partially trimmed IonXpress adaptor. Default(off)." << endl;
 cerr << "--hs-mask-only        flag to enable whole target variant calling where hotspots are used for error masking only. Default (off)." << endl;
 cerr << "--sample-name   name  sample name to be displayed in variant calls file." << endl;
 cerr << endl;
 exit(ext_code);
}
//--------------------------------------------------

int main(int argc, char *argv[])
{
	
  string full_version_string = IonVersion::GetVersion() + "." +IonVersion::GetRelease() +
                               " (" + IonVersion::GetGitHash() + ") (" + IonVersion::GetBuildNum() + ")";

// Set default values
  string ref_file  = "";
  string input_bams = "";
  vector<string> targets;
  string hotspot = "";
  string out_alt_family = "";
  string out_family_counts = "";
  string out_global_counts = "";
  string out_var_calls = "";
  string out_stats = "";
  string tag_level = "double";
  string tag_method = "in-BAM"; // nobc(0), ampliseq(1), bcprimer(2), preamp (3), in-BAM(4)
  string sample_name = "SAMPLE";
  int vc_min_fam_size = 3;
  int vc_min_func_cov = 3;
  int vc_min_num_fam = 2;
  float vc_min_func_maf = 0.0005;
  int hp_max_length = 5;
  int min_map_qv = 4;
  int max_var_amp = 2;
  bool hs_mask_only = false;
  
// Read input options
  int acnt = 1;
  while(acnt<argc)
  {
   if(strcmp(argv[acnt], "--fasta")==0 )      { ref_file = (++acnt) < argc ? argv[acnt] : ref_file; }
   else
   if(strcmp(argv[acnt], "--bam")==0 )        { input_bams = (++acnt) < argc ? argv[acnt] : input_bams; }
   else
   if(strcmp(argv[acnt], "--target")==0 )     { if(++acnt < argc) targets.push_back(argv[acnt]);} 
   else
   if(strcmp(argv[acnt], "--hotspot")==0 )    { hotspot = (++acnt) < argc ? argv[acnt] :  hotspot; }
   else
   if(strcmp(argv[acnt], "--oALT")==0 )       { out_alt_family = (++acnt) < argc ? argv[acnt] : out_alt_family;}
   else
   if(strcmp(argv[acnt], "--ofamily")==0 )    { out_family_counts = (++acnt) < argc ? argv[acnt] : out_family_counts;}
   else
   if(strcmp(argv[acnt], "--oglobal")==0 )    { out_global_counts = (++acnt) < argc ? argv[acnt] : out_global_counts;}
   else
   if(strcmp(argv[acnt], "--ostats")==0 )     { out_stats = (++acnt) < argc ? argv[acnt] : out_stats;}
   else
   if(strcmp(argv[acnt], "--ocalls")==0 )     { out_var_calls = (++acnt) < argc ? argv[acnt] : out_var_calls;}
   else
   if(strcmp(argv[acnt], "--tag-method")==0 ) { tag_method = (++acnt) < argc ? argv[acnt] : tag_method;}
   else
   if(strcmp(argv[acnt], "--sample-name")==0 ){ sample_name = (++acnt) < argc ? argv[acnt] : sample_name;}
   else
   if(strcmp(argv[acnt], "--tag-level")==0 )    { tag_level = (++acnt) < argc ? argv[acnt] : tag_level;}
   else
   if(strcmp(argv[acnt], "--fam-size")==0 )     { vc_min_fam_size = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : vc_min_fam_size;}
   else
   if(strcmp(argv[acnt], "--func-cov")==0 )     { vc_min_func_cov = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : vc_min_func_cov;}
   else
    if(strcmp(argv[acnt], "--min-num-fam")==0 )   { vc_min_num_fam = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : vc_min_num_fam;}
   else
   if(strcmp(argv[acnt], "--func-maf")==0 )       { vc_min_func_maf = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atof(argv[acnt]) : vc_min_func_maf;}
   else
   if(strcmp(argv[acnt], "--hp-max-length")==0 )  { hp_max_length = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : hp_max_length;}
   else
   if(strcmp(argv[acnt], "--min-map-qv")==0 )     { min_map_qv = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : min_map_qv;}
   else
   if(strcmp(argv[acnt], "--max-var-amp")==0 )    { max_var_amp = ((++acnt) < argc && isdigit(argv[acnt][0])) ? atoi(argv[acnt]) : max_var_amp;}
   else
   if(strcmp(argv[acnt], "--a-handle-on")==0 )    { is_correct_adaptor_trimming = false;}
   else	
   if(strcmp(argv[acnt], "--hs-mask-only")==0 )   { hs_mask_only = true;}
   else	   
		{
           cerr << endl << "\nUnknown parameter " << argv[acnt] << endl; 
           print_usage(1, full_version_string);
         }
    ++acnt;
  }

  if(input_bams.empty()){
           cerr << endl << "\nMissing input BAM file. Use --bam. " << endl; 
           print_usage(1, full_version_string);
 }
 
  bool use_one_index = (tag_level.compare("single")==0)?true:false;
  int tagging_method = (tag_method.compare("nobc")==0?0:
                       (tag_method.compare("ampliseq")==0?1:
                       (tag_method.compare("bcprimer")==0?2:
                       (tag_method.compare("preamp")==0?3:
                       (tag_method.compare("in-BAM")==0?4:
                       (-1))))));

  if(tagging_method==-1)  {cerr << tag_method << " unsupported tagging method. available options are: ampliseq, bcprimer, preamp, in-BAM." << endl; print_usage(1, full_version_string); }
 
 if(out_family_counts.empty()){cerr << "Missing output filename for family size info. Use --ofamily. " << endl;print_usage(1, full_version_string); }
 if(out_global_counts.empty()){cerr << "Missing output filename for global allele counts. Use --oglobal. " << endl;print_usage(1, full_version_string); }

  cout << "tmol version: " << full_version_string << endl;
  cout << "tagging method: " << tag_method << endl;
  cout << "tagging level: " << tag_level << endl << endl; 

   long TARGET_SIZE = -1;
 VCFList * target = NULL;
  
 if(targets.size()>0) 
 {
 
  for(unsigned int i=0;i<targets.size();i++)
  {
   cout << "\nloading target file ... " << targets.at(i) << endl;
   VCFList * tmp_target = new VCFList();
   VCFList * merged_target = new VCFList();
   tmp_target->load_bed_file(targets.at(i));
   
   //merging target is required for correct target size calculation and intersection of multiple BEDs
   //get_target_vcf supports BEDs with overlapping contigs
   
   cout << "merging overlapping targets ... " << targets.at(i) << endl;
   TARGET_SIZE = tmp_target->merge_overlapping_segments(merged_target);
   cout << "merged size: "<< TARGET_SIZE << endl;
   
   delete tmp_target;
   if(i==0) target = merged_target; else 
   {
    tmp_target = new VCFList();
    cout << "intersect previous target with " << targets.at(i) << endl;
    TARGET_SIZE = target->intersect(merged_target,tmp_target);
    cout << "resulting target size: " << TARGET_SIZE << endl;
    delete target;
    delete merged_target;
    target = tmp_target;
   }
  }

 } else{
           cerr << endl << "\nMissing target file. Use --target. " << endl; 
           print_usage(1, full_version_string);
 }
  
  VCFList * varVC = NULL;
  if(!hotspot.empty()){

   cout << "\nLoading hotspot positions: " << hotspot << endl;
            varVC = new VCFList();
            varVC->load_file(hotspot, true, true);
 // intersect variant list with target regions 
   cerr << "extract on-target vcf records ... " << endl;  
   if(varVC!=NULL)
   {
    VCFList * _varVC = new VCFList();
    long vcrecords = target->get_target_vcf(varVC, _varVC);
    cout << "hotspot-vcf records on target .. " << vcrecords << endl;
    delete varVC;
    varVC = _varVC;    
   } 
 } 

 cout << "\nLoading reference: " << ref_file << endl;
 FASTA reference;
 if(!ref_file.empty()){
 reference.load_file(ref_file);
 VCFList * tmp_target = new VCFList();
 target->add_ref_seq(tmp_target, reference);
 delete target;
 target = tmp_target; 
 reference.contig.clear(); //removed contig sequence, keep names for later
 }else{
           cerr << endl << "\nMissing reference file. Use --fasta. " << endl; 
           print_usage(1, full_version_string);
 }
 
  ofstream alt_fam, global_cons, all_fam, fam_sz_dist, targ_fam_sz, var_calls;
 
  alt_fam.open(out_alt_family.c_str());
  if(!alt_fam) { cerr << "Error writing output file: " << out_alt_family << endl; }
  else alt_fam <<  "#CHROM\tPOS\tALT_COUNTS\tFAMILY_SIZE\tREF\tALT\tTOTAL_COV" << endl;
  
  all_fam.open((out_family_counts).c_str());
  if(!all_fam) { cerr << "Error writing output file: " << out_family_counts << endl; }
  else   all_fam << "# List of families of size >= 3 per region" << endl;
  
  fam_sz_dist.open((out_family_counts+".dist.txt").c_str());
  if(!fam_sz_dist) { cerr << "Error writing output file: " << out_family_counts << "dist.txt" << endl; }
  else   fam_sz_dist << "#family_size\tcounts" << endl;
  
  targ_fam_sz.open((out_family_counts+".targ.txt").c_str());
  if(!targ_fam_sz) { cerr << "Error writing output file: " << out_family_counts << "targ.txt"<< endl; }
  else  targ_fam_sz << "#CHROM_POS\tREADS_ON_TARGET\tREADS_WITH_P1\tNUMBER_FAMILIES_SIZE_L3\tNUMBER_FAMILIES_SIZE_L4\tNUMBER_FAMILES_SIZE_L7\tNUMBER_ALT_FAMILIES_SIZE_L3\tNUMBER_ALT_FAMILIES_SIZE_L4\tNUMBER_ALT_FAMILES_SIZE_L7\tNUMBER_EXCLUDED_FAM" << endl;
  
  global_cons.open(out_global_counts.c_str());
  if(!global_cons) { cerr << "Error writing output file: " << out_global_counts << endl; }
  else {
	global_cons << "##fileformat=VCFv4.1" << endl;
    global_cons << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">" << endl;
    global_cons << "##FORMAT=<ID=ALT,Number=A,Type=Integer,Description=\"Counts for alternative alleles A,C,G,T\">" << endl;
    global_cons << "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">" << endl;
    global_cons << "##INFO=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">" << endl;
    global_cons << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">" << endl;
	global_cons << "##Tagged Molecule Caller tool v." << full_version_string  << endl;
	global_cons << "##Tagging Method=None"  << endl;
	global_cons << "##Tagging Level=None" << endl;
	global_cons << "##BAM=" << input_bams  << endl;
	global_cons << "##BED=" << targets[0]  << endl;
	global_cons << "##HOTSPOT=" << hotspot << endl;
    global_cons << "##PARAMETERS=<Min Number of Reads supporting Variant=7, Min Coverage=15000,Min ALT Frequency=" << vc_min_func_maf <<">" << endl;
    global_cons << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" << sample_name << endl;

  }
  
  var_calls.open(out_var_calls.c_str());
  if(!var_calls) { cerr << "Error writing output file: " << out_var_calls << endl; }
  else 
  {  
    var_calls << "##fileformat=VCFv4.1" << endl;
    var_calls << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">" << endl;
    var_calls << "##FORMAT=<ID=FS,Number=A,Type=Integer,Description=\"Counts for families of size min_size, min_size+1, min_size+2,....\">" << endl;
    var_calls << "##FORMAT=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">" << endl;
    var_calls << "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">" << endl;
    var_calls << "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">" << endl;
	var_calls << "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">" << endl;
    var_calls << "##INFO=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">" << endl;
    var_calls << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on all reads\">" << endl;
    var_calls << "##INFO=<ID=MDP,Number=1,Type=Integer,Description=\"Molecular depth\">" << endl;
    var_calls << "##INFO=<ID=MAO,Number=A,Type=Integer,Description=\"Molecular counts for alternative allele. Number of molecules containing alternative allele.\">" << endl;
    var_calls << "##INFO=<ID=MAF,Number=A,Type=Float,Description=\"Molecular frequency of alternative allele. Ratio between the number of molecules containing alternative allele and the number of other molecules at that location\">" << endl;
	var_calls << "##INFO=<ID=LOD,Number=A,Type=Float,Description=\"Limit of Detection at genomic location. This number is calculated based on the number of identified unique molecules.\">" << endl;
	var_calls << "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"The type of allele, either snp, mnp, ins, del, or complex.\">" << endl;
	var_calls << "##INFO=<ID=HS,Number=0,Type=Flag,Description=\"Indicate it is at a hot spot\">" << endl;
	var_calls << "##INFO=<ID=FR,Number=.,Type=String,Description=\"Reason why the variant was filtered.\">" << endl;
	var_calls << "##Tagged Molecule Caller tool v." << full_version_string  << endl;
	var_calls << "##Tagging Method=" << tag_method  << endl;
	var_calls << "##Tagging Level=" << tag_level << endl;
	var_calls << "##BAM=" << input_bams  << endl;
	var_calls << "##BED=" << targets[0]  << endl;
	var_calls << "##HOTSPOT=" << hotspot << endl;	
    var_calls << "##PARAMETERS=<Min Functional Family Size=" << vc_min_fam_size << ",Min Number of Variant Families="<< vc_min_num_fam <<",Min Functional Coverage=" << vc_min_func_cov << ",Min Functional ALT Frequency=" << vc_min_func_maf <<">" << endl;
    var_calls << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" << sample_name << endl;
  }

  	unsigned * sz_dist = new unsigned[31];
	for(int i=0;i<31;i++) sz_dist[i]=0;

  
  cout << "\nStarting analysis... " << endl;
  unsigned num_targets = 0;
  unsigned long total_reads_on_target = 0;
  unsigned long total_reads_p1 = 0;
  vector<unsigned> reads_per_target;
  vector<unsigned> fam_sz3;
  vector<unsigned> fam_sz7;
  vector<double> cov_drop_middle;
  vector<double> cov_drop_end;
  unsigned cov_start, cov_middle, cov_end;
  
  std::stringstream bam_list(input_bams);
  string input_bam, input_bam_index;
  while(bam_list.good())
  {
  getline( bam_list, input_bam, ',');
  
  BamReader reader;
  if (!reader.Open(input_bam)) {
    cerr << "Failed to open input file " << input_bam << endl;
    return 1;
  }
   
  input_bam_index = input_bam + ".bai";
  if (!reader.OpenIndex(input_bam_index)) {
    cerr << "Failed to open input bam index file " << input_bam_index << ". Read access is slower!" << endl;
  }
  
  std::map<string,std::map<long int,VCFinfo> > * targetrec =  target->getList();
  for(unsigned int ch=0;ch<reference.name.size();ch++)
  {	  
   //for(std::map< string , std::map<long int,VCFinfo> >::iterator chrit = targetrec->begin(); chrit != targetrec->end(); ++chrit) 
   std::map< string , std::map<long int,VCFinfo> >::iterator chrit = targetrec->find(reference.name[ch]);
   if(/*reference.name[ch].compare("chr11")==0 && */chrit!=targetrec->end()) for(std::map<long int,VCFinfo>::iterator posit = chrit->second.begin(); posit != chrit->second.end(); posit++)
   {
	posit->second.parse_amplicon_names();
	string ampl_name = posit->second.get_str_flag_value(0,"AMP_NAME");
	string gene_id = posit->second.get_str_flag_value(0,"GENE_ID");
	
	int32_t refid = reference.get_contig_idx(chrit->first);
	int32_t start_pos = posit->first; // second end of amplicon
	int32_t end_pos = posit->second.get_zyg();
    int target_size = end_pos - start_pos;

	vector<long int> hotspot_pos;
	vector<VCFinfo> hotspot_rec;
	vector<std::map< int , string > > read_filter;
	int local_max_var_amp = max_var_amp;
	
	if(varVC != NULL){
		std::map<string,std::map<long int,VCFinfo> > * vcfrec =  varVC->getList();
		std::map< string , std::map<long int,VCFinfo> >::iterator chrVC = vcfrec->find(reference.name[ch]);
	    if(chrVC!=vcfrec->end()) 
			for(std::map<long int,VCFinfo>::iterator vcfpos = chrVC->second.begin(); vcfpos != chrVC->second.end(); vcfpos++)
		    if(vcfpos->first > start_pos && vcfpos->first < end_pos){
			 hotspot_pos.push_back(vcfpos->first);
  			 vcfpos->second.parse_info_values();
 			 hotspot_rec.push_back(vcfpos->second);
			 if(!load_read_filters( &read_filter, vcfpos->second)) local_max_var_amp = 100;
		    }
	if(!hs_mask_only && hotspot_pos.size()==0) continue;
	}
	
	string ref_seq = posit->second.get_ref(0);
	string tgdef2 = gene_id + "_" + ampl_name;
	if(gene_id.empty() && ampl_name.empty()) tgdef2 = chrit->first + "_" + itos(start_pos);
    string tgdef = chrit->first + "\t" + itos(start_pos) + "\t" + itos(end_pos) + "\t" + tgdef2 + "\tNREADS\t";
	
    cout << "evaluating:\t" << tgdef; cout.flush();

	
    vector<BamAlignment> * buffer = new std::vector<BamAlignment>(10000);  
    load_region_reads(reader, buffer, input_bam, refid, start_pos, end_pos, min_map_qv);
    cout <<  buffer->size() << endl;
	
    total_reads_on_target += buffer->size(); 
    reads_per_target.push_back(buffer->size());
	tgdef += itos(buffer->size());
  
    unsigned char * index_counts = new unsigned char[INT_MAX/125];
    for(int i=0;i<INT_MAX/125;i++) index_counts[i]=0;
	
    unsigned long local_total_reads_p1 = 0;
    
    build_index_table(buffer, index_counts, tagging_method, local_total_reads_p1, use_one_index);
    total_reads_p1 += local_total_reads_p1;

  unsigned * global_consensus = new unsigned[12*target_size+12];
  for(int i=0;i<12*target_size+12;i++) global_consensus[i]=0;
  
  map<unsigned,PileUp *> * top_family = consensus_analysis(ref_seq, buffer, index_counts, global_consensus, & read_filter, refid, start_pos, end_pos, tagging_method, vc_min_fam_size, use_one_index);
  cov_start = 0; cov_middle = 0;cov_end = 0;
  for(int i=0;i<12;i++) {
	  cov_start+=global_consensus[i];
	  cov_middle+=global_consensus[6*target_size+i];
	  cov_end+=global_consensus[12*target_size-i];
  }
  
  if(cov_start < cov_end) {unsigned tmp = cov_start; cov_start = cov_end; cov_end = tmp; }
  
  cov_drop_middle.push_back(((double)cov_middle)/((double)cov_start));
  cov_drop_end.push_back(((double)cov_end)/((double)cov_start));
  
  unsigned n_fam3=0;
  unsigned n_fam7=0;	

  // This one should be fine.
  report_target_family_counts(index_counts,tagging_method,tgdef,all_fam, sz_dist, n_fam3, n_fam7);
  fam_sz3.push_back(n_fam3);
  fam_sz7.push_back(n_fam7);  
  // This one, too.
  report_consensus_counts(ref_seq, chrit->first,top_family, global_consensus, tagging_method, refid, start_pos, end_pos, alt_fam, global_cons, targ_fam_sz, var_calls, hotspot_pos, hotspot_rec, buffer->size(), local_total_reads_p1, vc_min_fam_size, vc_min_func_cov, vc_min_func_maf, vc_min_num_fam, tgdef2, local_max_var_amp, hs_mask_only);
 
  delete [] index_counts;
  delete [] global_consensus;
  for (std::map<unsigned,PileUp *>::iterator ret = top_family->begin() ; ret != top_family->end(); ++ret) delete [] ret->second;
  delete top_family;
  delete buffer; 
   
  num_targets++;
  //if(num_targets >=3) break;
  }
  //if(num_targets >=3) break; 
 }
}

 for(int i=3;i<30;i++) fam_sz_dist << i << "\t" << sz_dist[i]<<endl;
 fam_sz_dist << ">=30" << "\t" << sz_dist[30];
 delete [] sz_dist;
 
 all_fam.close();
 alt_fam.close();
 global_cons.close();
 fam_sz_dist.close();
 targ_fam_sz.close();
 
  ofstream ostats;
  ostats.open(out_stats.c_str());
  if(!ostats) { cerr << "Error writing output file: " << out_stats << endl; } 
  else
  {
  ostats << "Number of targets:\t" << num_targets << endl;
  ostats << "Total number of reads on target:\t" << total_reads_on_target << endl;
  ostats << "Total number of reads on target with P1-adaptor:\t" << total_reads_p1 << endl;
  std::sort(reads_per_target.begin(), reads_per_target.end());
  unsigned median_reads_per_target = reads_per_target.at(reads_per_target.size()/2);
  ostats << "Median read counts per target:\t" << median_reads_per_target << endl;
  ostats << "median_reads_per_target:\t" << median_reads_per_target << endl;
  ostats.precision(4);
  double ampl_read_uniformity = 0;
  for(unsigned int i=0;i<reads_per_target.size();i++) ampl_read_uniformity += ((double)abs((long)reads_per_target[i]-median_reads_per_target))/((double)median_reads_per_target);
  ampl_read_uniformity = 1.0 - ampl_read_uniformity/reads_per_target.size();
  if(ampl_read_uniformity<0) ampl_read_uniformity = 0; 
  ostats << "Uniformity of read counts across targets:\t" << 100.00*ampl_read_uniformity << "%"<< endl;
  sort(cov_drop_middle.begin(), cov_drop_middle.end());
  sort(cov_drop_end.begin(), cov_drop_end.end());
  ostats << "Median percent coverage at middle of target relative to start:\t" << 100.00*cov_drop_middle.at(cov_drop_middle.size()/2) << "%" << endl;
  ostats << "Median percent coverage at end of target relative to start:\t" << 100.00*cov_drop_end.at(cov_drop_end.size()/2) << "%" << endl;
  sort(fam_sz3.begin(), fam_sz3.end());
  unsigned median_fam_sz3 = fam_sz3.at(fam_sz3.size()/2);
  ostats << "median_num_fam3:\t" << median_fam_sz3 << endl;
  sort(fam_sz7.begin(), fam_sz7.end());
  unsigned median_fam_sz7 = fam_sz7.at(fam_sz7.size()/2);
  ostats << "median_num_fam7:\t" << median_fam_sz7 << endl;
   double fam3_uniformity = 0;
  for(unsigned int i=0;i<fam_sz3.size();i++) fam3_uniformity += ((double)abs((long)fam_sz3[i]-median_fam_sz3))/((double)median_fam_sz3);
  fam3_uniformity = 1.0 - fam3_uniformity/fam_sz3.size();
  ostats << "unif_fm3:\t" << 100.00*fam3_uniformity << "%" << endl;
   double fam7_uniformity = 0;
  for(unsigned int i=0;i<fam_sz7.size();i++) fam7_uniformity += ((double)abs((long)fam_sz7[i]-median_fam_sz7))/((double)median_fam_sz7);
  fam7_uniformity = 1.0 - fam7_uniformity/fam_sz7.size();
  ostats << "unif_fm7:\t" << 100.00*fam7_uniformity << "%" << endl;
  double percent_balced_fam3 = 0;
  for(unsigned int i=0;i<fam_sz3.size();i++) if(fam_sz3[i]<=median_fam_sz3*2.0 && fam_sz3[i]>0.5*median_fam_sz3) percent_balced_fam3++;
  percent_balced_fam3 = 100.00*percent_balced_fam3/fam_sz3.size();
  ostats << "fm3_balance:\t"<< percent_balced_fam3 << "%" << endl;  
  percent_balced_fam3=0;
 for(unsigned int i=0;i<fam_sz3.size();i++) if(fam_sz3[i]>0.8*median_fam_sz3) percent_balced_fam3++;
  percent_balced_fam3 = 100.00*percent_balced_fam3/fam_sz3.size();
  ostats << "fm3_pass80:\t"<< percent_balced_fam3 << "%" << endl;  

    double percent_balced_fam7 = 0;
  for(unsigned int i=0;i<fam_sz7.size();i++) if(fam_sz7[i]<=median_fam_sz7*2.0 && fam_sz7[i]>0.5*median_fam_sz7) percent_balced_fam7++;
  percent_balced_fam7 = 100.00*percent_balced_fam7/fam_sz7.size();
  ostats << "fm7_balance:\t"<< percent_balced_fam7 << "%" << endl;   
  ostats << "Tagging Cost Efficiency:\t" << 100.00*((double)(median_fam_sz3 - median_fam_sz7))/((double)median_fam_sz3) << "%" << endl;
  ostats.close();
  }
  
}
//-----------------------------------------------------
