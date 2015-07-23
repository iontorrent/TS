/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define _SINGLETON_INITIALIZERS_

#include <bitops.h>
#include <compile_time_macro.h>

#include "alphabet.h"
#include "translator.h"
#include "gencode.h"
#include "wmatrix_templ.h"


namespace genstr
{

static char nnsym_ [] = "AGCT";
static char nxsym_ [] = "AGCTN";
static char nisym_ [] = "AGCTRYSWKMBDHVN";
static char aasym_ [] = "ARNDCQEGHILKMFPSTWYVBZX*";

///////////////////////////////
// common constants
///////////////////////////////

const unsigned NUMBER_OF_BASES = sizeof (nnsym_) - 1;
const unsigned BITS_PER_BASE = significant_bits<NUMBER_OF_BASES>::number;


const unsigned NUMBER_OF_RESIDUES = sizeof (aasym_) - 1;
const unsigned BITS_PER_RESIDUE = 5;

///////////////////////////////
// standard nucleotide alphabet
///////////////////////////////

Alphabet <char> nucleotides;
class initNucleotides
{
public:
    initNucleotides ()
    {
        nucleotides.configure (nnsym_, sizeof (nnsym_) - 1);
    }
};
static initNucleotides __singl_Nucleotides_init__;

///////////////////////////////
// standard aminoacids alphabet
///////////////////////////////

Alphabet <char> aminoacids;
class initAminoacids
{
public:
    initAminoacids ()
    {
        aminoacids.configure (aasym_, sizeof (aasym_) - 1);
    }
};
static initAminoacids __singl_Aminoacids_init__;

///////////////////////////////
// standard translators
///////////////////////////////

AlphaFwd<char> aa2num (aminoacids);
AlphaFwd<char> nn2num (nucleotides);
AlphaRev<char> num2aa (aminoacids);
AlphaRev<char> num2nn (nucleotides);

/////////////////////////
// standard GeneticCodes
/////////////////////////

// following genetic code(s) are for base order "AGCT"
static const char _standard_gcode [] = "KKNNRRSSTTTTIMIIEEDDGGGGAAAAVVVVQQHHRRRRPPPPLLLL**YY*WCCSSSSLLFF";

GeneticCode standardGeneticCode (_standard_gcode);

///////////////////////////////////////////////////
// standard nucleotide comparison matrix (unitary)
///////////////////////////////////////////////////

WeightMatrix<char, int, float> stdNucMatrix;
class initStdNucMatrix
{
public:
    initStdNucMatrix ()
    {
        UnitaryMatrix <int, 4> m;
        stdNucMatrix.configure ( nucleotides.symbols (), nucleotides.size (), m.values () );
    }
};
static initStdNucMatrix __singl_NucMatrix_init__;
}

