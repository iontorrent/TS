#include "swmatrix_facet.h"

// IUPAC_MATRIX_SIZE is defined in tmap_sw.h, included through tmap_align_util.h

// ACGTNBDHKMRSVWYN
// This gives a mask for match iupac characters versus A/C/G/T/N
//  A  C  G  T   N  B  D  H   K  M  R  S   V  W  Y  N
static int32_t matrix_iupac_mask [IUPAC_MATRIX_SIZE] = {
    1, 0, 0, 0,  1, 0, 1, 1,  0, 1, 1, 0,  1, 1, 0, 1, // A
    0, 1, 0, 0,  1, 1, 0, 1,  0, 1, 0, 1,  1, 0, 1, 1, // C
    0, 0, 1, 0,  1, 1, 1, 0,  1, 0, 1, 1,  1, 0, 0, 1, // G
    0, 0, 0, 1,  1, 1, 1, 1,  1, 0, 0, 0,  0, 1, 1, 1, // T
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1  // N
};

static void populate_sw_par_iupac_direct (tmap_sw_param_t* par, int32_t score_match, int32_t pen_mm, int32_t pen_gapo, int32_t pen_gape, int32_t bw)
{
    int32_t i;
    for (i = 0; i < IUPAC_MATRIX_SIZE; ++i)
    {
        if (0 < matrix_iupac_mask [i]) par->matrix [i] = score_match;
        else par->matrix [i] = -pen_mm;
    }
    par->gap_open = pen_gapo;
    par->gap_ext =  pen_gape;
    par->gap_end =  pen_gape;
    par->row = IUPAC_MATRIX_ROWSIZE;
    par->band_width = bw;
}

// #define TEST_SWM_FACET_INPLACE_FILL

SwMatrixFacet::SwMatrixFacet ()
:
TestFacet ("SwMatrix")
{
    auto& wrapper = matrices [defname];
    wrapper.sw_param.matrix = wrapper.matrix;
    populate_sw_par_iupac_direct (&wrapper.sw_param, 1, 3, 5, 2, 50);
    wrapper.sw_param.matrix_owned = 0;
#ifdef TEST_SWM_FACET_INPLACE_FILL
    auto check = matrices.find (defname);
    assert (check != matrices.end ());
    assert (check->first == defname);
    assert (&check->second == &wrapper);
#endif 
}

const tmap_sw_param_t* SwMatrixFacet::sw_param(const char* name)
{
    auto found = matrices.find (name);
    if (found == matrices.end ())
        return nullptr;
    else
        return &(found->second.sw_param);
}
