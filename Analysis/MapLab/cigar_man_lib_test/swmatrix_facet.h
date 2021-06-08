#ifndef __swmatrix_facet_h__
#define __swmatrix_facet_h__

#include "../min_common_lib/test_facet.h"
#include "TMAP/src/sw/tmap_sw.h"
#include <map>
#include <functional>

class SwMatrixFacet : public TestFacet
{
    struct Wrapper
    {
        int32_t matrix [IUPAC_MATRIX_SIZE];
        tmap_sw_param_t sw_param;
    };
    class strless
    {
    public:
        bool operator () (const char* a, const char* b)
        {
            if (a == b) return false;
            if (!a) return true;
            if (!b) return true;
            return (strcmp (a, b) < 0); 
        }
    };
    typedef std::map<const char*, Wrapper, strless> MatrixDir;
    MatrixDir matrices;
    static constexpr char* defname = nullptr;
public:
    SwMatrixFacet ();
    const tmap_sw_param_t* sw_param (const char* name = defname);
};


#endif  // __swmatrix_facet_h__

