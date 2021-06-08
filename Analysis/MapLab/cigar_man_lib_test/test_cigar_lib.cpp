#include "test_cigar_lib.h"
#include <TMAP/src/util/tmap_cigar_util.h>
#include <cstdio>

bool TestCigarLib::init ()
{
    return true;
};

TestCigarTranslator::TestCigarTranslator ()
:
Test ("CigarTranslator - cigar string and binary size estimation and conversions")
{
}

bool TestCigarTranslator::process ()
{
    typedef struct _T
    {
        const char* cigar;
        uint32_t opno;
        uint8_t syntax_ok;
        uint8_t semantics_ok;
        uint8_t canonic;
    }
    T;

    T tests [] = {
    {"14S3M4I140M5D6M2I5M3S",    9, 1, 1, 1},
    {"4=2X12=1X14=3I4=2X4=1I2=", 11,1, 1, 1},
    {"10H3S50M5S1H",             5, 1, 1, 1},
    {"2M3I3D3I3I3D",             6, 1, 1, 0},
    {"201M3S20M",                3, 1, 0, 0},
    {"M33I2S",                   UINT32_MAX, 0, 0, 0},
    {"45S44M47W",                UINT32_MAX, 0, 0, 0}
    };


    const size_t MAX_CIGAR = 64;
    uint32_t cigar_buf [MAX_CIGAR];
    const size_t MAX_STR = 256;
    char cigar_str [MAX_STR];

    for (int idx = 0; idx != sizeof (tests) / sizeof (T); ++idx)
    {
        const char* init_cigar_str = tests [idx].cigar;
        uint32_t init_cigar_strlen = strlen (init_cigar_str);
        uint32_t cigar_opno = string_to_cigar (init_cigar_str, cigar_buf, MAX_CIGAR);
        uint32_t computed_opno = compute_cigar_bin_size (init_cigar_str);
        TEST_ASSERTX (computed_opno == tests [idx].opno, "cigar: %s, computed opno = %u, expected %u", init_cigar_str, computed_opno, tests [idx].opno);
        if (tests [idx].opno != UINT32_MAX)
        {
            TEST_ASSERTX (cigar_opno == tests [idx].opno, "cigar %s, converted opno = %u, expected %u", init_cigar_str, cigar_opno, tests [idx].opno);
        }
        else
        {
            TEST_ASSERTX (cigar_opno == 0, "cigar %s, converted opno = %u, expected %u", init_cigar_str, cigar_opno, 0);
            continue;
        }

        if (tests [idx].syntax_ok)
        {
            TEST_ASSERTX (cigar_opno != 0, "Conversion to binary failed on good syntax cigar: %s", init_cigar_str);
        }
        else
        {
            TEST_ASSERTX (cigar_opno == 0, "Conversion to binary did not fail on bad syntax cigar: %s", init_cigar_str);
        }
        if (tests [idx].syntax_ok && cigar_opno != 0)
        {
            uint32_t computed_cigar_strlen = compute_cigar_strlen (cigar_buf, cigar_opno);
            TEST_ASSERTX (init_cigar_strlen == computed_cigar_strlen, "Wrong computed len for binary cigar derived from %s: expected %d, got %d", init_cigar_str, init_cigar_strlen, computed_cigar_strlen);

            size_t bufsz = MAX_STR;
            uint32_t written = cigar_to_string (cigar_buf, cigar_opno, cigar_str, bufsz);
            TEST_ASSERTX (written == computed_cigar_strlen, "Conversion back to string failed, initial cigar is: %s", init_cigar_str);
            TEST_ASSERTX (0 == strcmp (init_cigar_str, cigar_str), "Conversion back to string returned not matching value, expected %s, got %s", init_cigar_str, cigar_str);
        }
    }
    return true;
}

TestDecimalPositions :: TestDecimalPositions ()
:
Test ("DecimalPositions - decimal_positions function")
{
}

bool TestDecimalPositions::process ()
{
    uint32_t values [] = {0, 9, 98, 987, 9876, 98765, 987654, 9876543, 98765432, 987654321};

    for (uint32_t *valuep = values, *sent = values + sizeof (values) / sizeof (*values); valuep != sent; ++valuep)
    {
        size_t expected = snprintf (NULL, 0, "%u", *valuep);
        size_t obtained = decimal_positions (*valuep); 
        TEST_ASSERTX (expected == obtained, "Failed on value = %d, expected len: %d, decimal_positions returned %d", *valuep, expected, obtained);
    }
    return true;
}

TestCigarFootprint :: TestCigarFootprint ()
:
Test ("CigarFootprint - cigar footprint by query and reference")
{
}

bool TestCigarFootprint::process ()
{
    struct CigarFootprintProps
    {
        const char* cigar;
        uint32_t qry_len;
        uint32_t ref_len;
        uint32_t al_len;
        uint32_t clip_left;
        uint32_t clip_right;
    };
    const CigarFootprintProps tests [] = 
    {
        {"40M", 40, 40, 40, 0, 0},
        {"5S50M20S", 75, 50, 75, 5, 20},
        {"8I1M18D", 9, 19, 27, 0, 0},
        {"1S2M3I4M5D6M7I8M9D10M11S", 52, 44, 66, 1, 11}
    };

    for (const CigarFootprintProps *ct = tests, *sent = ct + sizeof (tests) / sizeof (*tests); ct != sent; ++ct)
    {
        uint32_t qlen, rlen, alen, lclip, rclip;
        const uint32_t cigar_opno = compute_cigar_bin_size (ct->cigar);
        TEST_ASSERTX (cigar_opno != UINT32_MAX, "Skipping tests for cigar %s", ct->cigar);
        if (cigar_opno == UINT32_MAX)
            continue;
        uint32_t cigar_bin [cigar_opno];
        uint32_t cigar_bin_len = string_to_cigar (ct->cigar, cigar_bin, cigar_opno);
        TEST_ASSERT (cigar_bin_len == cigar_opno);
        uint8_t proper = cigar_footprint (cigar_bin, cigar_bin_len, &qlen, &rlen, &alen, &lclip, &rclip);
        TEST_ASSERTX (proper, "Error found while computing footprint for cigar %s", ct->cigar);
        TEST_ASSERTX (ct->qry_len == qlen, "in cigar %s, expected = %u, observed = %u", ct->cigar, ct->qry_len, qlen);
        TEST_ASSERTX (ct->ref_len == rlen, "in cigar %s, expected = %u, observed = %u", ct->cigar, ct->ref_len, rlen);
        TEST_ASSERTX (ct->al_len == alen, "in cigar %s, expected = %u, observed = %u", ct->cigar, ct->al_len, alen);
        TEST_ASSERTX (ct->clip_left == lclip, "in cigar %s, expected = %u, observed = %u", ct->cigar, ct->clip_left, lclip);
        TEST_ASSERTX (ct->clip_right == rclip, "in cigar %s, expected = %u, observed = %u", ct->cigar, ct->clip_right, rclip);
    }
    return true;
}

TestNextCigarPos :: TestNextCigarPos ()
:
Test ("NextCigarPos - switching to next and previous positions")
{
}

struct CigarNextPosProps
{
    const char* cigar;
    int32_t opidx;
    int32_t opoff;
    int8_t dir;
    int32_t expected_opidx;
    int32_t expected_opoff;
    uint8_t expected_result;
};

static size_t present_props_x (CigarNextPosProps* pp, uint8_t result, uint32_t opidx, uint32_t opoff, unsigned testno, char* buf, size_t bufsz)
{
    return snprintf (buf, bufsz, "Test %3d:  %-12s (%2d %2d) (dir %d) [res %d] -> (%2d %2d), exp (%2d %2d) [res %d]", testno, pp->cigar, pp->opidx, pp->opoff, pp->dir, result, opidx, opoff, pp->expected_opidx, pp->expected_opoff, pp->expected_result);
}

bool TestNextCigarPos::process ()
{
    const char* cigars [] = 
    {
        "40M",
        "5S50M20S",
        "8I1M18D4M10I10M3S"
    };
    CigarNextPosProps tests [] = 
    {
        {cigars [0], 0,  1,  1, 0,  2, 1},
        {cigars [0], 0,  0, -1, 0, -1, 1},
        {cigars [0], 0,  0,  1, 0,  1, 1},
        {cigars [0], 0, 39,  1, 0, 40, 1},
        {cigars [0], 0, 39, -1, 0, 38, 1},
        {cigars [0], 1,  0, -1, 0, 39, 1},  // special case of decrementing at the end of alignment
        {cigars [0], 1, 19,  1, 0,  0, 0},
        {cigars [0], 1, 19, -1, 0,  0, 0},
        {cigars [1], 1, 49,  1, 2,  0, 1},
        {cigars [1], 1, 49, -1, 1, 48, 1},
        {cigars [1], 3, 19,  1, 0,  0, 0},
        {cigars [1], 1, 0,  -1, 0,  4, 1},
        {cigars [2], 6, 0,  -1, 5,  9, 1}
    };

    for (CigarNextPosProps *pp = tests, *sent = tests + sizeof (tests)/sizeof (*tests); pp != sent; ++pp)
    {
        uint32_t ncigar = compute_cigar_bin_size (pp->cigar);
        uint32_t cigar [ncigar]; 
        string_to_cigar (pp->cigar, cigar, ncigar);
        int32_t opidx = pp->opidx;
        int32_t opoff = pp->opoff;
        uint8_t res = next_cigar_pos (cigar, ncigar, &opidx, &opoff, pp->dir);
        size_t ssz = present_props_x (pp, res, opidx, opoff, pp - tests + 1, NULL, 0);
        char repr [ssz+1];
        present_props_x (pp, res, opidx, opoff, pp - tests + 1, repr, ssz+1);
        TEST_ASSERTX (res == pp->expected_result, repr);
        if (res)
        {
            TEST_ASSERTX (opidx == pp->expected_opidx, repr);
            TEST_ASSERTX (opoff == pp->expected_opoff, repr)
        }
    }
    return true;
}

