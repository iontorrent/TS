#include "test_tmap_binary_search.h"
#include <test_case.h>
#include <ctype.h>
#include <TMAP/src/util/tmap_bsearch.h>


TestTmapBinarySearch::TestTmapBinarySearch ()
:
Test ("TmapBinarySearch")
{
}

int lt_char (const void* k, const void* e)
{
    return (*(char *) k) < (*(char *) e);
}


bool TestTmapBinarySearch::process ()
{
    unsigned char test_data [] = "ACEFGIKLMPSTXZ";
    unsigned char buf [sizeof test_data];
    size_t test_data_size = sizeof (test_data)/sizeof (*test_data) - 1; // for char array only, exclude terminating zero

    TEST_ASSERT (test_data [test_data_size - 1] == 'Z');
    TEST_ASSERT (test_data [0] > 2);
    TEST_ASSERT (test_data [test_data_size - 1] < (unsigned char) 252);

    size_t tot_tests = 0;
    for (size_t upper_bound = 0; upper_bound <= test_data_size; ++upper_bound)
    {
        for (unsigned char key = test_data [0] - 2; key != test_data [test_data_size -1] + 2; ++key)
        {
            const unsigned char* expected_location = test_data;
            while (
                expected_location != test_data +  upper_bound
                && lt_char (expected_location, &key)
                )
                ++expected_location;

            unsigned const char* computed_location = (const unsigned char*) tmap_binary_search (&key, test_data, upper_bound, sizeof (*test_data), lt_char);

            int expected_off = expected_location - test_data;
            int computed_off = computed_location - test_data;

            memcpy (buf, test_data, upper_bound * sizeof (*test_data));
            buf [upper_bound] = 0;
            TEST_ASSERTX (computed_location == expected_location, "Data is %s (%d el), key is '%c' (%d), expected: '%c' (%d) at position %d, found '%c' (%d) at position %d", 
                                                                        buf, upper_bound,
                                                                        isprint (key)?key:'*', int (key), 
                                                                        isprint (*expected_location)?*expected_location:'*', int (*expected_location), expected_off, 
                                                                        isprint (*computed_location)?*computed_location:'*', int (*computed_location), computed_off);
            ++ tot_tests;
        }
    }
    o_ << name () << ":" << tot_tests << " invocations checked" << std::endl;

    return 1;
}
