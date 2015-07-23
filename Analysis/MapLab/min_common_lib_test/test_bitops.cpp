/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include "test_bitops.h"
#include <bitops.h>
#include <comma_locale.h>
#include <boost/utility.hpp>


bool TestBitops::process ()
{
    o_ << "BITS_PER_BYTE = " << BITS_PER_BYTE << "\n" <<
          "BITS_PER_BYTE_SHIFT = " << BITS_PER_BYTE_SHIFT << "\n" <<
          "BITS_PER_QWORD = " << BITS_PER_QWORD << "\n";
    TEST_ASSERT (BITS_PER_BYTE == 8);
    TEST_ASSERT (BITS_PER_BYTE_SHIFT == 3);
    TEST_ASSERT (BITS_PER_QWORD == 8*8);

    o_ << "bit_width <BYTE> = " << bit_width <BYTE>::v () << "\n" <<
           "bit_width <WORD> = " << bit_width <WORD>::v () << "\n" <<
           "bit_width <SWORD> = " << bit_width <SWORD>::v () << "\n" <<
           "bit_width <DWORD> = " << bit_width <DWORD>::v () << "\n" <<
           "bit_width <SDWORD> = " << bit_width <SDWORD>::v () << "\n" <<
           "bit_width <QWORD> = " << bit_width <QWORD>::v () << "\n" <<
           "bit_width <SQWORD> = " << bit_width <SQWORD>::v () << "\n" <<
           "bit_width <char> = " << bit_width <char>::v () << "\n" <<
           "bit_width <short> = " << bit_width <short>::v () << "\n" <<
           "bit_width <int> = " << bit_width <int>::v () << "\n" <<
           "bit_width <long> = " << bit_width <long>::v () << "\n" <<
           "bit_width <long long> = " << bit_width <long long>::v () << "\n";
    TEST_ASSERT (bit_width <BYTE>::width == 8);
    TEST_ASSERT (bit_width <WORD>::width == 16);
    TEST_ASSERT (bit_width <SWORD>::width == 16);
    TEST_ASSERT (bit_width <DWORD>::width == 32);
    TEST_ASSERT (bit_width <SDWORD>::width == 32);
    TEST_ASSERT (bit_width <QWORD>::width == 64);
    TEST_ASSERT (bit_width <SQWORD>::width == 64);
    TEST_ASSERT ((unsigned) bit_width <char>::width == (unsigned) bit_width <unsigned char>::width);
    TEST_ASSERT ((unsigned) bit_width <short>::width == (unsigned) bit_width <unsigned short>::width);
    TEST_ASSERT ((unsigned) bit_width <int>::width == (unsigned) bit_width <unsigned int>::width);
    TEST_ASSERT ((unsigned) bit_width <long>::width == (unsigned) bit_width <unsigned long>::width);
    TEST_ASSERT ((unsigned) bit_width <long long>::width == (unsigned) bit_width <unsigned long long>::width);
    TEST_ASSERT ((unsigned) bit_width <char>::width <= (unsigned) bit_width <short>::width);
    TEST_ASSERT ((unsigned) bit_width <short>::width <= (unsigned) bit_width <int>::width)
    TEST_ASSERT ((unsigned) bit_width <int>::width <= (unsigned) bit_width <long>::width)
    TEST_ASSERT ((unsigned) bit_width <long>::width <= (unsigned) bit_width <long long>::width)
    TEST_ASSERT ((unsigned) bit_width <int>::width >= 16);

    o_ << "ln_bit_width <BYTE> = " << ln_bit_width <BYTE>::v () << "\n" <<
           "ln_bit_width <WORD> = " << ln_bit_width <WORD>::v () << "\n" <<
           "ln_bit_width <SWORD> = " << ln_bit_width <SWORD>::v () << "\n" <<
           "ln_bit_width <DWORD> = " << ln_bit_width <DWORD>::v () << "\n" <<
           "ln_bit_width <SDWORD> = " << ln_bit_width <SDWORD>::v () << "\n" <<
           "ln_bit_width <QWORD> = " << ln_bit_width <QWORD>::v () << "\n" <<
           "ln_bit_width <SQWORD> = " << ln_bit_width <SQWORD>::v () << "\n" <<
           "ln_bit_width <char> = " << ln_bit_width <char>::v () << "\n" <<
           "ln_bit_width <short> = " << ln_bit_width <short>::v () << "\n" <<
           "ln_bit_width <int> = " << ln_bit_width <int>::v () << "\n" <<
           "ln_bit_width <long> = " << ln_bit_width <long>::v () << "\n" <<
           "ln_bit_width <long long> = " << ln_bit_width <long long>::v () << "\n";

    TEST_ASSERT (ln_bit_width <BYTE>::width == 3);
    TEST_ASSERT (ln_bit_width <WORD>::width == 4);
    TEST_ASSERT (ln_bit_width <DWORD>::width == 5);
    TEST_ASSERT (ln_bit_width <QWORD>::width == 6);

    #define xstr_(s) st_(s)
    #define st_(s) #s

    #define  TEST_VAL_LIT 00000000 01000111 11010000 00000111 10101010 00000000 11111111 10010010
    #define  TEST_VAL BOOST_BINARY_ULL ( TEST_VAL_LIT )

    o_ << "Using test value of " << xstr_(TEST_VAL_LIT) << " (";
    std::locale  svl = ((std::ostream&) o_).imbue (hexcomma_locale);
    o_ << std::hex << TEST_VAL << " hex, ";
    ((std::ostream&) o_).imbue (svl);
    o_ << std::dec << TEST_VAL << " decimal)\n";
    o_ << "Compile-time tests:\n" <<
          "  significant bits: " << (unsigned) significant_bits < TEST_VAL >::number << "\n" <<
          "  set bits:" << (unsigned) set_bits < TEST_VAL >::number << "\n";
    TEST_ASSERT (significant_bits < TEST_VAL >::number == 55);
    TEST_ASSERT (set_bits < TEST_VAL >::number == 25);

    QWORD test_val = TEST_VAL;
    o_ << "Run-time tests:\n" << 
          " significant bits: " << std::dec << (unsigned) count_significant_bits (test_val) << "\n" <<
          " set bits: " << std::dec << (unsigned) count_set_bits (test_val) << "\n" <<
          " unset bits: " << std::dec << (unsigned) count_unset_bits (test_val) << "\n";
    TEST_ASSERT (count_significant_bits (test_val) == 55);
    TEST_ASSERT (count_set_bits (test_val) == 25);
    TEST_ASSERT (count_unset_bits (test_val) == 64-25);

    o_ << "\nBit-wise  printout         : "; print_bits (test_val, o_);
    o_ << "\nReverse Bit-wise  printout : "; print_bits (test_val, o_, false);
    o_ << "\nByte-wise printout         : "; print_bytes (test_val, o_);
    o_ << "\nReverse Byte-wise printout : "; print_bytes (test_val, o_, false);

    o_ << "\nMask of 29 bits " << MASK(test_val, 29) << "\n";

    o_ << "Last set bit is " << std::dec << (unsigned) last_set (test_val);
    TEST_ASSERT (last_set (test_val) == 64-2);

    BYTE br = randval<BYTE> ();
    char cr = randval<char> ();
    WORD wr = randval<WORD> ();
    WORD sr = randval<short> ();
    DWORD dwr = randval<DWORD> ();
    DWORD lr = randval<long> ();
    QWORD qwr = randval<QWORD> ();
    long long llr = randval<long long> ();
    o_ << "\nRandoms: byte: " << (int) br << ", char: " << (int) cr << ", word: " << wr << ", short: " << sr << ", dword: " << dwr<< ", long: " << lr << ", qword: " << qwr << ", longlong: " << llr  << std::endl;
    return true; 
}
