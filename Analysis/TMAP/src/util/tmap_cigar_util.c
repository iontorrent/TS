#include "tmap_cigar_util.h"
#include "../sw/tmap_sw.h"
#include "samtools/bam.h"
#include <string.h>
#include <ctype.h>
#include <assert.h>

static const char ellipsis_ [] = "..";
static const char _CIGAR_OP2CHAR [] = BAM_CIGAR_STR;
static char _CIGAR_CHAR2OP [128];
static uint8_t _CIGAR_translator_init = 0;

#define lv(A,B) (((A)<(B))?(A):(B))

uint32_t decimal_positions (uint32_t value)
{
    uint32_t pos = 0;
    do
        value /= 10, ++pos;
    while (value);
    return pos;
}
__attribute__ ((constructor))
static void cigar_translator_init ()
{
    memset (_CIGAR_CHAR2OP, 0xff, sizeof (_CIGAR_CHAR2OP));
    for (int idx = 0; idx != sizeof (_CIGAR_OP2CHAR); ++idx)
        _CIGAR_CHAR2OP [(int) _CIGAR_OP2CHAR [idx]] = idx;
    _CIGAR_translator_init = 1;
}


uint32_t compute_cigar_strlen
(
    const uint32_t* cigar,
    size_t cigar_sz
)
{
    const uint32_t* sent;
    size_t buf_fill = 0;
    for (sent = cigar + cigar_sz; cigar != sent; ++cigar)
        buf_fill += decimal_positions (bam_cigar_oplen (*cigar)) + 1; // len+op
    return buf_fill;
}

uint32_t 
compute_cigar_bin_size
(
    const char* cigar
)
{
    uint32_t opno = 0;
    if (cigar == NULL)
        return UINT32_MAX;
    uint8_t in_char = 1; // so that next expected char is digit
    for (; *cigar; ++cigar)
    {
        if (isdigit (*cigar))
        {
            if (in_char)
            {
                in_char = !in_char;
            }
        }
        else if (strchr (_CIGAR_OP2CHAR, *cigar))
        {
            if (in_char) // two consequtive operation chars 
                return UINT32_MAX;
            else
            {
                ++ opno;
                in_char = !in_char;
            }
        }
        else // invalid char
            return UINT32_MAX;
    }
    if (!in_char) // does not end with operation char
        return UINT32_MAX;
    return opno;
}

uint8_t next_cigar_pos
(
    const uint32_t* cigar,
    uint32_t ncigar,
    int32_t* op_idx,
    int32_t* op_off,
    int8_t increment // 1 or -1
)
{
    assert (op_idx && op_off);
    int32_t cur_idx = *op_idx;
    int32_t cur_off = *op_off;

    // no ope pase ncigar
    if (cur_idx > ncigar) 
        return 0;
    // at ncigar op decrement is only allowed on 0 pos 
    if (cur_idx == ncigar && !(increment == -1 && cur_off == 0))
        return 0;

    if (increment == 1)
    {
        uint32_t oplen = bam_cigar_oplen (cigar [cur_idx]);
        assert ((cur_off < (int32_t) oplen) || (cur_off == (int32_t) oplen && cur_idx + 1 == (int32_t) ncigar));
        if (cur_off == oplen && cur_idx == ncigar - 1)
            return 0; // attempt to increment last centinel
        ++cur_off;
        while (cur_off == oplen) // this is to skip zero-length cigar ops
        {
            if (cur_idx + 1 == ncigar) // incremented to centinel
                break;
            else
            {
                ++cur_idx;
                oplen = bam_cigar_oplen (cigar [cur_idx]);
                cur_off = 0;
            }
        }
    }
    else // decrement
    {
        assert ((cur_off >= 0) || (cur_off == -1 && cur_idx == 0));
        if (cur_off == -1)
            return 0;
        while (cur_off == 0)
        {
            if (cur_idx == 0)
            {
                break;
            }
            else
            {
                --cur_idx;
                cur_off = bam_cigar_oplen (cigar [cur_idx]);
            }
        }
        --cur_off;
    }
    *op_idx = cur_idx;
    *op_off = cur_off;
    return 1;

}



uint32_t
cigar_to_string
(
    const uint32_t* cigar,
    size_t cigar_sz,
    char* buffer,
    size_t bufsz
)
{
    if (!bufsz) return 0;
    assert (buffer);
    const uint32_t* sent;
    size_t buf_fill = 0, last_complete_fill = 0;
    for (sent = cigar + cigar_sz; cigar != sent && buf_fill < bufsz; ++cigar)
    {
        uint32_t curop = bam_cigar_op (*cigar);
        uint32_t count = bam_cigar_oplen (*cigar);
        char schar = (curop < sizeof (_CIGAR_OP2CHAR))? (_CIGAR_OP2CHAR [curop]) : ('?');

        buf_fill += snprintf (buffer + buf_fill, bufsz - buf_fill, "%d%c", count, schar);

        if (bufsz > buf_fill + (sizeof (ellipsis_) - 1)) // sizeof (ellipsis_) - 1 is same as strlen (ellipsis) but avoids actual memory scan
            last_complete_fill = buf_fill;
    }
    if (buf_fill >= bufsz) // add ellipsis after last_complete_fill position
    {
        strncpy (buffer + last_complete_fill, ellipsis_, bufsz - last_complete_fill);
        buf_fill = lv ((bufsz - 1), (last_complete_fill + (sizeof (ellipsis_) - 1)));
    }
    buffer [buf_fill] = 0; // this covers the case of zero length (empty) cigar
    return buf_fill;
}


static const size_t _NUMBUF_SZ = 16;

uint32_t
string_to_cigar
(
    const char* cigar_str,
    uint32_t* buf,
    size_t bufsz
)
{
    char numbuf [_NUMBUF_SZ];
    uint8_t numsz;
    uint32_t bufpos = 0;
    uint32_t oplen;

    assert (_CIGAR_translator_init);

    while (*cigar_str && bufpos != bufsz)
    {
        numsz = 0;
        if (!isdigit (*cigar_str))
            // malformed cigar does not start with digit. abort
            return 0;
        while (isdigit (*cigar_str) && numsz  != _NUMBUF_SZ-1)
            numbuf [numsz++] = *(cigar_str++);
        numbuf [numsz] = 0;
        if (!*cigar_str) // malformed cigar ends with digit, no operation char. Do not add last operation
            break;
        if (!sscanf (numbuf, "%d", &oplen))
            // something is wrong, abort
            return 0;
        if (!strchr (_CIGAR_OP2CHAR, *cigar_str))
            // malformed cigar has wrong opcode, abort
            return 0;
        uint8_t op = _CIGAR_CHAR2OP [(int) *(cigar_str ++)];
        buf [bufpos++] = bam_cigar_gen (oplen, op);
    }
    return bufpos;
}

uint8_t cigar_footprint
(
    const uint32_t* cigar_bin,
    uint32_t cigar_bin_len,
    uint32_t* qlen,
    uint32_t* rlen,
    uint32_t* allen,
    uint32_t* left_clip,
    uint32_t* right_clip
)
{
    const uint32_t* sent;
    uint32_t op, oplen, constype;
    uint32_t _qlen = 0, _rlen = 0, _allen = 0, _left_clip = 0, _right_clip = 0;
    uint32_t left_clip_done = 0;
    uint8_t result = 1;
    for (sent = cigar_bin + cigar_bin_len; cigar_bin != sent; ++cigar_bin)
    {
        oplen = bam_cigar_oplen (*cigar_bin);
        op = bam_cigar_op (*cigar_bin);
        constype = bam_cigar_type (op);
        if (constype & CIGAR_CONSUME_QRY) _qlen += oplen;
        if (constype & CIGAR_CONSUME_REF) _rlen += oplen;
        switch (op)
        {
            case BAM_CSOFT_CLIP:
                 // right soft clip, 
                if (!left_clip_done)
                    _left_clip += oplen;
                else
                    _right_clip += oplen;
                break;
            case BAM_CMATCH:
            case BAM_CINS:
            case BAM_CDEL:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                left_clip_done = 1;
                if (_right_clip)
                    // this is a malformed cigar string, having inner softclips. 
                    // We do not validate it here, just treat inner clips as insertions
                    _right_clip = 0, result = 0;
            default:
                ;
            // all other cases just fall off
        }
        _allen += oplen;
    }
    if (qlen) *qlen = _qlen;
    if (rlen) *rlen = _rlen;
    if (allen) *allen = _allen;
    if (left_clip) *left_clip = _left_clip;
    if (right_clip) *right_clip = _right_clip;
    return result;
}
