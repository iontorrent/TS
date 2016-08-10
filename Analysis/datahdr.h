/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DATAHDR_H
#define DATAHDR_H

#ifdef GNUC
#include <cinttypes>
#define UINT32 uint32_t
#define UINT16 uint16_t
#else
typedef unsigned int UINT32; 
typedef unsigned short UINT16; 
typedef unsigned int uint32_t; 
typedef unsigned short uint16_t; 
#endif


#define DEVICE_MAX_CHANS 4

struct _file_hdr
{
    UINT32 signature;
    UINT32 struct_version;
    UINT32 header_size;
    UINT32 data_size;
};

struct _expmt_hdr_v2
{
    UINT32 first_frame_time;  /* "seconds since 1970" format */

    UINT16 chip_ID;
    UINT16 frames_in_file;
    UINT32 sample_rate;
    UINT16 full_scale_voltage[DEVICE_MAX_CHANS];
    UINT16 channel_offset[DEVICE_MAX_CHANS];
    UINT16 electrode;
    UINT16 frame_interval;
};

struct _expmt_hdr_v3
{
    UINT32 first_frame_time;  /* "seconds since 1970" format */

    UINT16 rows;
    UINT16 cols;
    UINT16 channels;
    UINT16 interlaceType;
    UINT16 frames_in_file;
	UINT16 uncomp_frames_in_file;
    UINT32 sample_rate;
    UINT16 full_scale_voltage[DEVICE_MAX_CHANS];
    UINT16 channel_offset[DEVICE_MAX_CHANS];
    UINT16 electrode;
    UINT16 frame_interval;
};

struct _expmt_hdr_v4
{
    UINT32 first_frame_time;  /* "seconds since 1970" format */

    UINT16 rows;
    UINT16 cols;
    UINT16 x_region_size;
    UINT16 y_region_size;
    UINT16 frames_in_file;
	UINT16 uncomp_frames_in_file;
    UINT32 sample_rate;
    UINT16 channel_offset[DEVICE_MAX_CHANS];
    UINT16 hw_interlace_type; // for un-doing deinterlace if needed
    UINT16 interlaceType;  // set to 5 for now

//    UINT16 channels;
//    UINT16 full_scale_voltage[DEVICE_MAX_CHANS];
//    UINT16 electrode;
//    UINT16 frame_interval;
};

struct _expmt_hdr_cmp_frame
{
	UINT32 timestamp;
	UINT32 Compressed;
	UINT32 len;
	UINT32 Transitions;
	UINT32 total;
	UINT32 sentinel;
};

#undef UINT32
#undef UINT16

#endif // DATAHDR_H

