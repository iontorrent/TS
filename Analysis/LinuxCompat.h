/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LINUXCOMPAT_H
#define LINUXCOMPAT_H

#ifndef _MSC_VER

const char* validate_fmt( const char* fmt, const char* file, int32_t line );
const char *validate_str(const char *src, int destsize, const char* file, int32_t line);

#define scanf_s( fmt, ... ) scanf( validate_fmt( fmt, __FILE__, __LINE__ ), __VA_ARGS__ )
#define fscanf_s( fp, fmt, ... ) fscanf( fp, validate_fmt( fmt, __FILE__, __LINE__ ), __VA_ARGS__ )
#define sprintf_s( s, size, fmt, ... ) snprintf( s, size, validate_fmt( fmt, __FILE__, __LINE__ ), __VA_ARGS__ )
#define fopen_s( fp, name, mode ) *(fp) = fopen(name, mode)
#define strcpy_s( dest, destsize, src ) strcpy( dest, validate_str(src, destsize, __FILE__, __LINE__) )

#endif 

#endif // LINUXCOMPAT_H

