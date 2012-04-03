/* fw_access.h:  Bootloader system call interface */

#if !defined(__FW_ACCESS_H__)
#define __FW_ACCESS_H__ 1

/* Define indices of the firmware trap entry points.  These numbers
 * are used both by NetComm firmware and by the operating system.
 */
#define NR_BSC 21            /* last used bootloader system call */

#define __BN_reset        0  /* reset and start the bootloader */
#define __BN_fwLog        3  /* collect (and optionally clear) firmware log */
#define __BN_program      4  /* program FLASH from a chain */
#define __BN_eraseFlash   5  /* erase sector(s) of FLASH */
#define __BN_envmode      6  /* set environment mode */
#define __BN_writeFlash   7  /* copy 16-bit words to flash */
#define __BN_checkFlash   8  /* Check flash accessibility (program/erase/read) */
#define __BN_flashConfig  9  /* Retrieve flash configuration */
#define __BN_getmacaddr   12 /* get the hardware address of my interfaces */
#define __BN_getserialnum 13 /* get the serial number of this board */
#define __BN_getbenv      14 /* get a bootloader environment variable */
#define __BN_setbenv      15 /* set a bootloader environment variable */
#define __BN_readbenv     17 /* indexed read environment variables */

/* define the indices of the fixed environment variables.  These
 * indices are used only in the firmware.
 */
#define ENV_FWVERSION     0  /* NetComm firmware version */
#define ENV_HWVERSION     1  /* NetComm hardware version */
#define ENV_CONSOLE       2  /* Linux console name */
#define ENV_MACADDRESS    3  /* Ethernet MAC address */
#define ENV_SERIALNUM     4  /* NetComm serial number */

/* define the control values for the environment mode index. */
#define ENVMODE_RAM       1  /* make RAM copy of env, for subsequent use */
#define ENVMODE_ROM       2  /* use ROM environment variables */
#define ENVMODE_ERASE     3  /* erase all bootloader environment variables,
			      * busy-waiting until complete
			      */
#define ENVMODE_REQ_ERASE 4  /* request start of erase operation:
			      * flash is inaccessible until checkFlash()
			      *	indicates completion of erase operation.
			      */
#define ENVMODE_CURMODE	  5  /* Determine current operating mode (ROM, RAM)
			      * ROM: Returns ENVMODE_ROM
			      * RAM: Returns ENVMODE_RAM
			      */
#define ENVMODE_FREE	  6  /* Return count of free bytes */
#define ENVMODE_GETFIXED  7  /* Return count of fixed entries */

/**
 * Define a structure used to inform higher layers about the
 * structure of a flash configuration.  The __BN_flashConfig
 * trap function expects a pointer to an array of flashConfigStruct
 * in order to fill them in.  The bootloader always returns the
 * number of structures as its return value; if the caller
 * provides a NULL pointer, then the bootloader does not try
 * to fill in the structures.
 */
#define NC_BOOTSECTORCOUNT 8    /* size of bootsector array */

struct flashConfigStruct
{
    unsigned char *chipBase;
    unsigned char *bootBase;
    int chipSize;
    int sectSize;
    int bootSectSize[NC_BOOTSECTORCOUNT];
};

/* Calling conventions compatible to (uC)linux/68k
 * We use similar macros to call into the bootloader as for uClinux
 */

#define __bsc_return(type, res) \
do { \
   if ((unsigned long)(res) >= (unsigned long)(-64)) { \
      /* let errno be a function, preserve res in %d0 */ \
      int __err = -(res); \
      errno = __err; \
      res = -1; \
   } \
   return (type)(res); \
} while (0)

#define _bsc0(type,name) \
type name(void) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#define _bsc1(type,name,atype,a) \
type name(atype a) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   register long __a __asm__ ("%d1") = (long)a; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res), "d" (__a) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#define _bsc2(type,name,atype,a,btype,b) \
type name(atype a, btype b) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   register long __a __asm__ ("%d1") = (long)a; \
   register long __b __asm__ ("%d2") = (long)b; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res), "d" (__a), "d" (__b) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#define _bsc3(type,name,atype,a,btype,b,ctype,c) \
type name(atype a, btype b, ctype c) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   register long __a __asm__ ("%d1") = (long)a; \
   register long __b __asm__ ("%d2") = (long)b; \
   register long __c __asm__ ("%d3") = (long)c; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res), "d" (__a), "d" (__b), \
                           "d" (__c) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#define _bsc4(type,name,atype,a,btype,b,ctype,c,dtype,d) \
type name(atype a, btype b, ctype c, dtype d) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   register long __a __asm__ ("%d1") = (long)a; \
   register long __b __asm__ ("%d2") = (long)b; \
   register long __c __asm__ ("%d3") = (long)c; \
   register long __d __asm__ ("%d4") = (long)d; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res), "d" (__a), "d" (__b), \
                           "d" (__c), "d" (__d) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#define _bsc5(type,name,atype,a,btype,b,ctype,c,dtype,d,etype,e) \
type name(atype a, btype b, ctype c, dtype d, etype e) \
{ \
   register long __res __asm__ ("%d0") = __BN_##name; \
   register long __a __asm__ ("%d1") = (long)a; \
   register long __b __asm__ ("%d2") = (long)b; \
   register long __c __asm__ ("%d3") = (long)c; \
   register long __d __asm__ ("%d4") = (long)d; \
   register long __e __asm__ ("%d5") = (long)e; \
   __asm__ __volatile__ ("trap #2" \
                         : "=g" (__res) \
                         : "0" (__res), "d" (__a), "d" (__b), \
                           "d" (__c), "d" (__d), "d" (__e) \
                         : "%d0"); \
   __bsc_return(type,__res); \
}

#endif /* __FW_ACCESS_H__ */
