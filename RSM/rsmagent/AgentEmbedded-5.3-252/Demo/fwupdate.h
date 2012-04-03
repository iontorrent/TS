#if !defined(__FWUPDATE_H__)
# define __FWUPDATE_H__ 1
/* fwupdate.h: structure and entry point definitions for firmware
 * update tool
 */

struct fwBlock
{
    struct fwBlock *	next;
    int			size;
    char		data[1];
};

struct _fwupdateParm
{
    unsigned int        address;
    unsigned int        length;
    unsigned int        mode;
    unsigned int	exec;
    const char *        hostAddr;
    const char *        hostPath;
    const char *        hostPort;
    struct fwBlock	blockHead;
    unsigned short      protocol;
};

/* Bitmask flags in the mode structure element */
# define MODE_READ      1
# define MODE_BINARY    2
# define MODE_CONTINUE	4
# define MODE_VERBOSE	8
# define MODE_WAIT      16
# define MODE_DRYRUN    32

/* Protocol numbers in the protocol structure element */
# define PROTO_UNKNOWN  0
# define PROTO_TFTP     1
# define PROTO_HTTP     2

#endif /* !defined(__FWUPDATE_H__) */
