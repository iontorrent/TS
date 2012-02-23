#!/bin/sh

. test.definitions.sh

error()
{
  echo $1
  exit 1
}

# Try to find any version of md5sum
mlist="Please, contact: bfast-help@lists.sourceforge.net"
md5bin="md5sum"
[ `which $md5bin` ] || md5bin="gmd5sum"
[ `which $md5bin` ] || error "I can't find md5sum in your system. $mlist"

echo "      Double-checking output files.";

CMD="$md5bin -c $OUTPUT_DIR/tests.md5";
eval $CMD 2> /dev/null > /dev/null;
# Get return code
if [ "$?" -ne "0" ]; then
	# Run again without piping anything
	echo $CMD;
	eval $CMD;
	exit 1;
fi

echo "      Output files are the same.";
