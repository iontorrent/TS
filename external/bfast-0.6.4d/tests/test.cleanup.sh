#!/bin/sh

. test.definitions.sh

echo "      Cleaning up files.";

rm -r $OUTPUT_DIR $TMP_DIR 2> /dev/null;
ls -1 $DATA_DIR/* | grep -v bz2 | xargs rm 2> /dev/null;

# Test passed!
echo "      Files cleaned up.";
exit 0
