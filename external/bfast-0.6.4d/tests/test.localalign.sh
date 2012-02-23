#!/bin/sh

. test.definitions.sh

echo "      Running local alignment.";

for SPACE in 0 1
do
	for CORNER_CASE in 0 1
	do
		NUM=`expr $CORNER_CASE \\* 2`;
		NUM=`expr $NUM + $SPACE`;
		case $NUM in
			0) OUTPUT_ID=$OUTPUT_ID_NT;
			REF_ID=$OUTPUT_ID;
			;;
			1) OUTPUT_ID=$OUTPUT_ID_CS;
			REF_ID=$OUTPUT_ID;
			;;
			2) OUTPUT_ID=$OUTPUT_ID_CC_NT;
			REF_ID=$REF_ID_CC;
			;;
			3) OUTPUT_ID=$OUTPUT_ID_CC_CS;
			REF_ID=$REF_ID_CC;
			;;
			default)
			exit 1;
		esac
		echo "        Testing -A "$SPACE "CC="$CORNER_CASE;

		RG_FASTA=$OUTPUT_DIR$REF_ID".fa";
		MATCHES=$OUTPUT_DIR"bfast.matches.file.$OUTPUT_ID.bmf";

		# Run local alignment
		CMD=$CMD_PREFIX"bfast localalign -f $RG_FASTA -m $MATCHES -A $SPACE -n $NUM_THREADS -o 15 > ${OUTPUT_DIR}bfast.aligned.file.$OUTPUT_ID.baf";
		eval $CMD 2> /dev/null;

		# Get return code
		if [ "$?" -ne "0" ]; then
			echo $CMD;
			# Run again without piping anything
			eval $CMD;
			exit 1
		fi
	done
done

# Test passed!
echo "      Local alignment complete.";
exit 0
