#!/bin/sh 
. test.definitions.sh

echo "      Building an index.";

for SPACE in 0 1
do
	for CORNER_CASE in 0 1
	do
		DEPTH=`expr 1 - $SPACE`;
		DEPTH=`expr 2 \* $DEPTH`;
		WIDTH="8";
		MASK="111101111011101111";
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
			MASK="11111";
			WIDTH="5";
			DEPTH="0";
			;;
			3) OUTPUT_ID=$OUTPUT_ID_CC_CS;
			REF_ID=$REF_ID_CC;
			MASK="11111";
			WIDTH="5";
			DEPTH="0";
			;;
			default)
			exit 1;
		esac
		echo "        Testing -A "$SPACE "CC="$CORNER_CASE;

		RG_FASTA=$OUTPUT_DIR$REF_ID".fa";

		# Make an index
		CMD=$CMD_PREFIX"bfast index -f $RG_FASTA -A $SPACE -m $MASK -w $WIDTH -d $DEPTH -i 1 -n $NUM_THREADS -T $TMP_DIR";
		eval $CMD 2> /dev/null;

		# Get return code
		if [ "$?" -ne "0" ]; then
			# Run again without piping anything
			echo "RETURN CODE=$?";
			echo $CMD;
			eval $CMD;
			exit 1;
		fi
	done
done

# Test passed!
echo "      Index successfully built.";
exit 0
