#include "LeftAlign.h"

//bool debug;
#define VERBOSE_DEBUG

// Attempts to left-realign all the indels represented by the alignment cigar.
//
// This is done by shifting all indels as far left as they can go without
// mismatch, then merging neighboring indels of the same class.  leftAlign
// updates the alignment cigar with changes, and returns true if realignment
// changed the alignment cigar.
//
// To left-align, we move multi-base indels left by their own length as long as
// the preceding bases match the inserted or deleted sequence.  After this
// step, we handle multi-base homopolymer indels by shifting them one base to
// the left until they mismatch the reference.
//
// To merge neighboring indels, we iterate through the set of left-stabilized
// indels.  For each indel we add a new cigar element to the new cigar.  If a
// deletion follows a deletion, or an insertion occurs at the same place as
// another insertion, we merge the events by extending the previous cigar
// element.
//
// In practice, we must call this function until the alignment is stabilized.
//
bool leftAlign(string& querySequence, string& cigar, string& baseReferenceSequence, int& offset, bool debug) {

    debug = false;

    string referenceSequence = baseReferenceSequence.substr(offset);

    int arsOffset = 0; // pointer to insertion point in aligned reference sequence
    string alignedReferenceSequence, alignedQuerySequence;
    if (debug) alignedReferenceSequence = referenceSequence;
    if (debug) alignedQuerySequence = querySequence;
    int aabOffset = 0;

    // store information about the indels
    vector<IndelAllele> indels;

    int rp = 0;  // read position, 0-based relative to read
    int sp = 0;  // sequence position

    string softBegin;
    string softEnd;

    string cigarbefore = cigar;

    vector<pair<int, string> > cigarData = splitCIGAR(cigar);
    for (vector<pair<int, string> >::const_iterator c = cigarData.begin();
        c != cigarData.end(); ++c) {
        unsigned int l = c->first;
        string t = c->second;
	if (debug) cerr << l << t << " " << sp << " " << rp << endl;
        if (t == "M") { // match or mismatch
            sp += l;
            rp += l;
        } else if (t == "D") { // deletion
            indels.push_back(IndelAllele(false, l, sp, rp, referenceSequence.substr(sp, l)));
            if (debug) { cerr << indels.back() << endl;  alignedQuerySequence.insert(rp + aabOffset, string(l, '-')); }
            aabOffset += l;
            sp += l;  // update reference sequence position
        } else if (t == "I") { // insertion
            indels.push_back(IndelAllele(true, l, sp, rp, querySequence.substr(rp, l)));
            if (debug) { cerr << indels.back() << endl; alignedReferenceSequence.insert(sp + softBegin.size() + arsOffset, string(l, '-')); }
            arsOffset += l;
            rp += l;
        } else if (t == "S") { // soft clip, clipped sequence present in the read not matching the reference
            // remove these bases from the refseq and read seq, but don't modify the alignment sequence
            if (rp == 0) {
                alignedReferenceSequence = string(l, '*') + alignedReferenceSequence;
		//indels.push_back(IndelAllele(true, l, sp, rp, querySequence.substr(rp, l)));
                softBegin = querySequence.substr(0, l);
            } else {
                alignedReferenceSequence = alignedReferenceSequence + string(l, '*');
		//indels.push_back(IndelAllele(true, l, sp, rp, querySequence.substr(rp, l)));
                softEnd = querySequence.substr(querySequence.size() - l, l);
            }
            rp += l;
        } else if (t == "H") { // hard clip on the read, clipped sequence is not present in the read
        } else if (t == "N") { // skipped region in the reference not present in read, aka splice
            sp += l;
        }
    }


    if (debug) cerr << "| " << cigarbefore << endl
		    << "| " << alignedReferenceSequence << endl
		    << "| " << alignedQuerySequence << endl;

    // if no indels, return the alignment
    if (indels.empty()) { return false; }

    if (debug) {
	for (vector<IndelAllele>::iterator a = indels.begin(); a != indels.end(); ++a) cerr << *a << " ";
	cerr << endl;
    }

    // for each indel, from left to right
    //     while the indel sequence repeated to the left and we're not matched up with the left-previous indel
    //         move the indel left

    vector<IndelAllele>::iterator previous = indels.begin();
    for (vector<IndelAllele>::iterator id = indels.begin(); id != indels.end(); ++id) {

        // left shift by repeats
        //
        // from 1 base to the length of the indel, attempt to shift left
        // if the move would cause no change in alignment optimality (no
        // introduction of mismatches, and by definition no change in gap
        // length), move to the new position.
        // in practice this moves the indel left when we reach the size of
        // the repeat unit.
        //
        int steppos, readsteppos;
        IndelAllele& indel = *id;
        int i = 1;

        while (i <= indel.length) {

            int steppos = indel.position - i;
            int readsteppos = indel.readPosition - i;

            if (debug) {
                if (steppos >= 0 && readsteppos >= 0) {
                    cerr << "refseq flank " << referenceSequence.substr(steppos, indel.length) << endl;
                    cerr << "qryseq flank " << querySequence.substr(readsteppos, indel.length) << endl;
                    cerr << "indelseq     " << indel.sequence << endl;
                }
            }

            while (steppos >= 0 && readsteppos >= 0
                   && indel.sequence == referenceSequence.substr(steppos, indel.length)
                   && indel.sequence == querySequence.substr(readsteppos, indel.length)
                   && (id == indels.begin()
                       || (previous->insertion && steppos >= previous->position)
                       || (!previous->insertion && steppos >= previous->position + previous->length))) {
                LEFTALIGN_DEBUG((indel.insertion ? "insertion " : "deletion ") << indel << " shifting " << i << "bp left" << endl);
                indel.position -= i;
                indel.readPosition -= i;
                steppos = indel.position - i;
                readsteppos = indel.readPosition - i;
            }
            do {
                ++i;
            } while (i <= indel.length && indel.length % i != 0);
        }



        // left shift indels with exchangeable flanking sequence
        //
        // for example:
        //
        //    GTTACGTT           GTTACGTT
        //    GT-----T   ---->   G-----TT
        //
        // GTGTGACGTGT           GTGTGACGTGT
        // GTGTG-----T   ---->   GTG-----TGT
        //
        // GTGTG-----T           GTG-----TGT
        // GTGTGACGTGT   ---->   GTGTGACGTGT
        //
        //

        steppos = indel.position - 1;
        readsteppos = indel.readPosition - 1;
        while (steppos >= 0 && readsteppos >= 0
               && querySequence.at(readsteppos) == referenceSequence.at(steppos)
	       && (int)referenceSequence.size() > steppos + indel.length
	       && indel.sequence.at((int) indel.sequence.size() - 1) == referenceSequence.at(steppos + indel.length) // are the exchanged bases going to match wrt. the reference?
               && querySequence.at(readsteppos) == indel.sequence.at((int) indel.sequence.size() - 1)
               && (id == indels.begin()
                   || (previous->insertion && indel.position - 1 >= previous->position)
                   || (!previous->insertion && indel.position - 1 >= previous->position + previous->length))) {
            if (debug) cerr << (indel.insertion ? "insertion " : "deletion ") << indel << " exchanging bases " << 1 << "bp left" << endl;
            indel.sequence = indel.sequence.at(indel.sequence.size() - 1) + indel.sequence.substr(0, indel.sequence.size() - 1);
            indel.position -= 1;
            indel.readPosition -= 1;
	    if (debug) cerr << indel << endl;
            steppos = indel.position - 1;
            readsteppos = indel.readPosition - 1;
	    //if (debug && steppos && readsteppos) cerr << querySequence.at(readsteppos) << " ==? " << referenceSequence.at(steppos) << endl;
	    //if (debug && steppos && readsteppos) cerr << indel.sequence.at((int) indel.sequence.size() - 1) << " ==? " << referenceSequence.at(steppos + indel.length) << endl;
        }
        // tracks previous indel, so we don't run into it with the next shift
        previous = id;
    }

    if (debug) {
	for (vector<IndelAllele>::iterator a = indels.begin(); a != indels.end(); ++a) cerr << *a << " ";
	cerr << endl;
    }

    if (debug) cerr << "bring together floating indels" << endl;

    // bring together floating indels
    // from left to right
    // check if we could merge with the next indel
    // if so, adjust so that we will merge in the next step
    if (indels.size() > 1) {
        previous = indels.begin();
        for (vector<IndelAllele>::iterator id = (indels.begin() + 1); id != indels.end(); ++id) {
            IndelAllele& indel = *id;
            // parsimony: could we shift right and merge with the previous indel?
            // if so, do it
            int prev_end_ref = previous->insertion ? previous->position : previous->position + previous->length;
            int prev_end_read = !previous->insertion ? previous->readPosition : previous->readPosition + previous->length;
            if (previous->insertion == indel.insertion
                    && ((previous->insertion
                        && (previous->position < indel.position
                        && previous->readPosition < indel.readPosition))
                        ||
                        (!previous->insertion
                        && (previous->position + previous->length < indel.position)
                        && (previous->readPosition < indel.readPosition)
                        ))) {
                if (previous->homopolymer()) {
                    string seq = referenceSequence.substr(prev_end_ref, indel.position - prev_end_ref);
                    string readseq = querySequence.substr(prev_end_read, indel.position - prev_end_ref);
                    if (debug) cerr << "seq: " << seq << endl << "readseq: " << readseq << endl;
                    if (previous->sequence.at(0) == seq.at(0)
                            && homopolymer(seq)
                            && homopolymer(readseq)) {
                        if (debug) cerr << "moving " << *previous << " right to " 
					<< (indel.insertion ? indel.position : indel.position - previous->length) << endl;
                        previous->position = indel.insertion ? indel.position : indel.position - previous->length;
			previous->readPosition = !indel.insertion ? indel.readPosition : indel.readPosition - previous->readLength(); // should this be readLength?
                    }
                }
		/*
                else {
                    int pos = previous->position;
		    int readpos = previous->readPosition;
                    while (pos < (int) referenceSequence.length() &&
                            ((previous->insertion && pos + previous->length <= indel.position)
                            ||
                            (!previous->insertion && pos + previous->length < indel.position))
			   && previous->sequence == referenceSequence.substr(pos + previous->length, previous->length)
			   && previous->sequence == querySequence.substr(readpos + previous->length, previous->length)
			) {
                        pos += previous->length;
			readpos += previous->length;
                    }
		    string seq = previous->sequence;
                    if (pos > previous->position) {
			// wobble bases right to left as far as we can go
			int steppos = previous->position + seq.size();
			int readsteppos = previous->readPosition + seq.size();

			while (querySequence.at(readsteppos) == referenceSequence.at(steppos)
			       && querySequence.at(readsteppos) == seq.at(0)
			       && (id == indels.begin()
				   || (indel.insertion && pos + seq.size() - 1 <= indel.position)
				   || (!previous->insertion && indel.position - 1 >= pos + previous->length))) {
			    seq = seq.substr(1) + seq.at(0);
			    ++pos;
			    ++readpos;
			    steppos = pos + 1;
			    readsteppos = readpos + 1;
			}

			if (((previous->insertion && pos + previous->length == indel.position)
			     ||
			     (!previous->insertion && pos == indel.position - previous->length))
			    ) {
			    if (debug) cerr << "right-merging tandem repeat: moving " << *previous << " right to " << pos << endl;
			    previous->position = pos;
			    previous->readPosition = readpos;
			    previous->sequence = seq;
			}
                    }
                }
		*/
            }
            previous = id;
        }
    }

    if (debug) {
	for (vector<IndelAllele>::iterator a = indels.begin(); a != indels.end(); ++a) cerr << *a << " ";
	cerr << endl;
    }


    if (debug) cerr << "bring in indels at ends of read" << endl;

    // try to "bring in" repeat indels at the end, for maximum parsimony
    //
    // e.g.
    //
    // AGAAAGAAAGAAAAAGAAAAAGAACCAAGAAGAAAA
    // AGAAAG------AAAGAAAAAGAACCAAGAAGAAAA
    //
    //     has no information which distinguishes it from:
    //
    // AGAAAGAAAAAGAAAAAGAACCAAGAAGAAAA
    // AGAAAG--AAAGAAAAAGAACCAAGAAGAAAA
    //
    // here we take the parsimonious explanation

    if (!indels.empty()) {
	// deal with the first indel
	// the first deletion ... or the biggest deletion
	vector<IndelAllele>::iterator a = indels.begin();
	vector<IndelAllele>::iterator del = indels.begin();
	for (; a != indels.end(); ++a) {
	    //if (!a->insertion && a->length > biggestDel->length) biggestDel = a;
	    if (!a->insertion && a->length) del = a;
	if (!del->insertion) {
	    //if (!indel.insertion) { // only handle deletions like this for now
	    //if (!indel.insertion && !(indels.size() > 1 && indel.readPosition == indels.at(1).readPosition)) { // only handle deletions like this for now
	    int insertedBpBefore = 0;
	    int deletedBpBefore = 0;
	    for (vector<IndelAllele>::iterator i = indels.begin(); i != del; ++i) {
		if (i->insertion) insertedBpBefore += i->length;
		else deletedBpBefore += i->length;
	    }
	    IndelAllele& indel = *del;
	    int minsize = indel.length;
	    int flankingLength = indel.readPosition;
	    if (debug) cerr << indel << endl;
	    string flanking = querySequence.substr(0, flankingLength);
	    if (debug) cerr << flanking << endl;

	    size_t p = referenceSequence.substr(0, indel.position + indel.length).rfind(flanking);
	    if (p == string::npos) {
		if (debug) cerr << "flanking not found" << endl;
	    } else {
		if (debug) {
		    cerr << "flanking is at " << p << endl;
		    cerr << "minsize would be " << (indel.position + indel.length) - ((int) p + flankingLength) << endl;
		}
		minsize = (indel.position + indel.length) - ((int) p + flankingLength);
	    }

	    if (debug) cerr << minsize << endl;

	    if (minsize >= 0 && minsize < indel.length) {

		int softdiff = softBegin.size();
		if (!softBegin.empty()) { // remove soft clips if we can
		    if (flankingLength < (int)softBegin.size()) {
			softBegin = softBegin.substr(0, flankingLength - softBegin.size());
			softdiff -= softBegin.size();
		    } else {
			softBegin.clear();
		    }
		}

		// the new read position == the current read position
		// the new reference position == the flanking length size
		// the positional offset of the reference sequence == the new position of the deletion - the flanking length

		int diff = indel.length - minsize - softdiff  + deletedBpBefore - insertedBpBefore;
		//int querydiff = indel.length - minsize - softBegin.size() - insertedBpBefore + deletedBpBefore;
		if (debug) cerr << "adjusting " << indel.length <<" " << minsize << "  " << softdiff << " " << diff << endl;
		offset += diff;
		///
		indel.length = minsize;
		indel.sequence = indel.sequence.substr(indel.sequence.size() - minsize);
		indel.position = flankingLength;
		indel.readPosition = indel.position; // if we have removed all the sequence before, this should be ==
		referenceSequence = referenceSequence.substr(diff);

		for (vector<IndelAllele>::iterator i = indels.begin(); i != indels.end(); ++i) {
		    if (i < del) {
			i->length = 0; // remove
		    } else if (i > del) {
			i->position -= diff;
		    }
		}
	    }
	    if (debug) cerr << indel << endl;

	    // now, do the same for the reverse
	    if (indel.length > 0) {
		int minsize = indel.length + 1;
		int flankingLength = querySequence.size() - indel.readPosition + indel.readLength();
		string flanking = querySequence.substr(indel.readPosition + indel.readLength(), flankingLength);
		//int indelRefEnd = indel.position + indel.referenceLength();

		size_t p = referenceSequence.find(flanking, indel.position);
		if (p == string::npos) {
		    if (debug)
			cerr << "flanking not found" << endl;
		} else {
		    if (debug) {
			cerr << "flanking is at " << p << endl;
			cerr << "minsize would be " << (int) p - indel.position << endl;
		    }
		    minsize = (int) p - indel.position;
		}

		if (debug) cerr << "minsize " << minsize << endl;
		if (minsize >= 0 && minsize <= indel.length) {
		    //referenceSequence = referenceSequence.substr(0, referenceSequence.size() - (indel.length - minsize));
		    if (debug) cerr << "adjusting " << indel << endl;
		    indel.length = minsize;
		    indel.sequence = indel.sequence.substr(0, minsize);
		    //cerr << indel << endl;
		    if (!softEnd.empty()) { // remove soft clips if we can
			if (flankingLength < (int)softEnd.size()) {
			    softEnd = softEnd.substr(flankingLength - softEnd.size());
			} else {
			    softEnd.clear();
			}
		    }
		    for (vector<IndelAllele>::iterator i = indels.begin(); i != indels.end(); ++i) {
			if (i > del) {
			    i->length = 0; // remove
			}
		    }
		}
	    }
	}
	}
    }

    if (debug) {
	for (vector<IndelAllele>::iterator a = indels.begin(); a != indels.end(); ++a) cerr << *a << " ";
	cerr << endl;
    }

    if (debug) cerr << "parsing indels" << endl;

    // if soft clipping can be reduced by adjusting the tailing indels in the read, do it
    // TODO

    /*
    int numEmptyIndels = 0;

    if (!indels.empty()) {
	vector<IndelAllele>::iterator a = indels.begin();
	while (a != indels.end()) {
	    if (debug) cerr << "parsing " << *a << endl;
	    if (!(a->length > 0 && a->position >= 0)) {
		++numEmptyIndels;
	    }
	    ++a;
	}
    }
    */

    for (vector<IndelAllele>::iterator i = indels.begin(); i != indels.end(); ++i) {
	if (i->length == 0) continue;
	if (i->insertion) {
	    if (querySequence.substr(i->readPosition, i->readLength()) != i->sequence) {
		cerr << "failure: " << *i << " should be " << querySequence.substr(i->readPosition, i->readLength()) << endl;
		cerr << baseReferenceSequence << endl;
		cerr << querySequence << endl;
		throw 1;
	    }
	} else {
	    if (referenceSequence.substr(i->position, i->length) != i->sequence) {
		cerr << "failure: " << *i << " should be " << referenceSequence.substr(i->position, i->length) << endl;
		cerr << baseReferenceSequence << endl;
		cerr << querySequence << endl;
		throw 1;
	    }
	}
    }

    if (indels.size() > 1) {
        vector<IndelAllele>::iterator id = indels.begin();
	while ((id + 1) != indels.end()) {
	    if (debug) {
		cerr << "indels: ";
		for (vector<IndelAllele>::iterator a = indels.begin(); a != indels.end(); ++a) cerr << *a << " ";
		cerr << endl;
		//for (vector<IndelAllele>::iterator a = newIndels.begin(); a != newIndels.end(); ++a) cerr << *a << " ";
		//cerr << endl;
	    }

	    // get the indels to try to merge
	    while (id->length == 0 && (id + 1) != indels.end()) ++id;
	    vector<IndelAllele>::iterator idn = (id + 1);
	    while (idn != indels.end() && idn->length == 0) ++idn;
	    if (idn == indels.end()) break;

            IndelAllele& indel = *idn;
	    IndelAllele& last = *id;
	    if (debug) cerr << "trying " << last << " against " << indel << endl;

	    int lastend = last.insertion ? last.position : (last.position + last.length);
	    if (indel.position == lastend) {
		if (debug) cerr << "indel.position " << indel.position << " lastend " << lastend << endl;
		if (indel.insertion == last.insertion) {
		    last.length += indel.length;
		    last.sequence += indel.sequence;
		    indel.length = 0;
		    indel.sequence.clear();
		    id = idn;
		} else if (last.length && indel.length) { // if the end of the previous == the start of the current, cut it off of both the ins and the del
		    if (debug) cerr << "Merging " << last << " " << indel << endl;
		    int matchsize = 1;
		    int biggestmatchsize = 0;

		    while (matchsize <= (int)last.sequence.size() && matchsize <= (int)indel.sequence.size()) {
			if (last.sequence.substr(last.sequence.size() - matchsize) == indel.sequence.substr(0, matchsize)) {
			    biggestmatchsize = matchsize;
			}
			++matchsize;
		    }
		    if (debug) cerr << "biggestmatchsize " << biggestmatchsize << endl;

		    last.sequence = last.sequence.substr(0, last.sequence.size() - biggestmatchsize);
		    last.length -= biggestmatchsize;
		    indel.sequence = indel.sequence.substr(biggestmatchsize);
		    indel.length -= biggestmatchsize;
		    if (indel.insertion) indel.readPosition += biggestmatchsize;
		    else indel.position += biggestmatchsize;

		    if (indel.length > 0) {
			id = idn;
		    }
		}
	    } else {
		if (last.insertion != indel.insertion) {
		    if (debug) cerr << "merging by overlap " << last << " " << indel << endl;
		    // see if we can slide the sequence in between these two indels together
		    string lastOverlapSeq;
		    string indelOverlapSeq;

		    if (last.insertion) {
			lastOverlapSeq =
			    last.sequence
			    + querySequence.substr(last.readPosition + last.readLength(),
						   indel.readPosition - (last.readPosition + last.readLength()));
			indelOverlapSeq =
			    referenceSequence.substr(last.position + last.referenceLength(),
						     indel.position - (last.position + last.referenceLength()))
			    + indel.sequence;
		    } else {
			lastOverlapSeq =
			    last.sequence
			    + referenceSequence.substr(last.position + last.referenceLength(),
						       indel.position - (last.position + last.referenceLength()));
			indelOverlapSeq =
			    querySequence.substr(last.readPosition + last.readLength(),
						 indel.readPosition - (last.readPosition + last.readLength()))
			    + indel.sequence;
		    }

		    if (debug) {
			if (!last.insertion) {
			    if (last.insertion) cerr << string(last.length, '-');
			    cerr << lastOverlapSeq;
			    if (indel.insertion) cerr << string(indel.length, '-');
			    cerr << endl;
			    if (!last.insertion) cerr << string(last.length, '-');
			    cerr << indelOverlapSeq;
			    if (!indel.insertion) cerr << string(indel.length, '-');
			    cerr << endl;
			} else {
			    if (last.insertion) cerr << string(last.length, '-');
			    cerr << indelOverlapSeq;
			    if (indel.insertion) cerr << string(indel.length, '-');
			    cerr << endl;
			    if (!last.insertion) cerr << string(last.length, '-');
			    cerr << lastOverlapSeq;
			    if (!indel.insertion) cerr << string(indel.length, '-');
			    cerr << endl;
			}
		    }


		    int dist = min(last.length, indel.length);
		    int matchingInBetween = indel.position - (last.position + last.referenceLength());
		    int previousMatchingInBetween = matchingInBetween;
		    //int matchingInBetween = indel.position - last.position;
		    if (debug) cerr << "matchingInBetween " << matchingInBetween << endl;
		    if (debug) cerr << "dist " << dist << endl;
		    //int mindist = matchingInBetween - dist;
		    if (lastOverlapSeq == indelOverlapSeq) {
			matchingInBetween = lastOverlapSeq.size();
		    } else {
			// TODO change to use string::find()
			for (int i = dist; i > 0; --i) {
			    if (debug) cerr << "lastoverlap: "
					    << lastOverlapSeq.substr(lastOverlapSeq.size() - previousMatchingInBetween - i)
					    << " thisoverlap: "
					    << indelOverlapSeq.substr(0, i + previousMatchingInBetween) << endl;
			    if (lastOverlapSeq.substr(lastOverlapSeq.size() - previousMatchingInBetween - i)
				== indelOverlapSeq.substr(0, i + previousMatchingInBetween)) {
				matchingInBetween = previousMatchingInBetween + i;
				break;
			    }
			}
		    }
		    //cerr << last << " " << indel << endl;
		    if (matchingInBetween > 0 && matchingInBetween > previousMatchingInBetween) {
			if (debug) cerr << "matching " << matchingInBetween  << "bp between " << last << " " << indel
				        << " was matching " << previousMatchingInBetween << endl;
			int diff = matchingInBetween - previousMatchingInBetween;
			last.length -= diff;
			last.sequence = last.sequence.substr(0, last.length);
			indel.length -= diff;
			indel.sequence = indel.sequence.substr(diff);
			if (!indel.insertion) indel.position += diff;
			else indel.readPosition += diff;
			if (debug) cerr << last << " " << indel << endl;
		    }// else if (matchingInBetween == 0 || matchingInBetween == indel.position - last.position) {
			//if (!newIndels.empty()) newIndels.pop_back();
		    //} //else { newIndels.push_back(indel); }
		    id = idn;
		    //newIndels.push_back(indel);
		} else {
		    id = idn;
		    //newIndels.push_back(indel);
		}
	    }
	}
    }

    vector<IndelAllele> newIndels;
    for (vector<IndelAllele>::iterator i = indels.begin(); i != indels.end(); ++i) {
	if (!i->insertion && i->position == 0) { offset += i->length;
	} else if (i->length > 0) newIndels.push_back(*i); // remove dels at front
    }

    // for each indel
    //     if ( we're matched up to the previous insertion (or deletion) 
    //          and it's also an insertion or deletion )
    //         merge the indels
    //
    // and simultaneously reconstruct the cigar
    //
    // if there are spurious deletions at the start and end of the read, remove them
    // if there are spurious insertions after soft-clipped bases, make them soft clips

    vector<pair<int, string> > newCigar;

    if (!softBegin.empty()) {
	newCigar.push_back(make_pair(softBegin.size(), "S"));
    }

    if (newIndels.empty()) {

	int remainingReadBp = querySequence.size() - softEnd.size() - softBegin.size();
	newCigar.push_back(make_pair(remainingReadBp, "M"));

	if (!softEnd.empty()) {
	    newCigar.push_back(make_pair(softEnd.size(), "S"));
	}

	cigar = joinCIGAR(newCigar);

	// check if we're realigned
	if (cigar == cigarbefore) {
	    return false;
	} else {
	    return true;
	}
    }

    vector<IndelAllele>::iterator id = newIndels.begin();
    vector<IndelAllele>::iterator last = id++;

    if (last->position > 0) {
	newCigar.push_back(make_pair(last->position, "M"));
	newCigar.push_back(make_pair(last->length, (last->insertion ? "I" : "D")));
    } else if (last->position == 0) { // discard floating indels
	if (last->insertion) newCigar.push_back(make_pair(last->length, "S"));
	else  newCigar.push_back(make_pair(last->length, "D"));
    } else {
	cerr << "negative indel position " << *last << endl;
    }

    int lastend = last->insertion ? last->position : (last->position + last->length);
    LEFTALIGN_DEBUG(*last << ",");

    for (; id != newIndels.end(); ++id) {
	IndelAllele& indel = *id;
	if (indel.length == 0) continue; // remove 0-length indels
	if (debug) cerr << indel << " " << *last << endl;
	LEFTALIGN_DEBUG(indel << ",");
	if ((((id + 1) == newIndels.end()
	    && indel.insertion && indel.position == (int)referenceSequence.size())
		|| (!indel.insertion && indel.position + indel.length == (int)referenceSequence.size()))
	    ) {
	    if (indel.insertion) {
		if (!newCigar.empty() && newCigar.back().second == "S") {
		    newCigar.back().first += indel.length;
		} else {
		    newCigar.push_back(make_pair(indel.length, "S"));
		}
	    }
	} else if (indel.position < lastend) {
	    cerr << "impossibility?: indel realigned left of another indel" << endl;
	    return false;
	} else if (indel.position == lastend) {
	    // how?
	    if (indel.insertion == last->insertion) {
		pair<int, string>& op = newCigar.back();
		op.first += indel.length;
	    } else {
		newCigar.push_back(make_pair(indel.length, (indel.insertion ? "I" : "D")));
	    }
        } else if (indel.position > lastend) {  // also catches differential indels, but with the same position
	    if (!newCigar.empty() && newCigar.back().second == "M") newCigar.back().first += indel.position - lastend;
	    else newCigar.push_back(make_pair(indel.position - lastend, "M"));
            newCigar.push_back(make_pair(indel.length, (indel.insertion ? "I" : "D")));
        }

	last = id;
	lastend = last->insertion ? last->position : (last->position + last->length);

	if (debug) {
	    for (vector<pair<int, string> >::iterator c = newCigar.begin(); c != newCigar.end(); ++c)
		cerr << c->first << c->second;
	    cerr << endl;
	}

    }

    int remainingReadBp = querySequence.size() - (last->readPosition + last->readLength()) - softEnd.size();
    if (remainingReadBp > 0) {
	if (debug) cerr << "bp remaining = " << remainingReadBp << endl;
	if (newCigar.back().second == "M") newCigar.back().first += remainingReadBp;
	else newCigar.push_back(make_pair(remainingReadBp, "M"));
    }

    if (newCigar.back().second == "D") newCigar.pop_back(); // remove trailing deletions

    if (!softEnd.empty()) {
	if (newCigar.back().second == "S") newCigar.back().first += softEnd.size();
	else newCigar.push_back(make_pair(softEnd.size(), "S"));
    }

    LEFTALIGN_DEBUG(endl);

    cigar = joinCIGAR(newCigar);

    LEFTALIGN_DEBUG(cigar << endl);

    // check if we're realigned
    if (cigar == cigarbefore) {
        return false;
    } else {
        return true;
    }

}

int countMismatches(string& querySequence, string& cigar, string referenceSequence) {

    int mismatches = 0;
    int sp = 0;
    int rp = 0;
    vector<pair<int, string> > cigarData = splitCIGAR(cigar);
    for (vector<pair<int, string> >::const_iterator c = cigarData.begin();
        c != cigarData.end(); ++c) {
        unsigned int l = c->first;
        string t = c->second;
        if (t == "M") { // match or mismatch
            for (int i = 0; i < (int)l; ++i) {
                if (querySequence.at(rp) != referenceSequence.at(sp))
                    ++mismatches;
                ++sp;
                ++rp;
            }
        } else if (t == "D") { // deletion
            sp += l;  // update reference sequence position
        } else if (t == "I") { // insertion
            rp += l;  // update read position
        } else if (t == "S") { // soft clip, clipped sequence present in the read not matching the reference
            rp += l;
        } else if (t == "H") { // hard clip on the read, clipped sequence is not present in the read
        } else if (t == "N") { // skipped region in the reference not present in read, aka splice
            sp += l;
        }
    }

    return mismatches;

}

// Iteratively left-aligns the indels in the alignment until we have a stable
// realignment.  Returns true on realignment success or non-realignment.
// Returns false if we exceed the maximum number of realignment iterations.
//
bool stablyLeftAlign(string querySequence, string& cigar, string referenceSequence, int& offset, int maxiterations, bool debug) {

    if (!leftAlign(querySequence, cigar, referenceSequence, offset)) {

        LEFTALIGN_DEBUG("did not realign" << endl);
        return true;

    } else {

        while (leftAlign(querySequence, cigar, referenceSequence, offset) && --maxiterations > 0) {
            LEFTALIGN_DEBUG("realigning ..." << endl);
        }

        if (maxiterations <= 0) {
            return false;
        } else {
            return true;
        }
    }
}

string mergeCIGAR(const string& c1, const string& c2) {
    vector<pair<int, string> > cigar1 = splitCIGAR(c1);
    vector<pair<int, string> > cigar2 = splitCIGAR(c2);
    // check if the middle elements are the same
    if (cigar1.back().second == cigar2.front().second) {
        cigar1.back().first += cigar2.front().first;
        cigar2.erase(cigar2.begin());
    }
    for (vector<pair<int, string> >::iterator c = cigar2.begin(); c != cigar2.end(); ++c) {
        cigar1.push_back(*c);
    }
    return joinCIGAR(cigar1);
}

vector<pair<int, string> > splitCIGAR(const string& cigarStr) {
    vector<pair<int, string> > cigar;
    string number;
    string type;
    // strings go [Number][Type] ...
    for (string::const_iterator s = cigarStr.begin(); s != cigarStr.end(); ++s) {
        char c = *s;
        if (isdigit(c)) {
            if (type.empty()) {
                number += c;
            } else {
                // signal for next token, push back the last pair, clean up
                cigar.push_back(make_pair(atoi(number.c_str()), type));
                number.clear();
                type.clear();
                number += c;
            }
        } else {
            type += c;
        }
    }
    if (!number.empty() && !type.empty()) {
        cigar.push_back(make_pair(atoi(number.c_str()), type));
    }
    return cigar;
}

string joinCIGAR(const vector<pair<int, string> >& cigar) {
    string cigarStr;
    for (vector<pair<int, string> >::const_iterator c = cigar.begin(); c != cigar.end(); ++c) {
        if (c->first) {
            cigarStr += convert(c->first) + c->second;
        }
    }
    return cigarStr;
}
