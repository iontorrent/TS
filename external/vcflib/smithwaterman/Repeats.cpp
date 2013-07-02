#include "Repeats.h"

map<string, int> repeatCounts(long int position, const string& sequence, int maxsize) {
    map<string, int> counts;
    for (int i = 1; i <= maxsize; ++i) {
        // subseq here i bases
        string seq = sequence.substr(position, i);
        // go left.

        int j = position - i;
        int leftsteps = 0;
        while (j >= 0 && seq == sequence.substr(j, i)) {
            j -= i;
            ++leftsteps;
        }

        // go right.
        j = position;

        int rightsteps = 0;
        while (j + i <= (int)sequence.size() && seq == sequence.substr(j, i)) {
            j += i;
            ++rightsteps;
        }
        // if we went left and right a non-zero number of times, 
        if (leftsteps + rightsteps > 1) {
            counts[seq] = leftsteps + rightsteps;
        }
    }

    // filter out redundant repeat information
    if (counts.size() > 1) {
        map<string, int> filteredcounts;
        map<string, int>::iterator c = counts.begin();
        string prev = c->first;
        filteredcounts[prev] = c->second;  // shortest sequence
        ++c;
        for (; c != counts.end(); ++c) {
            int i = 0;
            string seq = c->first;
            while (i + prev.length() <= seq.length() && seq.substr(i, prev.length()) == prev) {
                i += prev.length();
            }
            if (i < (int)seq.length()) {
                filteredcounts[seq] = c->second;
                prev = seq;
            }
        }
        return filteredcounts;
    } else {
        return counts;
    }
}

bool isRepeatUnit(const string& seq, const string& unit) {

    if (seq.size() % unit.size() != 0) {
	return false;
    } else {
	int maxrepeats = seq.size() / unit.size();
	for (int i = 0; i < maxrepeats; ++i) {
	    if (seq.substr(i * unit.size(), unit.size()) != unit) {
		return false;
	    }
	}
	return true;
    }

}
