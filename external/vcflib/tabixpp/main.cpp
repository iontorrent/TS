#include "tabix.hpp"
#include <vector>

using namespace std;

int main(int argc, char** argv) {

    if (argc < 2) {
        cerr << argv[0] << " [file] [ [region] ... ]" << endl
             << "Writes out regions from bgzf-compressed, tabix-indexed file." << endl
             << "Supply 'header' to print out the header, and no regions to" << endl
             << "print the contents of the entire file." << endl;
        return 1;
    }

    string filename = string(argv[1]);
    vector<string> regions;
    for (int i = 2; i < argc; ++i) {
        regions.push_back(string(argv[i]));
    }

    Tabix file(filename);

    if (!regions.empty()) {
        for (vector<string>::iterator r = regions.begin(); r != regions.end(); ++r) { 
            string& region = *r;
            if (region == "header") {
                string header;
                file.getHeader(header);
                cout << header;
            } else {
                string line;
                file.setRegion(region);
                while (file.getNextLine(line)) {
                    cout << line << endl;
                }
            }
        }
    } else {
        string line;
        while (file.getNextLine(line)) {
            cout << line << endl;
        }
    }

    return 0;
}
