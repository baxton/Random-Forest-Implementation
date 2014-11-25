


//
// g++ -L. -l rf fast.cpp -o fast.exe
//

#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;


typedef unsigned long long ULONG;       // I have to care about sizes as I'm on 32 bit system


const int ALLIGN = 64;

const ULONG VEC_LEN = 32;
const ULONG VEC_LEN_BYTES = VEC_LEN * sizeof(int);
//const ULONG LINES = 1 << 18;
const ULONG LINES = 1 << 16;
const ULONG BUFFER_SIZE = LINES * VEC_LEN;
const ULONG BUFFER_SIZE_BYTES = BUFFER_SIZE * sizeof(int);

int buffer[ALLIGN + BUFFER_SIZE];



const ULONG total_file_size= 5174907776;
const ULONG total_lines = total_file_size / (32 * 4);
const ULONG iterations = total_lines / LINES;
const ULONG rest_bytes = total_file_size - (iterations * BUFFER_SIZE_BYTES);               // how many bytes are left in the file after iterations



const int TOTAL_SPARSE_FEATURES = 9449205;
const int TOTAL_FEATURES = 4 + TOTAL_SPARSE_FEATURES;


const int TREES   = 1;
void* hLearners[TREES];

const int COLUMNS = TOTAL_FEATURES;
const int K       = 3074; //3500;
const int LNUM    = 5;

const int LINES_IN_TRAIN = 40428967;
const int LINES_IN_BS    = 500000;



//
// definitions are in RF_sparse.cpp (rf.dll)
//
extern "C" {
    void predict(void* ptree, double* x, double* p);
    void* alloc_tree_learner(int columns, int k, int lnum);
    void* alloc_tree();
    void free_tree(void* ptree);
    void fromfile_tree(void* ptree, const char* fname);
    void tofile_tree(void* ptree, const char* fname);
    void print_tree(void* ptree);
    void start_fit_tree(void* ptree);
    void fit_tree(void* ptree, double* x, double y);
    void stop_fit_tree(void* ptree);
    int end_of_splitting(void* ptree);
};   // END of extern "C"





void process(int* matrix, int rows);



int main(int argc, const char* argv[]) {
    const char* fname = "C:\\Temp\\kaggle\\CTR\\data\\sub1\\train.csv.b";
    //const char* fname = "C:\\Temp\\kaggle\\CTR\\data\\sub1\\test.csv.b";


    ///////////////////////////////////////////////////////////////////////
    // prepare learner - random forest
    ///////////////////////////////////////////////////////////////////////
    for (int t = 0; t < TREES; ++t) {
        hLearners[t] = alloc_tree_learner(COLUMNS, K, LNUM);
        start_fit_tree(hLearners[t]);
    }

    // Allign the read buffer
    char* p = (char*)buffer;
    while (0 != ((ULONG)p % ALLIGN)) {
        ++p;
    }

    ///////////////////////////////////////////////////////////////////////
    // Reading data
    ///////////////////////////////////////////////////////////////////////

    int iter_num = 0;
    clock_t t = clock();

    FILE* fin = fopen(fname, "rb");

    cout << "# expected iters " << iterations << endl;

    ULONG cnt = 0;
    ULONG read = 0;
    for (int i = 0; i < iterations; ++i) {
        read = (ULONG)fread(p, BUFFER_SIZE_BYTES, 1, fin);
        cnt += read;
        cout << "# read " << cnt << " elements (" << (cnt * BUFFER_SIZE_BYTES) << " bytes)" << endl;


        process((int*)p, LINES);
    }

    read = (ULONG)fread(p, rest_bytes, 1, fin);
    cout << "# rest " << read << " elements (" << (read * rest_bytes) << " bytes)" << endl;
    cout << "# total bytes: " << (cnt * BUFFER_SIZE_BYTES + read * rest_bytes) << endl;

    process((int*)p, (rest_bytes / (32*4)));

    fseek(fin, 0, SEEK_SET);

    ++iter_num;

    // print time
    t = clock() - t;
    cout << "# iteration " << iter_num << " time " << (((float)t)/CLOCKS_PER_SEC) << " sec" << endl;

    ///////////////////////////////////////////////////////////////////////
    // END of reading data
    ///////////////////////////////////////////////////////////////////////


    // free the learner before exiting
    for (int t = 0; t < TREES; ++t) {
        free_tree(hLearners[t]);
    }

    return 0;

}




void process(int* matrix, int rows) {
    double vec[VEC_LEN];

    clock_t t = clock();

    for (int r = 0; r < rows; ++r) {
        // convert to double
        for (int c = 0; c < VEC_LEN; ++c) {
            vec[c] = (double)matrix[r * VEC_LEN + c];
        }

        // feed to RF
        for (int t = 0; t < TREES; ++t) {
            fit_tree(hLearners[t], &vec[2], vec[1]);
        }

        // some output
        if (r > 0 && 0 == (r % 10000)) {
            cout << "# processed rows " << r << endl;
        }
    }

    // print time
    t = clock() - t;
    cout << "# feeding to RF for " << rows << " rows took " << (((float)t)/CLOCKS_PER_SEC) << " sec" << endl;
}

