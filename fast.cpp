


//
// g++ -O3  -ftree-vectorize -L. -l rf fast.cpp -o fast60_8_51.exe
//


#define _FILE_OFFSET_BITS  64
#define _USE_32BIT_TIME_T 1

#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <cerrno>
#include <algorithm>
#include <unistd.h>

using namespace std;

typedef int DATATYPE;
typedef double RFTYPE;

typedef unsigned long long ULONG;       // I have to care about sizes as I'm on 32 bit system


const int ALLIGN = 64;

const ULONG VEC_LEN = 32;
const ULONG VEC_LEN_BYTES = VEC_LEN * sizeof(int);
const ULONG LINES = 1 << 22;
const ULONG BUFFER_SIZE = LINES * VEC_LEN;
const ULONG BUFFER_SIZE_BYTES = BUFFER_SIZE * sizeof(int);

int buffer[ALLIGN + BUFFER_SIZE];

const ULONG total_file_size = 5174907776;
const ULONG total_lines = total_file_size / VEC_LEN_BYTES;

const ULONG rest_bytes = total_file_size - (total_file_size / BUFFER_SIZE_BYTES) * BUFFER_SIZE_BYTES;


const int TOTAL_FEATURES = 4 + 21;

const int TREES   = 3;
void* hLearners[TREES];

const int COLUMNS = TOTAL_FEATURES;
const int K       = TOTAL_FEATURES / 3;
const int LNUM    = 51;

const int LINES_IN_TRAIN = 40428967;
const int LINES_IN_BS    = 24257380;    // 60%
//const int LINES_IN_BS    = 32343173;        // 80%
//const int LINES_IN_BS    = 40428967;   // 100%
//const int LINES_IN_BS      = 36386070; // 90%
//const int LINES_IN_BS    = 100;

int bootstraps[TREES][LINES_IN_BS];
int bootstraps_inclusion[TREES][LINES_IN_BS];







//
// random numbers
//
struct random {
    static void seed(int s=-1) {
        if (s == -1)
            srand(time(NULL));
        else
            srand(s);
    }

    static int randint() {
#if 0x7FFF < RAND_MAX
        return rand();
#else
        return (int)rand() * ((int)RAND_MAX + 1) + (int)rand();
#endif
    }


    static int randint(int low, int high) {
        int r = randint();
        r = r % (high - low) + low;
        return r;
    }

    static void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i) {
            numbers[i] = randint() % (high - low) + low;
        }
    }

};



//
// definitions are in RF_sparse.cpp (rf.dll)
//
extern "C" {
    void predict(void* ptree, DATATYPE* x, double* p, int* can_predict);
    void* alloc_tree_learner(int columns, int k, int lnum);
    void* alloc_tree();
    void free_tree(void* ptree);
    void fromfile_tree(void* ptree, const char* fname);
    void tofile_tree(void* ptree, const char* fname);
    void print_tree(void* ptree);
    void start_fit_tree(void* ptree);
    void fit_tree(void* ptree, DATATYPE* x, DATATYPE y, int* drop_x);
    void stop_fit_tree(void* ptree);
    int end_of_splitting(void* ptree);
};   // END of extern "C"



template <class ForwardIterator, class T>
bool index_in_vec(ForwardIterator first, ForwardIterator last, const T& val, int& index);
bool need_to_read();
void process(int* matrix, int rows);

bool exists(const char* fname);


int main(int argc, const char* argv[]) {
    const char* fname = "C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b";
    //const char* fname = "C:\\Temp\\kaggle\\CTR\\data\\sub2\\test.csv.b";
    //const char* fname = "/home/maxim/kaggle/CTR/data/sub2/train.csv.b";

    if (2 == argc) {
        fname = argv[1];
    }


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
    // prepare bootstraps
    ///////////////////////////////////////////////////////////////////////
    random::seed();
    for (int t = 0; t < TREES; ++t) {
        random::randint(0, LINES_IN_TRAIN, &bootstraps[t][0], LINES_IN_BS);
        sort(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS]);

        for (int i = 0; i < LINES_IN_BS; ++i)
            bootstraps_inclusion[t][i] = 1;
    }

    ///////////////////////////////////////////////////////////////////////
    // Reading data
    ///////////////////////////////////////////////////////////////////////

    FILE* fin = fopen64(fname, "rb");
    if (!fin) {
        cout << "# Error on openning data file: " << errno << endl;
        return 0;
    }


    time_t start_time = time(NULL);

    int iter_num = 0;

    while (true) {
        // I will repeat going through the data file
        // untill all trees are done
        //

        time_t iter_start_time = time(NULL);

        if (need_to_read()) {
            size_t read = fread(p, BUFFER_SIZE_BYTES, 1, fin);
            while (read) {
                process((int*)p, LINES);
                read = fread(p, BUFFER_SIZE_BYTES, 1, fin);
                usleep(3);

            }

            // read the rest if any
            read = fread(p, rest_bytes, 1, fin);
            if (read) {
                process((int*)p, rest_bytes / VEC_LEN_BYTES);
            }

            fseeko64(fin, 0, SEEK_SET);

            usleep(3);
        }
        else {
            static int number_of_rows = 0;
            if (!number_of_rows) {
                // filling in the buffer
                fseeko64(fin, 0, SEEK_SET);

                char buffer[VEC_LEN_BYTES];
                int offset = 0;

                size_t read = fread(buffer, VEC_LEN_BYTES, 1, fin);
                while (read) {
                    int ID = *(int*)buffer;
                    for (int t = 0; t < TREES; ++t) {
                        int bs_idx = -1;
                        if (!index_in_vec(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS], ID, bs_idx))
                            continue;
                        if (0 == bootstraps_inclusion[t][bs_idx])
                            continue;

                        memcpy(p + offset, buffer, VEC_LEN_BYTES);
                        offset += VEC_LEN_BYTES;
                        ++number_of_rows;
                    }
                    read = fread(buffer, VEC_LEN_BYTES, 1, fin);
                }
                cout << "# read the rest to memory, " << number_of_rows << " vectors" << endl;
            }

            process((int*)p, number_of_rows);
        }

        ++iter_num;

        // print time
        cout << "# iteration " << iter_num << " time " << difftime(time(NULL), iter_start_time) << " sec" << endl;


        // stop this pass and check if I need to stop
        int number_of_finished = 0;
        for (int t = 0; t < TREES; ++t) {
            stop_fit_tree(hLearners[t]);
            number_of_finished += end_of_splitting(hLearners[t]);
        }

        if (TREES == number_of_finished) {
            break;
        }

    } // while true

    cout << "# Total time of processing " << difftime(time(NULL), start_time) << " sec" << endl;

    fclose(fin);
    fin = NULL;

    ///////////////////////////////////////////////////////////////////////
    // END of reading data
    ///////////////////////////////////////////////////////////////////////


    // free the learner before exiting
    for (int t = 0; t < TREES; ++t) {
        stringstream ss;
        //ss << "/home/maxim/kaggle/CTR/rf/sub2/80_1_5_" << t << ".b";
        ss << "c:\\Temp\\kaggle\\CTR\\rf\\sub2\\60_8_51_" << t << ".b";

        int fcnt = 0;
        while (exists(ss.str().c_str())) {
            ++fcnt;
            ss.str("");
            //ss << "/home/maxim/kaggle/CTR/rf/sub2/80_1_5_" << t << "(" << fcnt << ")" << ".b";
            ss << "c:\\Temp\\kaggle\\CTR\\rf\\sub2\\60_8_51_" << t << "(" << fcnt << ")" << ".b";
        }

        cout << "# saving tree " << t << " into '" << ss.str() << "'" << endl;
        tofile_tree(hLearners[t], ss.str().c_str());

        free_tree(hLearners[t]);
    }

    return 0;

}




void process(int* matrix, int rows) {
    DATATYPE vec[VEC_LEN];

    for (int r = 0; r < rows; ++r) {
        // convert to double
        for (int c = 0; c < VEC_LEN/4; ++c) {
            int idx = r * VEC_LEN + c * 4;

            vec[c * 4 + 0] = (DATATYPE)matrix[idx + 0];
            vec[c * 4 + 1] = (DATATYPE)matrix[idx + 1];
            vec[c * 4 + 2] = (DATATYPE)matrix[idx + 2];
            vec[c * 4 + 3] = (DATATYPE)matrix[idx + 3];
        }

        int ID = vec[0];

        vec[2] = 0;
        vec[3] = 0;
        vec[4] = 0;
        vec[15] = 0;
        vec[18] = 0;

        for (int t = 0; t < TREES; ++t) {
            int bs_idx = -1;
            if (!index_in_vec(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS], ID, bs_idx))
                continue;

            if (0 == bootstraps_inclusion[t][bs_idx])
                continue;

            // feed to RF
            int drop_x = 0;
            fit_tree(hLearners[t], &vec[2], vec[1], &drop_x);

            if (drop_x)
                bootstraps_inclusion[t][bs_idx] = 0;
        }
    }
}


bool need_to_read() {
    static bool need = true;

    if (need) {
        const int cap = LINES;
        int cnt = 0;

        static int notcha = total_lines;

        for (int t = 0; t < TREES && cnt <= cap; ++t) {
            int ID = -1;
            for (int i = 0; i < LINES_IN_BS && cnt <= cap; ++i) {
                if (ID != bootstraps[t][i]) {
                    ID = bootstraps[t][i];
                    cnt += bootstraps_inclusion[t][i] == 1;
                }
            }
        }

        if (cnt <= cap) {
            need = false;
        }

        if (100000 < (notcha - cnt)) {
            notcha = cnt;
            cout << "# need vectors: " << cnt << ", cap: " << cap << endl;
        }
    }

    return need;
}

template <class ForwardIterator, class T>
bool index_in_vec(ForwardIterator first, ForwardIterator last, const T& val, int& index) {
    ForwardIterator it = lower_bound(first, last, val);
    if (it != last && *it == val) {
        index = (int)(it - first);
        return true;
    }
    return false;
}


bool exists(const char* fname) {
    FILE* fd = fopen(fname, "rb");
    bool result = (NULL != fd);
    if (fd)
        fclose(fd);
    return result;
}
