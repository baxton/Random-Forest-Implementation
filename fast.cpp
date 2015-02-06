

//
// g++ -O3  -ftree-vectorize fast.cpp -L. -l rf -o sse_80_7_51.exe
//


#define _FILE_OFFSET_BITS  64
#define _USE_32BIT_TIME_T 1

#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <cerrno>
#include <algorithm>
#include <unistd.h>
#include <iomanip>

using namespace std;

typedef float DATATYPE;
typedef float RFTYPE;



const int ALLIGN = 512;

const int VEC_LEN = 1 + 30;
const int VEC_LEN_BYTES = VEC_LEN * sizeof(int);
const int BUFFER_SIZE_BYTES = 372000000;

char buffer[ALLIGN + BUFFER_SIZE_BYTES];


const int TOTAL_FEATURES = 30;

const int TREES   = 1;
void* hLearners[TREES];

const int COLUMNS = TOTAL_FEATURES;
const int K       = TOTAL_FEATURES / 4;
const int LNUM    = 51;

const int LINES_IN_TRAIN = 40428967;
//const int LINES_IN_BS    = 24257380;    // 60%
//const int LINES_IN_BS    = 32343173;        // 80%
//const int LINES_IN_BS    = 40428967;   // 100%
const int LINES_IN_BS      = 36386070; // 90%
//const int LINES_IN_BS    = 100;

int bootstraps[TREES][LINES_IN_BS];
vector<bool> bootstraps_inclusion[TREES]; //[LINES_IN_BS];



const int FILES_NUM = 14;
//const int FILES_NUM = 7;
FILE* fds[FILES_NUM];
const char* files[/*FILES_NUM*/] = {
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.0",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.1",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.2",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.3",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.4",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.5",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.6",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.7",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.8",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.9",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.10",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.11",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.12",
"C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\rf_train.b.13",
};



// to expedite processing
const int FINISH_NO  = 0;
const int FINISH_YES = 1;
const int FINISH_SWITCH = 2;
int finish_out = FINISH_NO;

int number_of_active = 0;
const int EXP_VECTORS_NUM = 2600000;
const int EXP_VEC_LEN = VEC_LEN;
const int EXP_BUF_SIZE = EXP_VEC_LEN * EXP_VECTORS_NUM;
int exp_ids[EXP_VECTORS_NUM];
float exp_buffer[EXP_BUF_SIZE];
int exp_idx = 0;
int exp_vectors_num = 0;


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

    /*
     * Retuns:
     * k indices out of [0-n) range
     * with no repetitions
     */
    static void get_k_of_n(int k, int n, int* numbers) {
        for (int i = 0; i < k; ++i) {
            numbers[i] = i;
        }

        for (int i = k; i < n; ++i) {
            int r = randint(0, i);
            if (r < k) {
                numbers[r] = i;
            }
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
    void stop_fit_tree(void* ptree, int do_not_reactivate);
    int end_of_splitting(void* ptree);
};   // END of extern "C"



template <class ForwardIterator, class T>
bool index_in_vec(ForwardIterator first, ForwardIterator last, const T& val, int& index);
int check_for_finish(FILE* fd, char* p);
void process(DATATYPE* matrix, int rows, int*);
void finish();
bool exists(const char* fname);


void process_file(FILE* fd, char* p, void(*func)(DATATYPE*,int, int*), int* ID) {
    fseek(fd, 0, SEEK_END);
    size_t size = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    size_t ret = fread(p, size, 1, fd);

    //usleep(3);

    func((DATATYPE*)p, size / VEC_LEN_BYTES, ID);
}


int main(int argc, const char* argv[]) {

    ///////////////////////////////////////////////////////////////////////
    //
    ///////////////////////////////////////////////////////////////////////
    for (int f = 0; f < FILES_NUM; ++f) {
        fds[f] = fopen(files[f], "rb");
        if (!fds[f]) {
            cout << "# ERROR cannot open file " << f << ": " << errno << endl;
        }
        setbuf(fds[f], NULL);
    }

    ///////////////////////////////////////////////////////////////////////
    // prepare learner - random forest
    ///////////////////////////////////////////////////////////////////////
    for (int t = 0; t < TREES; ++t) {
        hLearners[t] = alloc_tree_learner(COLUMNS, K, LNUM);
        start_fit_tree(hLearners[t]);
    }

    ///////////////////////////////////////////////////////////////////////
    // Allign the read buffer
    ///////////////////////////////////////////////////////////////////////
    char* p = (char*)buffer;
    while (0 != ((int)p % ALLIGN)) {
        ++p;
    }

    ///////////////////////////////////////////////////////////////////////
    // prepare bootstraps
    ///////////////////////////////////////////////////////////////////////
    random::seed();
    for (int t = 0; t < TREES; ++t) {
        random::randint(0, LINES_IN_TRAIN, &bootstraps[t][0], LINES_IN_BS);
        //random::get_k_of_n(LINES_IN_BS, LINES_IN_TRAIN, &bootstraps[t][0]); // w/o repetitions
        sort(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS]);

        for (int i = 0; i < LINES_IN_BS; ++i)
            bootstraps_inclusion[t].push_back(1);
    }

    ///////////////////////////////////////////////////////////////////////
    // Reading data
    ///////////////////////////////////////////////////////////////////////

    time_t start_time = time(NULL);

    int iter_num = 0;

    while (true) {
        // I will repeat going through the data file
        // untill all trees are done
        //

        time_t iter_start_time = time(NULL);

        number_of_active = 0;

        int ID = -1;

        for (int f = 0; f < FILES_NUM; ++f) {
            process_file(fds[f], p, process, &ID);
        }

        ++iter_num;

        // print time
        cout << "# iteration " << iter_num << " time " << difftime(time(NULL), iter_start_time) << " sec; active vectors " << number_of_active << endl;

        // stop this pass and check if I need to stop
        int number_of_finished = 0;
        for (int t = 0; t < TREES; ++t) {
            int no_re_activate = (finish_out == FINISH_YES);
            stop_fit_tree(hLearners[t], no_re_activate);
            if (1 == end_of_splitting(hLearners[t])) {
                ++number_of_finished;
            }
        }

        if (TREES == number_of_finished) {
            break;
        }


        if (EXP_VECTORS_NUM >= number_of_active) {
            if (FINISH_YES == finish_out) {
                finish();
                finish_out = FINISH_SWITCH;
            }
            else if (FINISH_NO == finish_out) {
                finish_out = FINISH_YES;
                exp_idx = 0;
                exp_vectors_num = 0;
            }
            else if (FINISH_SWITCH == finish_out) {
                finish_out = FINISH_NO;
            }
        }

    } // while true

    cout << "# Total time of processing " << difftime(time(NULL), start_time) << " sec" << endl;

    for (int f = 0; f < FILES_NUM; ++f) {
        fclose(fds[f]);
        fds[f] = NULL;
    }

    ///////////////////////////////////////////////////////////////////////
    // END of reading data
    ///////////////////////////////////////////////////////////////////////


    // free the learner before exiting
    for (int t = 0; t < TREES; ++t) {
        stringstream ss;
        ss << "C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\step2_80_7_51_" << t << ".b";
        //ss << "c:\\Temp\\kaggle\\CTR\\rf\\sub2\\log_f_90_8_11_" << t << ".b";

        int fcnt = 0;
        while (exists(ss.str().c_str())) {
            ++fcnt;
            ss.str("");
            ss << "C:\\Temp\\kaggle\\CTR\\rf\\rf_train\\step2_80_7_51_" << t << "(" << fcnt << ")" << ".b";
            //ss << "c:\\Temp\\kaggle\\CTR\\rf\\sub2\\log_f_90_8_5_" << t << "(" << fcnt << ")" << ".b";
        }

        cout << "# saving tree " << t << " into '" << ss.str() << "'" << endl;
        tofile_tree(hLearners[t], ss.str().c_str());

        free_tree(hLearners[t]);
    }

    return 0;

}




void process(DATATYPE* matrix, int rows, int* ID) {
    DATATYPE* vec = (DATATYPE*)&matrix[0];

    for (int r = 0; r < rows; ++r) {
        int idx = r * VEC_LEN;

        *ID += 1;

        for (int t = 0; t < TREES; ++t) {
            int bs_idx = -1;
            if (!index_in_vec(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS], *ID, bs_idx))
                continue;

            if (0 == bootstraps_inclusion[t][bs_idx])
                continue;

            // feed to RF
            const int DROP_X = 0;
            const int ACTIVE_X = 1;
            const int NON_ACTIVE_X = 2;

            int drop_x = ACTIVE_X;
            while (bootstraps[t][bs_idx] == *ID && (DROP_X != drop_x)) {
                fit_tree(hLearners[t], &vec[idx+1], vec[idx+0], &drop_x);
                if (DROP_X != drop_x)
                    ++bs_idx;
            }

            if (DROP_X == drop_x) {
                bootstraps_inclusion[t][bs_idx] = 0;
            }
            else if (ACTIVE_X == drop_x) {
                ++number_of_active; // unique IDs only

                if (FINISH_YES == finish_out) {
                    if (number_of_active > EXP_VECTORS_NUM)
                        finish_out = FINISH_NO;

                    ///cout << "# ID for fast " << *ID << " row index " << (exp_idx / EXP_VEC_LEN) << endl;
                    exp_ids[exp_idx/EXP_VEC_LEN] = *ID;
                    for (int i = 0; i < VEC_LEN; ++i)
                        exp_buffer[exp_idx+i] = vec[idx+i];
                    exp_idx += EXP_VEC_LEN;
                    ++exp_vectors_num;
                }
            }
        }
    }
}



void finish() {
    while (true) {

        int dropped_vectors = 0;

        int non_active_num = 0;
        int actives_num = 0;

        for (int r = 0; r < exp_vectors_num; ++r) {
            int idx = r * EXP_VEC_LEN;

            int ID = exp_ids[r];



            for (int t = 0; t < TREES; ++t) {
                int bs_idx = -1;
                if (!index_in_vec(&bootstraps[t][0], &bootstraps[t][LINES_IN_BS], ID, bs_idx)) {
                /*
                    cout << "# ERROR not for finish " << ID << " row index " << r << endl;

                    for (int i = 0; i < EXP_VEC_LEN; ++i)
                        cout << setprecision(16) << exp_buffer[idx-EXP_VEC_LEN+i] << ", ";
                    cout << endl;
                    for (int i = 0; i < EXP_VEC_LEN; ++i)
                        cout << setprecision(16) << exp_buffer[idx+i] << ", ";
                    cout << endl;
                    for (int i = 0; i < EXP_VEC_LEN; ++i)
                        cout << setprecision(16) << exp_buffer[idx+EXP_VEC_LEN+i] << ", ";
                    cout << endl;

                    cout << "# bootstrap: " << endl;
                    for (int i = 0; i < LINES_IN_BS; ++i) {
                        cout << bootstraps[t][i] << "\n";
                    }
                    cout << endl;
                    abort();
                */
                    continue;
                }

                if (0 == bootstraps_inclusion[t][bs_idx]) {
                    ++dropped_vectors;
                    continue;
                }

                // feed to RF
                const int DROP_X = 0;
                const int ACTIVE_X = 1;
                const int NON_ACTIVE_X = 2;

                int drop_x = ACTIVE_X;
                while (bootstraps[t][bs_idx] == ID && (DROP_X != drop_x)) {
                    fit_tree(hLearners[t], &exp_buffer[idx+1], exp_buffer[idx], &drop_x);
                    if (DROP_X != drop_x)
                        ++bs_idx;
                }

                if (DROP_X == drop_x) {
                    bootstraps_inclusion[t][bs_idx] = 0;
                    ++dropped_vectors;
                }
                else if (ACTIVE_X == drop_x) {
                    ++actives_num;
                }
                else if (NON_ACTIVE_X == drop_x) {
                    ++non_active_num;
                }
            }
        }

        cout << "# vectors to drop " << exp_vectors_num << "; curently dropped " << dropped_vectors << "; actives " << actives_num << "; non actives " << non_active_num << endl;

        for (int t = 0; t < TREES; ++t) {
            stop_fit_tree(hLearners[t], true);
        }

        if (dropped_vectors == exp_vectors_num)
            break;
    }
    cout << "# finished " << number_of_active << " (" << exp_vectors_num << ") vectors" << endl;
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








