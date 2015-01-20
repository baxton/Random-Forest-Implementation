//
// g++ -O3 RF.cpp -shared -o rf.dll
// g++ RF.cpp -o rf.exe
//
// g++ -O3 RF.cpp -shared -o librf.so
//


#include <cstdlib>
#include <time.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include<sstream>
#include <fstream>
#include <map>
#include <set>



using namespace std;


typedef int DATATYPE;
typedef float RFTYPE;


#define LOG(...) {cout << __VA_ARGS__ << endl;}

#define ALIGNMENT  __attribute__((aligned(16))


const RFTYPE NOVAL = -999999999;


//
// data
//
template<class T>
struct data {
    data() :
        count(0),
        idx(-1),
        x_val(-1),
        y_sum(0)
    {}

    int count;
    int idx;
    T x_val;
    T y_sum;


    void tofile(FILE* fd) const {
        fwrite(&count, sizeof(int), 1, fd);
        fwrite(&idx, sizeof(int), 1, fd);
        fwrite(&x_val, sizeof(T), 1, fd);
        fwrite(&y_sum, sizeof(T), 1, fd);
    }

    void fromfile(FILE* fd) {
        fread(&count, sizeof(int), 1, fd);
        fread(&idx, sizeof(int), 1, fd);
        fread(&x_val, sizeof(T), 1, fd);
        fread(&y_sum, sizeof(T), 1, fd);
    }
};



//
// Memory management
//
namespace memory {

#define MEM_SIZE 2000000

}   // memory

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
#if (0x7FFF < RAND_MAX)
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
        for (int i = 0; i < size; ++i)
            numbers[i] = randint() % (high - low) + low;
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
// Fast dtree
//

template<class T>
class dtree_node {

    struct split_data {
        vector<int> indices;
        vector<std::map<T, data<T> > > accums;

        void clear() {
            indices.clear();
            accums.clear();
        }

        void tofile(FILE* fd) const {
            // format: <K><K indices><K maps>
            // map format: <num elements><elements>
            // elment: "data<T>" structure

            int K = indices.size();
            fwrite(&K, sizeof(int), 1, fd);

            // indices
            for (int i = 0; i < K; ++i) {
                fwrite(&indices[i], sizeof(int), 1, fd);
            }

            // accumulators - every index has a set of accumulators according to the number of values
            for (int i = 0; i < K; ++i) {
                int size = accums[i].size();
                fwrite(&size, sizeof(int), 1, fd);

                for (typename std::map<T, data<T> >::const_iterator it = accums[i].begin(); it != accums[i].end(); ++it) {
                    const data<T>& d = it->second;
                    d.tofile(fd);
                }
            }
        }

        void fromfile(FILE* fd) {
            // format: <K><K indices><K maps>
            // map format: <num elements><elements>
            // elment: "data<T>" structure

            clear();

            int K = 0;
            fread(&K, sizeof(int), 1, fd);

            for (int i = 0; i < K; ++i) {
                int tmp;
                fread(&tmp, sizeof(int), 1, fd);
                indices.push_back(tmp);
            }
            for (int i = 0; i < K; ++i) {
                int size;
                fread(&size, sizeof(int), 1, fd);
                if (size) {
                    accums.push_back(std::map<T, data<T> >());
                    for (int m = 0; m < size; ++m) {
                        data<T> d;
                        d.fromfile(fd);
                        accums[i][d.x_val] = d;
                    }
                }
            }
        }
    };

    int K;
    int LNUM;

    split_data sd_;

    T total_y_sum;
    int total_count;

    int node_vector_idx;

public:
    dtree_node(int kf, int ln) :
        K(kf),
        LNUM(ln),
        sd_(),
        total_y_sum(0),
        total_count(0),
        node_vector_idx(-1)
    {}

    void set_node_vector_idx(int idx) { node_vector_idx = idx; }
    int get_node_vector_idx() const { return node_vector_idx; }

    double get_mean() const { return (double)total_y_sum / (double)total_count; }
    int get_count() const { return total_count; }


    void tofile(FILE* fd) const {
        // format: <K><LNUM><node idx><sd><total_y_sum><total_count>
        // sd: "split_data" structure

        fwrite(&K, sizeof(int), 1, fd);
        fwrite(&LNUM, sizeof(int), 1, fd);
        fwrite(&node_vector_idx, sizeof(int), 1, fd);
        sd_.tofile(fd);
        fwrite(&total_y_sum, sizeof(T), 1, fd);
        fwrite(&total_count, sizeof(int), 1, fd);
    }

    void fromfile(FILE* fd) {
        // format: <K><LNUM><node idx><sd><total_y_sum><total_count>
        // sd: "split_data" structure

        fread(&K, sizeof(int), 1, fd);
        fread(&LNUM, sizeof(int), 1, fd);
        fread(&node_vector_idx, sizeof(int), 1, fd);
        sd_.fromfile(fd);
        fread(&total_y_sum, sizeof(T), 1, fd);
        fread(&total_count, sizeof(int), 1, fd);
    }

    void start_splitting(int* indices) {
        sd_.clear();

        for (int i = 0; i < K; ++i) {
            sd_.indices.push_back(indices[i]);
            sd_.accums.push_back(std::map<T, data<T> >());
        }

        total_y_sum = 0;
        total_count = 0;
    }

    void process_splitting(const T* x, T y) {
        // go through all selected features and take its values
        for (int i = 0; i < K; ++i) {
            // value of the current feature
            T val = x[ sd_.indices[i] ];

            typename std::map<T, data<T> >::iterator result = sd_.accums[i].find(val);

            if (sd_.accums[i].end() == result) {
                data<T> d;
                d.idx = sd_.indices[i];
                d.x_val = val;
                d.count = 1;
                d.y_sum = y;
                sd_.accums[i].insert(std::pair<T, data<T> >(val, d));
            }
            else {
                data<T>& d = result->second;
                d.count += 1;
                d.y_sum += y;
            }
        }

        total_y_sum += y;
        total_count += 1;
    }

    void stop_splitting(int* idx, T* val, double* gain) {

        int can_idx = -1;
        T can_val = -1.;
        double can_gain = -1.;

        int best_idx = -1;
        T best_val = -1.;
        double best_gain = -1.;

        for (int i = 0; i < K; ++i) {

            double mean = (double)total_y_sum / (double)total_count;

            // name "accumulator" is legacy one
            // as I do not accumulate values here actually
            double left_sum_accum = 0.;
            double right_sum_accum = 0.;
            int left_count_accum = 0;
            int right_count_accum = 0;

            // this will go from the smallest value to the biggest one
            for (typename std::map<T, data<T> >::iterator it = sd_.accums[i].begin(); it != sd_.accums[i].end(); ++it) {
                data<T>& d = it->second;

                left_sum_accum += d.y_sum;                           // equal to val go to left
                right_sum_accum = (total_y_sum - left_sum_accum);    // not equal to val go to right

                left_count_accum += d.count;
                right_count_accum = ((int)total_count - left_count_accum);

                if (left_count_accum && right_count_accum) {
                    double left_mean = left_count_accum ? left_sum_accum / left_count_accum : 0.;
                    double right_mean = right_count_accum ? right_sum_accum / right_count_accum : 0.;

                    double e = entropy(mean);
                    double e_left = entropy(left_mean);
                    double e_right = entropy(right_mean);


                    double g = e - ((double)left_count_accum / (double)total_count * e_left + (double)right_count_accum / (double)total_count * e_right);
                    // sanity check
                    if (isnan(g)) {
                        LOG("# ERROR: gain is nan, feature: " << i << ", feature idx: " << d.idx)
                        LOG("# ERROR: left count: " << left_count_accum << ", right count: " << right_count_accum)
                    }

                    if (g > 0. && g > best_gain) {
                        if (LNUM <= left_count_accum && LNUM <= right_count_accum) {
                            best_idx = d.idx;
                            best_val = d.x_val;
                            best_gain = g;
                        }
                    }

                    if (g > 0. && g > can_gain) {
                        can_idx = d.idx;
                        can_val = d.x_val;
                        can_gain = g;
                    }
                }
            }
        }

        if (-1 == best_idx && -1 != can_idx) {
            double m = get_mean();
            if (.3 < m && m < .7) {
                // I think this is too much mixed, so let's try to split it more
                best_idx = can_idx;
                best_val = can_val;
                best_gain = can_gain;
            }
        }

        *idx = best_idx;
        *val = best_val;
        *gain = best_gain;

        sd_.clear();
    }

    double entropy(double mean) {

        double p0 = 1. - mean;
        double p1 = mean;

        p0 = p0 > 0. ? (-p0 * log(p0)) : 0.;
        p1 = p1 > 0. ? (-p1 * log(p1)) : 0.;

        return p0 + p1;
    }

};


template<class T, class RF>
class dtree {
public:
    enum {
        LEAF = 0,
        NON_LEAF = 1,
        SPLITTING = 2,

        VEC_LEN = 6,
        DATA_LEN = 0,

        ID_IDX = 0,
        CLS_IDX = 1,
        DATA_IDX = 2,
        IDX_IDX = 2,
        VAL_IDX = 3,
        LEFT_IDX = 4,
        RIGHT_IDX = 5,

        CNT_IDX = 5,
    };

protected:
    // vector of length VEC_LEN: leaf [node_id, class, data, ...] or intermediate node [node_id, class, idx, val, left, right]
    vector<RF> tree_;

public:
    dtree() : tree_() {
        tree_.reserve(3000000 * sizeof(RF));
    }

    virtual ~dtree() {}


    virtual void tofile(const char* fname) {
        FILE* fd = fopen(fname, "wb+");
        for (int i = 0; i < tree_.size(); ++i) {
            fwrite(&tree_[i], sizeof(RF), 1, fd);
        }

        fclose(fd);
    }

    virtual void fromfile(const char* fname) {
        tree_.clear();

        FILE* fd = fopen(fname, "rb");
        while(!feof(fd)) {
            RF d;
            fread(&d, sizeof(RF), 1, fd);
            tree_.push_back(d);
        }
        fclose(fd);
    }

    void print() {
        int num = tree_.size() / dtree::VEC_LEN;

        cout << "// DTree array" << endl;
        cout << "int dtree_array_size = " << tree_.size() << ";" << endl;
        cout << "float dtree_array[] = {" << endl;
        for (int i = 0; i < num; ++i) {
            cout << tree_[i * dtree::VEC_LEN + 0] << ","
                 << tree_[i * dtree::VEC_LEN + 1] << ","
                 << tree_[i * dtree::VEC_LEN + 2] << ","
                 << tree_[i * dtree::VEC_LEN + 3] << ","
                 << tree_[i * dtree::VEC_LEN + 4] << ","
                 << tree_[i * dtree::VEC_LEN + 5] << "," << endl;
        }
        cout << "};" << endl;
        cout << "// END of DTree array" << endl;
    }

    double predict(const T* x, int start_idx=0, double* cnt=NULL) {
        if (!tree_.size())
            return 0.;

        double val = 0.;

        while (true) {
            RF* node = &tree_[start_idx];

            if (dtree::LEAF == node[CLS_IDX]) {
                val = (double)node[DATA_IDX];
                if (cnt)
                    *cnt = (double)node[CNT_IDX];
                break;
            }
            else {
                if (NOVAL == x[ (int)node[IDX_IDX] ]) {
                    double left_cnt = 0.;
                    double pl = predict(x, (int)node[LEFT_IDX] * dtree::VEC_LEN, &left_cnt);
                    double right_cnt = 0.;
                    double pr = predict(x, (int)node[RIGHT_IDX] * dtree::VEC_LEN, &right_cnt);
                    val = (left_cnt * pl + right_cnt * pr) / (left_cnt + right_cnt);
                    if (cnt)
                        *cnt = left_cnt + right_cnt;
                    break;
                }
                else if (x[ (int)node[IDX_IDX] ] <= node[VAL_IDX]) {
                    start_idx = (int)node[LEFT_IDX] * dtree::VEC_LEN;
                }
                else {
                    start_idx = (int)node[RIGHT_IDX] * dtree::VEC_LEN;
                }
            }
        }

        return val;
    }

};


template<class T, class RF=double>
class dtree_learner : public dtree<T, RF> {
    map<int, dtree_node<T> > nodes_;

    int COLUMN_NUMBER;
    int K;
    int LNUM;


    int add_node(map<int, dtree_node<T> >& nodes) {
        int nodes_num = dtree<T, RF>::tree_.size() / dtree<T, RF>::VEC_LEN;
        int new_id = nodes_num;

        dtree<T, RF>::tree_.push_back(new_id);
        dtree<T, RF>::tree_.push_back(dtree<T, RF>::SPLITTING);
        dtree<T, RF>::tree_.push_back(-1);    // idx
        dtree<T, RF>::tree_.push_back(-1);    // val
        dtree<T, RF>::tree_.push_back(-1);    // left
        dtree<T, RF>::tree_.push_back(-1);    // right

        // and node for this new splitting node
        int indices[K];
        random::get_k_of_n(K, COLUMN_NUMBER, &indices[0]);

        dtree_node<T> root(K, LNUM);
        root.set_node_vector_idx(new_id);
        root.start_splitting(indices);

        nodes.insert(pair<int, dtree_node<T> >(new_id, root));

        return new_id;
    }

    bool get_node(const T* x, dtree_node<T>** node) {
        int nodes_num = dtree<T, RF>::tree_.size() / dtree<T, RF>::VEC_LEN;
        if (!nodes_num) {
            add_node(nodes_);
        }

        bool result = true;
        int start_idx = 0;

        while (true) {
            RF *vec = &dtree<T, RF>::tree_[start_idx];

            if (dtree<T, RF>::SPLITTING == vec[dtree<T, RF>::CLS_IDX]) {
                // found, get the object
                typename map<int, dtree_node<T> >::iterator it = nodes_.find(vec[dtree<T, RF>::ID_IDX]);
                *node = &it->second;
//cout << "# FOUND NODE " << vec[dtree<T, RF>::ID_IDX] << endl;
                break;
            }
            else if (dtree<T, RF>::LEAF == vec[dtree<T, RF>::CLS_IDX]) {
                // nothing to return for learning
                result = false;
//cout << "# LEAF " << vec[dtree<T, RF>::ID_IDX] << endl;
                break;
            }
            else {
                // go down
                int idx = (int)vec[dtree<T, RF>::IDX_IDX];
                RF val = vec[dtree<T, RF>::VAL_IDX];

                if (x[idx] == val) {
                    start_idx = (int)vec[dtree<T, RF>::LEFT_IDX] * dtree<T, RF>::VEC_LEN;
                }
                else {
                    start_idx = (int)vec[dtree<T, RF>::RIGHT_IDX] * dtree<T, RF>::VEC_LEN;
                }
            }
        }

        return result;
    }

public:
    dtree_learner(int column_number, int kf, int lnum) :
        nodes_(),
        COLUMN_NUMBER(column_number),
        K(kf),
        LNUM(lnum)
    {}

    void start_fit() {
        random::seed();
        nodes_.clear();
    }


    virtual void tofile(const char* fname) {
        FILE* fd = fopen(fname, "wb+");

        fwrite(&COLUMN_NUMBER, sizeof(int), 1, fd);
        fwrite(&K, sizeof(int), 1, fd);
        fwrite(&LNUM, sizeof(int), 1, fd);

        int size = nodes_.size();
        fwrite(&size, sizeof(int), 1, fd);

        for (typename map<int, dtree_node<T> >::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
            const dtree_node<T>& n = it->second;
            n.tofile(fd);
        }

        size = dtree<T, RF>::tree_.size();
        fwrite(&size, sizeof(int), 1, fd);

        for (int i = 0; i < size; ++i) {
            fwrite(&dtree<T, RF>::tree_[i], sizeof(RF), 1, fd);
        }

        fclose(fd);
    }

    virtual void fromfile(const char* fname) {
        // reset everything
        random::seed();
        nodes_.clear();
        dtree<T, RF>::tree_.clear();

        // read from file
        FILE* fd = fopen(fname, "rb");

        fread(&COLUMN_NUMBER, sizeof(int), 1, fd);
        fread(&K, sizeof(int), 1, fd);
        fread(&LNUM, sizeof(int), 1, fd);

        int size;
        fread(&size, sizeof(int), 1, fd);

        for (int i = 0; i < size; ++i) {
            dtree_node<T> n(K, LNUM);
            n.fromfile(fd);
            nodes_.insert(pair<int, dtree_node<T> >(n.get_node_vector_idx(), n));
        }

        fread(&size, sizeof(int), 1, fd);
        for (int i = 0; i < size; ++i) {
            RF tmp;
            fread(&tmp, sizeof(RF), 1, fd);
            dtree<T, RF>::tree_.push_back(tmp);
        }

        fclose(fd);
    }

    bool process_fit(const T* x, T y) {
        dtree_node<T>* node = NULL;
        if (get_node(x, &node)) {
            node->process_splitting(x, y);
            return true;
        }
        return false;
    }

    void stop_splitting() {
        map<int, dtree_node<T> > new_nodes;

        for (typename map<int, dtree_node<T> >::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
            int idx;
            T val;
            double gain;

            dtree_node<T>& node = it->second;

            node.stop_splitting(&idx, &val, &gain);
            if (-1 == idx) {
                // cannot split, make a leaf
                int id = it->first;
                int start_idx = id * dtree<T, RF>::VEC_LEN;
                RF* vec = &dtree<T, RF>::tree_[start_idx];

                vec[dtree<T, RF>::CLS_IDX] = dtree<T, RF>::LEAF;
                vec[dtree<T, RF>::DATA_IDX] = node.get_mean();
                vec[5] = node.get_count();

                // sanity check
                if (0 == node.get_count())
                    LOG("# ERROR RF count of samples is zero")
            }
            else {
                // make an intermediate node
                int id = it->first;
                int start_idx = id * dtree<T, RF>::VEC_LEN;
                RF* vec = &dtree<T, RF>::tree_[start_idx];

                vec[dtree<T, RF>::CLS_IDX] = dtree<T, RF>::NON_LEAF;
                vec[dtree<T, RF>::IDX_IDX] = idx;
                vec[dtree<T, RF>::VAL_IDX] = val;

                int new_left_id = add_node(new_nodes);
                int new_right_id = add_node(new_nodes);

                dtree<T, RF>::tree_[start_idx + dtree<T, RF>::LEFT_IDX] = new_left_id;
                dtree<T, RF>::tree_[start_idx + dtree<T, RF>::RIGHT_IDX] = new_right_id;
            }
        }

        nodes_.swap(new_nodes);
    }

    bool need_splitting() {
        return (0 == dtree<T, RF>::tree_.size()) || nodes_.size();
    }
};







/////////////////////////////////////////////////////////////////////////
// Testing
/////////////////////////////////////////////////////////////////////////



extern "C" {
    void predict(void* ptree, DATATYPE* x, double* p);
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



void test1() {

    cout << "TEST 1" << endl;

    const int N = 10;
    int C = 25;

    DATATYPE X[] = {
        14, 10, 21, 0, 6, 11, 600, 12094, 12502, 20410, 21332, 21637, 1801945, 8539971, 9439754, 9445819, 9445824, 9446014, 9448457, 9448467, 9448500, 9448905, 9448941, 9448977, 9449201, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 600, 12094, 12502, 20410, 21332, 21637, 1801945, 6666616, 9441179, 9445819, 9445823, 9446012, 9448457, 9448467, 9448500, 9448905, 9448941, 9449042, 9449201, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 600, 12094, 12502, 20410, 21332, 21637, 1801945, 7436606, 9442027, 9445819, 9445823, 9446012, 9448457, 9448467, 9448500, 9448905, 9448941, 9449042, 9449201, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 600, 12094, 12502, 20410, 21332, 21637, 1801945, 8811694, 9440730, 9445819, 9445823, 9446014, 9448457, 9448467, 9448500, 9448905, 9448941, 9449042, 9449201, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 12, 4713, 9211, 12500, 20410, 21332, 21637, 1801945, 6660464, 9441402, 9445819, 9445823, 9446319, 9448457, 9448467, 9448553, 9448905, 9448941, 9448977, 9449164, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 3966, 10472, 12524, 20410, 21332, 21637, 1801945, 2842725, 9442027, 9445819, 9445823, 9446065, 9448457, 9448467, 9448520, 9448905, 9448951, 9449036, 9449159, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 2696, 5948, 12524, 20410, 21332, 21637, 1801945, 7399073, 9443645, 9445819, 9445823, 9446468, 9448457, 9448467, 9448595, 9448905, 9448943, 9448977, 9449164, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 12, 4176, 8648, 12524, 20410, 21332, 21637, 1801945, 8780283, 9443646, 9445819, 9445823, 9446506, 9448457, 9448467, 9448605, 9448908, 9448943, 9448977, 9449181, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 6, 11, 600, 12094, 12502, 20410, 21332, 21637, 1801945, 4178230, 9440563, 9445819, 9445824, 9446015, 9448457, 9448467, 9448500, 9448905, 9448941, 9448977, 9449201, 0, 0, 0, 0, 0,
        14, 10, 21, 0, 5, 11, 2492, 10755, 12506, 20410, 21332, 21637, 2071596, 9061602, 9439317, 9445818, 9445823, 9446746, 9448457, 9448467, 9448656, 9448908, 9448923, 9449122, 9449181, 0, 0, 0, 0, 0,
    };
    //double Y[] = {0., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    DATATYPE Y[] = {0., 1., 1., 1., 0., 0., 1., 1., 0., 1.};

//    dtree_learner t(30, 5, 1);
//    t.start_fit();

    void* t = alloc_tree_learner(30, 5, 1);
    start_fit_tree(t);

    int indices[N];
    for (int i = 0; i < N; ++i)
        indices[i] = i;

    while (true) {
        for (int i = 0; i < N; ++i) {
            int idx = indices[i];
            if (-1 == idx)
                continue;
            //bool res = t.process_fit(&X[idx*C], Y[idx]);
            int res = 0;
            fit_tree(t, &X[idx*C], Y[idx], &res);
            if (res) {
                indices[i] = -1;
                LOG("index " << idx << " will be dropped")
            }
        }
        //t.stop_splitting();
        //if (!t.need_splitting()) {
        stop_fit_tree(t);
        if (end_of_splitting(t)) {
            break;
        }
    }

    DATATYPE* v = &X[0*C];
    //double p = t.predict(v);
    double p = 0.;
    predict(t, v, &p);
    LOG("predict(0) == " << p << " vs 0")

    v = &X[1*C];
    //p = t.predict(v);
    predict(t, v, &p);
    LOG("predict(1) == " << p << " vs 1")

    v = &X[6*C];
    //p = t.predict(v);
    predict(t, v, &p);
    LOG("predict(6) == " << p << " vs 1")

    //t.print();
    print_tree(t);

    //t.tofile("rf.b");
}


int main() {

    random::seed();

    test1();

    return 0;
}

//
// C type interface for Python
//

extern "C" {

    void predict(void* ptree, DATATYPE* x, double* p) {
        *p = static_cast<dtree<DATATYPE, RFTYPE>*>(ptree)->predict(x, 0);
    }

    void* alloc_tree_learner(int columns, int k, int lnum) {
        return new dtree_learner<DATATYPE, RFTYPE>(columns, k, lnum);
    }

    void* alloc_tree() {
        return new dtree<DATATYPE, RFTYPE>();
    }

    void free_tree(void* ptree) {
        delete (dtree<DATATYPE, RFTYPE>*)ptree;
    }

    void fromfile_tree(void* ptree, const char* fname) {
        static_cast<dtree<DATATYPE, RFTYPE>*>(ptree)->fromfile(fname);
    }

    void tofile_tree(void* ptree, const char* fname) {
        static_cast<dtree<DATATYPE, RFTYPE>*>(ptree)->tofile(fname);
    }

    void print_tree(void* ptree) {
        static_cast<dtree<DATATYPE, RFTYPE>*>(ptree)->print();
    }

    void start_fit_tree(void* ptree) {
        static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->start_fit();
    }

    void fit_tree(void* ptree, DATATYPE* x, DATATYPE y, int* drop_x) {
        //*drop_x =  (int)(false == static_cast<dtree_learner*>(ptree)->process_fit(x, y));
        bool res = static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->process_fit(x, y);
        *drop_x = (int)(false == res);
    }

    void stop_fit_tree(void* ptree) {
        static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->stop_splitting();
    }

    int end_of_splitting(void* ptree) {
        return (int)(false == static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->need_splitting());
    }

};   // END of extern "C"











