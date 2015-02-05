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


typedef float DATATYPE;
typedef float RFTYPE;


#define LOG(...) {cout << __VA_ARGS__ << endl;}

#define ALIGNMENT  __attribute__((aligned(16))


const int MAX_ACTIVES = 8;
int ACTIVES_NUM = MAX_ACTIVES;


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
    };

    int K;
    int LNUM;

    split_data sd_;

    T total_y_sum;
    int total_count;

    int node_vector_idx;

    bool active_;

public:
    dtree_node(int kf, int ln) :
        K(kf),
        LNUM(ln),
        sd_(),
        total_y_sum(0),
        total_count(0),
        active_(true)
    {}

    void set_node_vector_idx(int idx) { node_vector_idx = idx; }
    int get_node_vector_idx() const { return node_vector_idx; }

    double get_mean() const { return (double)total_y_sum / (double)total_count; }
    int get_count() const { return total_count; }

    void set_active(bool val) {active_ = val;}
    bool get_active() const {return active_;}

    void fill_vals(std::map<T, data<T> >& m, int idx) {
        double N = 100;
        double step = 1. / (N - 1);
        double val = 0.;

        for (int i = 0; i < N; ++i) {
            data<T> d;
            d.idx = idx;
            d.x_val = val;

            m[val] = d;

            val += step;
            if (val > 1.)
                val = 1.;
        }
    }

    void start_splitting(int* indices) {
        sd_.clear();

        for (int i = 0; i < K; ++i) {
            sd_.indices.push_back(indices[i]);
            sd_.accums.push_back(std::map<T, data<T> >());

            fill_vals(sd_.accums[i], sd_.indices[i]);
        }

        total_y_sum = 0;
        total_count = 0;
    }

    void process_splitting(const T* x, T y) {
        // go through all selected features and take its values
        for (int i = 0; i < K; ++i) {
            // value of the current feature
            T val = x[ sd_.indices[i] ];

            //typename std::map<T, data<T> >::iterator result = sd_.accums[i].find(val);
            typename std::map<T, data<T> >::iterator result = sd_.accums[i].lower_bound(val);

//            if (sd_.accums[i].end() == result) {
//                data<T> d;
//                d.idx = sd_.indices[i];
//                d.x_val = val;
//                d.count = 1;
//                d.y_sum = y;
//                sd_.accums[i].insert(std::pair<T, data<T> >(val, d));
//            }
//            else {
                data<T>& d = result->second;
                d.count += 1;
                d.y_sum += y;
//            }
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
            // latest element contains sums for total set

            double mean = (double)total_y_sum / (double)total_count;
            double mean_squared = mean * mean;

            // name "accumulator" is legacy one
            // as I do not accumulate values here actually
            double left_sum_accum = 0.;
            double right_sum_accum = 0.;
            double left_sum_squared_accum = 0.;
            double right_sum_squared_accum = 0.;
            int left_count_accum = 0;
            int right_count_accum = 0;

            double total_y_sum_squared = total_y_sum;

            // this will go from the smallest value to the biggest one
            for (typename std::map<T, data<T> >::iterator it = sd_.accums[i].begin(); it != sd_.accums[i].end(); ++it) {
                data<T>& d = it->second;


                left_sum_accum += d.y_sum;
                right_sum_accum = (total_y_sum - left_sum_accum);
                left_sum_squared_accum += d.y_sum;
                right_sum_squared_accum = (total_y_sum_squared - left_sum_squared_accum);

                left_count_accum += d.count;
                right_count_accum = ((int)total_count - left_count_accum);

                if (left_count_accum && right_count_accum) {

                    double left_mean = left_count_accum ? left_sum_accum / left_count_accum : 0.;
                    double right_mean = right_count_accum ? right_sum_accum / right_count_accum : 0.;

                    double left_mean_squared = left_mean * left_mean;
                    double right_mean_squared = right_mean * right_mean;

                    double e = (total_y_sum_squared + mean_squared * total_count - 2. * mean * total_y_sum);
                    double e_left = (left_sum_squared_accum + left_mean_squared * left_count_accum - 2. * left_mean * left_sum_accum);
                    double e_right = (right_sum_squared_accum + right_mean_squared * right_count_accum - 2. * right_mean * right_sum_accum);

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
            if (.4 < m && m < .6) {
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


};


template<class T, class RF=double>
class dtree {
public:
    enum {
        VEC_LEN = 4,

        DATA_IDX = 0,
        IDX_IDX = 0,
        VAL_IDX = 1,
        LEFT_IDX = 2,
        RIGHT_IDX = 3,

        CNT_IDX = 3,
    };

protected:
    // vector of length VEC_LEN: leaf [data, -2, -2, count] or intermediate node [idx, val, left, right]
    vector<RF> tree_;

public:
    dtree() : tree_() {
        tree_.reserve(3000000 * sizeof(RF));
    }

    virtual ~dtree() {}


    void tofile(const char* fname) {
        FILE* fd = fopen(fname, "wb+");
        for (int i = 0; i < tree_.size(); ++i) {
            fwrite(&tree_[i], sizeof(RF), 1, fd);
        }

        fclose(fd);
    }

    void fromfile(const char* fname) {
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
        cout << "double dtree_array[] = {" << endl;
        for (int i = 0; i < num; ++i) {
            cout << tree_[i * dtree::VEC_LEN + 0] << ","
                 << tree_[i * dtree::VEC_LEN + 1] << ","
                 << tree_[i * dtree::VEC_LEN + 2] << ","
                 << tree_[i * dtree::VEC_LEN + 3] << "," << endl;
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

            if (-2 == node[VAL_IDX]) {
                // leaf:w
                val = (double)node[DATA_IDX];
                if (cnt)
                    *cnt = (double)node[CNT_IDX];
                break;
            }
            else {
                if (-1 == x[ (int)node[IDX_IDX] ]) {
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


    int add_node(map<int, dtree_node<T> >& nodes, bool active) {
        int nodes_num = dtree<T, RF>::tree_.size() / dtree<T, RF>::VEC_LEN;
        int new_id = nodes_num;

        dtree<T, RF>::tree_.push_back(-1);    // idx
        dtree<T, RF>::tree_.push_back(-1);    // val
        dtree<T, RF>::tree_.push_back(-1);    // left
        dtree<T, RF>::tree_.push_back(-1);    // right

        // and node for this new splitting node
        int indices[K];
        random::get_k_of_n(K, COLUMN_NUMBER, &indices[0]);

        dtree_node<T> root(K, LNUM);
        root.set_active(active);
        root.set_node_vector_idx(new_id);
        root.start_splitting(indices);

        nodes.insert(pair<int, dtree_node<T> >(new_id, root));

        return new_id;
    }

    bool get_node(const T* x, dtree_node<T>** node) {
        int nodes_num = dtree<T, RF>::tree_.size() / dtree<T, RF>::VEC_LEN;
        if (!nodes_num) {
            add_node(nodes_, true);
        }

        bool result = true;
        int start_idx = 0;

        while (true) {
            RF *vec = &dtree<T, RF>::tree_[start_idx];

            if (-1 == vec[dtree<T, RF>::VAL_IDX]) {
                // found, get the object
                int ID = start_idx / dtree<T, RF>::VEC_LEN;
                typename map<int, dtree_node<T> >::iterator it = nodes_.find(ID);
                *node = &it->second;
                break;
            }
            else if (-2 == vec[dtree<T, RF>::VAL_IDX]) {
                // leaf - nothing to return for learning
                result = false;
                break;
            }
            else {
                // go down
                int idx = (int)vec[dtree<T, RF>::IDX_IDX];
                RF val = vec[dtree<T, RF>::VAL_IDX];

                if (x[idx] <= val) {
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

    int process_fit(const T* x, T y) {
        const int DROP_X = 0;
        const int ACTIVE_X = 1;
        const int NON_ACTIVE_X = 2;

        dtree_node<T>* node = NULL;
        if (get_node(x, &node)) {
            if (node->get_active()) {
                node->process_splitting(x, y);
                return ACTIVE_X;
            }
            return NON_ACTIVE_X;
        }
        return DROP_X;
    }

    void stop_splitting(int do_not_re_activate) {
        map<int, dtree_node<T> > new_nodes;

        for (typename map<int, dtree_node<T> >::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {

            int idx;
            T val;
            double gain;

            dtree_node<T>& node = it->second;
            if (node.get_active()) {
                node.stop_splitting(&idx, &val, &gain);
                if (-1 == idx) {
                    // cannot split, make a leaf
                    int id = it->first;
                    int start_idx = id * dtree<T, RF>::VEC_LEN;
                    RF* vec = &dtree<T, RF>::tree_[start_idx];

                    vec[dtree<T, RF>::VAL_IDX] = -2;
                    vec[dtree<T, RF>::LEFT_IDX] = -2;
                    vec[dtree<T, RF>::DATA_IDX] = node.get_mean();
                    vec[dtree<T, RF>::CNT_IDX] = node.get_count();

                    // sanity check
                    if (0 == node.get_count())
                        LOG("# ERROR RF count of samples is zero")
                }
                else {
                    // make an intermediate node
                    int id = it->first;
                    int start_idx = id * dtree<T, RF>::VEC_LEN;
                    RF* vec = &dtree<T, RF>::tree_[start_idx];

                    vec[dtree<T, RF>::IDX_IDX] = idx;
                    vec[dtree<T, RF>::VAL_IDX] = val;

                    int new_left_id = add_node(new_nodes, true);
                    int new_right_id = add_node(new_nodes, true);

                    dtree<T, RF>::tree_[start_idx + dtree<T, RF>::LEFT_IDX] = new_left_id;
                    dtree<T, RF>::tree_[start_idx + dtree<T, RF>::RIGHT_IDX] = new_right_id;
                }
            }
            else {
                // transfer non-active to the new collection
                new_nodes.insert(pair<int, dtree_node<T> >(it->first, node));
            }
        }

        nodes_.swap(new_nodes);

        if (0 == do_not_re_activate) {
            if (16384 <= nodes_.size()) {
                ACTIVES_NUM = 1;
            }
            else if (8192 <= nodes_.size()) {
                if (nodes_.size() > new_nodes.size()) {     // and number didnt drop
                    ACTIVES_NUM /= 2;
                }
            }
            else {
                ACTIVES_NUM *= 2;
                if (MAX_ACTIVES < ACTIVES_NUM)
                    ACTIVES_NUM = MAX_ACTIVES;
            }
            int non_active = 0;
            int num_of_active_nodes = 0;
            int nodes_to_deactivate = nodes_.size() > ACTIVES_NUM ? nodes_.size() - ACTIVES_NUM : 0;
            for (typename map<int, dtree_node<T> >::reverse_iterator it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
                dtree_node<T>& node = it->second;
                if (nodes_to_deactivate) {
                    node.set_active(false);
                    --nodes_to_deactivate;
                    ++non_active;
                }
                else {
                    node.set_active(true);
                    ++num_of_active_nodes;
                }
            }
            if (!num_of_active_nodes && nodes_.size()) {
                // just in case
                dtree_node<T>& node = nodes_.begin()->second;
                node.set_active(true);
            }
            cout << "# processed nodes: " << (dtree<T, RF>::tree_.size() / dtree<T, RF>::VEC_LEN) << "; pending: " << nodes_.size() << "; non-active: " << non_active << "; active: " << num_of_active_nodes << endl;
        }
        else {
            cout << "# no reactivation" << endl;
        }
    }

    bool need_splitting() {
        return nodes_.size();
    }
};




/*
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
*/


//
// C stype interface for Python
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
        int res = static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->process_fit(x, y);
        *drop_x = res;
    }

    void stop_fit_tree(void* ptree, int do_not_re_activate) {
        static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->stop_splitting(do_not_re_activate);
    }

    int end_of_splitting(void* ptree) {
        return (int)(false == static_cast<dtree_learner<DATATYPE, RFTYPE>*>(ptree)->need_splitting());
    }

};   // END of extern "C"












