//
// g++ RF_sparse.cpp -shared -o rf.dll
// g++ RF_sparse.cpp -o rf.exe
//


//
// Random Forest specially adapted to work with sparse vectors
//
// Vector:
// [ year, month, day, hour, <9M catecorical features> ]
//
// Where:
// <9M categorical features> is just about 20 indices, each index means the feature is set (equal to "1")
// i.e. this is the sparse part of the vector
// This part _must_ be sorted
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


#define LOG(...) {cout << __VA_ARGS__ << endl;}



#define ALIGNMENT  __attribute__((aligned(16))

// 9449205 total sparse features
#define TOTAL_SPARSE_FEATURES 9449205
#define TOTAL_FEATURES (4 + TOTAL_SPARSE_FEATURES)
#define SPARSE_PART_SIZE 21
#define F_IDX 4


//
// Utils
//



namespace utils {

double abs(double d) {return d > 0. ? d : -d;}

bool equal(double v1, double v2, double e = 0.0001) {
    if (utils::abs(v1 - v2) < e)
        return true;
    return false;
}

}   // utils






//
// Memory management
//
namespace memory {

template<class T>
struct DESTRUCTOR {
    static void destroy(T* p) {delete p;}
};

template<class T>
struct DESTRUCTOR_ARRAY {
    static void destroy(T* p) {delete [] p;}
};



template<class T, class D=DESTRUCTOR_ARRAY<T> >
struct ptr {
    T* p_;
    int* ref_;

    ptr() : p_(NULL), ref_(NULL) {}
    ptr(T* p) : p_(p), ref_(new int(1)) {}
    ptr(const ptr& other) {
        p_ = other.p_;
        ref_ = other.ref_;
        if (ref_)
            ++(*ref_);
    }

    ~ptr() {free();}

    ptr& operator=(const ptr& other) {
        ptr tmp = other;
        swap(tmp);
        return *this;
    }

    T* get() {return p_;}

    T* operator->() {return p_;}

    T& operator[] (int i) {
        return get()[i];
    }

    void free() {
        if (ref_) {
            if (0 == --(*ref_)) {
                D::destroy(p_);
                delete ref_;

                p_ = NULL;
                ref_ = NULL;
            }
        }
    }

    void swap(ptr& other) {
        T* tmp = p_;
        p_ = other.p_;
        other.p_ = tmp;

        int* tmp_i = ref_;
        ref_ = other.ref_;
        other.ref_ = tmp_i;
    }
};




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
        return rand();
    }

    static int randint(int low, int high) {
        int r = rand();
        r = r % (high - low) + low;
        return r;
    }

    static void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i)
            numbers[i] = rand() % (high - low) + low;
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

    template<class T>
    static void shuffle(T* buffer, int size) {
        for (int i = 0; i < size; ++i) {
            int r = randint(0, size);

            T tmp = buffer[i];
            buffer[i] = buffer[r];
            buffer[r] = tmp;
        }
    }

};


//
// Lin algebra
//
struct linalg {

    static void bootstrap(const double* __restrict__ X,
                          const double* __restrict__ Y,
                          int rows,
                          int columns,
                          double* __restrict__ bs_X,
                          double* __restrict__ bs_Y,
                          int bs_rows) {
        random::seed();
        int indices[bs_rows];
        random::randint(0, rows, indices, bs_rows);

        for (int x = 0; x < bs_rows; ++x) {
            copy(&X[ indices[x] * columns ], &bs_X[ x * columns ], columns);
            bs_Y[x] = Y[ indices[x] ];
        }
    }

    static void range(int low, int high, int* buffer) {
        for (int i = low; i < high; ++i) {
            buffer[i] = i;
        }
    }


    static void linspace(double min, double max, int num, double* buffer) {
        double delta = (max - min) / (num - 1.);

        buffer[0] = min;
        for (int i = 1; i < num-1; ++i) {
            buffer[i] = buffer[i-1]+delta;
        }
        buffer[num-1] = max;
    }

    // op with scalar
    static void mul(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] * scalar;
        }
    }

    static void div(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] / scalar;
        }
    }

    static void sub(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] - scalar;
        }
    }

    static void add(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v[i] + scalar;
        }
    }

    static void pow(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = ::pow(v[i], scalar);
        }
    }

    static void pow_inplace(double scalar, double* __restrict__ v, int size) {
        for (int i = 0; i < size; ++i) {
            v[i] = ::pow(v[i], scalar);
        }
   }

    static void mul_and_add(double scalar, const double* __restrict__ v, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] += v[i] * scalar;
        }
    }

    // vec to vec
    static void mul(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] * v2[i];
        }
    }

    static void mul_and_add(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] += v1[i] * v2[i];
        }
    }

    static void div(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] / v2[i];
        }
    }

    static void sub(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v1[i] - v2[i];
        }
    }

    static void add(const double* __restrict__ v1, const double* __restrict__ v2, double* r, int rows, int columns) {
        int N = rows * columns;
        for (int i = 0; i < N; ++i) {
            r[i] = v1[i] + v2[i];
        }
    }

    static double sum(const double* __restrict__ v, int size) {
        double sum = 0.;
        for (int i = 0; i < size; ++i) {
            sum += v[i];
        }
        return sum;
    }

    // dot product
    static double dot(const double* __restrict__ v1, const double* __restrict__ v2, int size) {
        double r = 0.;
        for (int i = 0; i < size; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }

    /* rows - number of rows in the matrix 'm' and elements in the vector 'r'
     * columns - number of columns in the matrix 'm' and elements in the vector 'v'
     */
    static void dot_m_to_v(const double* __restrict__ m, const double* __restrict__ v, double* r, int rows, int columns) {
        for (int row = 0; row < rows; ++row) {
            int begin = row * columns;
            double sum = 0.;
            for (int col = 0; col < columns; ++col) {
                sum += m[begin + col] * v[col];
            }
            r[row] = sum;
        }
    }

    static void dot_v_to_m(const double* __restrict__ v, const double* __restrict__ m, double* r, int rows, int columns) {
        set(r, 0., columns);
        for (int row = 0; row < rows; ++row) {
            mul_and_add(v[row], &m[row*columns], r, columns);
        }
    }


    static void min_max(const double* __restrict__ m, int rows, int columns, int idx, double& min, double& max) {
        min = 1000000000.;
        max = -1000000000.;
        for (int i = 0; i < rows; ++i) {
            if (min > m[i*columns+idx])
                min = m[i*columns+idx];
            if (max < m[i*columns+idx])
                max = m[i*columns+idx];
        }
    }


    static double mean(const double* __restrict__ v, int size) {
        double sum = 0.;
        for (int i = 0; i < size; ++i) {
            sum += v[i];
        }
        return sum / size;
    }

    static double* clone(const double* __restrict__ v, int size) {
        double * new_v = new double[size];
        for (int i = 0; i < size; ++i) {
            new_v[i] = v[i];
        }
        return new_v;
    }

    static void copy(const double* __restrict__ s, double* __restrict__ d, int size) {
        for (int i = 0; i < size; ++i) {
            d[i] = s[i];
        }
    }

    static double* zeros(int size) {
        return alloc_and_set(size, 0.);
    }

    static double* alloc_and_set(int size, double val) {
        double* v = new double[size];
        for (int i = 0; i < size; ++i) {
            v[i] = val;
        }
        return v;
    }

    static void set(double* __restrict__ v, double scalar, int size) {
        for (int i = 0; i < size; ++i)
            v[i] = scalar;
    }


    static void split_array(const double* origX, const double* origY, int rows, int columns,
                            int feature_idx, double feature_val,
                            double** leftX, double** leftY, int& left_rows,
                            double** rightX, double** rightY, int& right_rows) {
        left_rows = 0;
        right_rows = 0;
        for (int i = 0; i < rows; ++i) {
            if (origX[i * columns + feature_idx] <= feature_val) {
                ++left_rows;
            }
            else {
                ++right_rows;
            }
        }

        // allocate
        *leftX = new double[left_rows * columns];
        *leftY = new double[left_rows];
        *rightX = new double[right_rows * columns];
        *rightY = new double[right_rows];

        //
        int left_idx = 0;
        int right_idx = 0;
        for (int i = 0; i < rows; ++i) {
            if (origX[i * columns + feature_idx] <= feature_val) {
                copy(&origX[i*columns], &(*leftX)[left_idx*columns], columns);
                (*leftY)[left_idx] = origY[i];
                ++left_idx;
            }
            else {
                copy(&origX[i*columns], &(*rightX)[right_idx*columns], columns);
                (*rightY)[right_idx] = origY[i];
                ++right_idx;
            }
        }
    }

};


//
// Optimization
//
struct optimize {
    typedef void (*FUNC)(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns);

    static void minimize_gc(double* __restrict__ theta, const double* X, int rows, int columns, const double* Y, FUNC func, int max_iterations) {
        double cost = 0.;
        double grad[columns];

        double e = 0.0001;
        double a = .4;

        func(theta, X, Y, &cost, grad, rows, columns);

        int cur_iter = 0;

        while (cost > e && cur_iter < max_iterations) {
            ++cur_iter;

            for (int i = 0; i < columns; ++i) {
                theta[i] = theta[i] - a * grad[i];
            }

            double new_cost;
            func(theta, X, Y, &new_cost, grad, rows, columns);

            if (cost < new_cost)
                a /= 2.;

            cost = new_cost;
        }
    }

    // for simple func: 1/M * 1/2 * SUM( (H - Y)**2 )
    static void quadratic_cost(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns) {
        double M = rows;

        memory::ptr<double> tmp( linalg::zeros(rows) );
        linalg::dot_m_to_v(X, theta, tmp.get(), rows, columns);

        memory::ptr<double> deltas( linalg::zeros(rows) );
        linalg::sub(tmp.get(), Y, deltas.get(), rows);

        linalg::pow(2., deltas.get(), tmp.get(), rows);
       *cost = (linalg::sum(tmp.get(), rows) / 2.) / M;

        linalg::dot_v_to_m(deltas.get(), X, grad, rows, columns);
        linalg::div(M, grad, grad, columns);
    }

    //
    // Logistic regression
    //
    static void sigmoid(const double* __restrict__ x, double* r, int size, bool correct_borders=true) {
        for (int i = 0; i < size; ++i) {
            r[i] = 1. / (1. + ::exp(-x[i]));
            if (correct_borders) {
                if (r[i] == 1.)
                    r[i] = .99999999;
                else if (r[i] == 0.)
                    r[i] = .00000001;
            }
        }
    }

    static void logistic_h(const double* __restrict__ theta, const double* __restrict__ X, double* r, int rows, int columns) {
        linalg::dot_m_to_v(X, theta, r, rows, columns);
        sigmoid(r, r, rows);
    }

    // for logistic cos func: sigmoid( h(X) )
    static void logistic_cost(const double* theta, const double* X, const double* Y, double* cost, double* grad, int rows, int columns) {
        double M = rows;

        memory::ptr<double> h = linalg::zeros(rows);
        logistic_h(theta, X, h.get(), rows, columns);

        double E = 0.;
        for (int i = 0; i < rows; ++i) {
            E += (-Y[i]) * ::log(h.get()[i]) - (1. - Y[i]) * ::log(1. - h.get()[i]);
        }
        E /= M;

        *cost = E;

        memory::ptr<double> deltas = linalg::zeros(rows);
        linalg::sub(h.get(), Y, deltas.get(), rows);
        linalg::dot_v_to_m(deltas.get(), X, grad, rows, columns);
    }
};      // optimize



//
// RF node val
//

struct node_val_base {
    virtual ~node_val_base(){};
    virtual double get_val(const double* v, int size) = 0;
    virtual int get_val_size() = 0;
};

struct node_val_mean : public node_val_base {
    double val_;

    node_val_mean(const double* X,
                  const double* Y,
                  int rows,
                  int columns) {
        val_ = linalg::mean(Y, rows);
    }
    virtual ~node_val_mean() {}

    virtual int get_val_size() { return 1; }

    virtual double get_val(const double*, int) {
        return val_;
    }
};

struct node_val_linear_regression : public node_val_base {

    int theta_len_;
    memory::ptr<double> theta_;

    node_val_linear_regression(const double* X,
                               const double* Y,
                               int rows,
                               int columns) {
        memory::ptr<double> tmp = linalg::zeros(rows*(columns+1));
        for (int i = 0; i < rows; ++i) {
            tmp.get()[i*(columns+1) + 0] = 1.;
            linalg::copy(&X[i*columns], &(tmp.get()[i*(columns+1) + 1]), columns);
        }

        theta_ = memory::ptr<double>(new double[columns+1]);
        for (int i = 0; i < columns+1; ++i)
            theta_.get()[i] = ((double)random::randint(0, 1000) / 1000.) - .5;
        optimize::minimize_gc(theta_.get(), tmp.get(), rows, columns + 1, Y, optimize::quadratic_cost, 100);

        theta_len_ = columns + 1;
    }

    virtual ~node_val_linear_regression() {}

    virtual int get_val_size() {
        return theta_len_;
    }

    virtual double get_val(const double* v, int size) {
        if (!v)
            return -1.;

        double tmp[size+1];
        tmp[0] = 1.;
        linalg::copy(v, &tmp[1], size);
        double p = linalg::dot(tmp, theta_.get(), size);
        return p;
    }
};

struct node_val_logistic_regression : public node_val_base {

    int theta_len_;
    memory::ptr<double> theta_;

    node_val_logistic_regression(const double* X,
                               const double* Y,
                               int rows,
                               int columns) {
        memory::ptr<double> tmp = linalg::zeros(rows*(columns+1));
        for (int i = 0; i < rows; ++i) {
            tmp.get()[i*(columns+1) + 0] = 1.;
            linalg::copy(&X[i*columns], &(tmp.get()[i*(columns+1) + 1]), columns);
        }

        theta_ = memory::ptr<double>(new double[columns+1]);
        for (int i = 0; i < columns+1; ++i)
            theta_.get()[i] = ((double)random::randint(0, 1000) / 1000.) - .5;
        optimize::minimize_gc(theta_.get(), tmp.get(), rows, columns + 1, Y, optimize::logistic_cost, 1000);

        theta_len_ = columns + 1;
    }

    virtual ~node_val_logistic_regression() {}

    virtual int get_val_size() {
        return theta_len_;
    }

    virtual double get_val(const double* v, int size) {
        if (!v)
            return -1.;

        double tmp[size+1];
        tmp[0] = 1.;
        linalg::copy(v, &tmp[1], size);
        double r;
        optimize::logistic_h(theta_.get(), tmp, &r, 1, size+1);

        return r;
    }
};





//
// RandomForest
//




double get_x_val(const double* x, int feature_index) {

    double val = 0.;

    if (feature_index < F_IDX) {
        // fixed part
        val = x[ feature_index ];
    }
    else {
        // sparse part
        const double* first = &x[F_IDX];
        const double* last = &x[F_IDX + SPARSE_PART_SIZE];
        const double* px = std::lower_bound(first, last, feature_index);
        if (px != last && *px == feature_index) {
            val = 1.;
        }
        else {
            val = 0.;
        }
    }

    return val;
}



//
// Fast dtree
//

class dtree_node {
    int K;
    int LNUM;


public:
    dtree_node(int kf, int ln) :
        K(kf),
        LNUM(ln)
    {}

private:
    struct data {
        data() :
            count(0),
            idx(-1),
            x_val(-1.),
            y_sum(0.),
            y_sum_squared(0.)
        {}

        int count;
        int idx;
        double x_val;
        double y_sum;
        double y_sum_squared;
    };

    struct split_data {
        vector<int> indices;
        vector<std::map<double, data> > accums;

        void clear() {
            indices.clear();
            accums.clear();
        }
    };


    // temporary data for splitting a node
    split_data sd_;

    double total_y_sum;
    double total_y_sum_squared;
    double total_count;

    int node_vector_idx;

public:
    void set_node_vector_idx(int idx) { node_vector_idx = idx; }
    int get_node_vector_idx() const { return node_vector_idx; }

    double get_mean() const { return total_y_sum / total_count; }
    int get_count() const { return total_count; }

    void start_splitting(int* indices) {
        sd_.clear();

        for (int i = 0; i < K; ++i) {
            sd_.indices.push_back(indices[i]);
            sd_.accums.push_back(std::map<double, data>());
        }

        total_y_sum = 0.;
        total_y_sum_squared = 0.;
        total_count = 0;
    }

    void process_splitting(const double* x, double y) {
        // go through all selected features and take its values
        for (int i = 0; i < K; ++i) {
            int feature_index = sd_.indices[i];

            // value of the current feature
            double val = get_x_val(x, feature_index);

            typename std::map<double, data>::iterator result = sd_.accums[i].find(val);
            if (sd_.accums[i].end() == result) {
                data d;
                d.idx = feature_index;
                d.x_val = val;
                d.count = 1;
                d.y_sum = y;
                d.y_sum_squared = y * y;
                sd_.accums[i].insert(std::pair<double, data>(val, d));
            }
            else {
                data& d = result->second;
                d.count += 1;
                d.y_sum += y;
                d.y_sum_squared += y * y;
            }

        }

        total_y_sum += y;
        total_y_sum_squared += y * y;
        total_count += 1;
    }

    void stop_splitting(int* idx, double* val, double* gain) {
        int best_idx = -1;
        double best_val = -1.;
        double best_gain = -1.;

        for (int i = 0; i < K; ++i) {
            // latest element contains sums for total set

            double mean = total_y_sum / total_count;
            double mean_squared = mean * mean;

            //
            double left_sum_accum = 0.;
            double right_sum_accum = 0.;
            double left_sum_squared_accum = 0.;
            double right_sum_squared_accum = 0.;
            int left_count_accum = 0;
            int right_count_accum = 0;

            // this will go from the smallest value to the biggest one
            for (typename std::map<double, data>::iterator it = sd_.accums[i].begin(); it != sd_.accums[i].end(); ++it) {
                data& d = it->second;

                left_sum_accum += d.y_sum;
                right_sum_accum = (total_y_sum - left_sum_accum);
                left_sum_squared_accum += d.y_sum_squared;
                right_sum_squared_accum = (total_y_sum_squared - left_sum_squared_accum);

                left_count_accum += d.count;
                right_count_accum = ((int)total_count - left_count_accum);

                double left_mean = left_sum_accum / left_count_accum;
                double right_mean = right_sum_accum / right_count_accum;

                double left_mean_squared = left_mean * left_mean;
                double right_mean_squared = right_mean * right_mean;

                double g = (total_y_sum_squared + mean_squared * total_count - 2. * mean * total_y_sum) -
                           (left_sum_squared_accum + left_mean_squared * left_count_accum - 2. * left_mean * left_sum_accum) -
                           (right_sum_squared_accum + right_mean_squared * right_count_accum - 2. * right_mean * right_sum_accum);
                if (!isnan(g) && g > 0. && (/*best_idx == -1 ||*/ g > best_gain)) {
                    if (LNUM <= left_count_accum && LNUM <= right_count_accum) {
                        best_idx = d.idx;
                        best_val = d.x_val;
                        best_gain = g;

//LOG((total_y_sum_squared + mean_squared * total_count - 2. * mean * total_y_sum))
//LOG((left_sum_squared_accum + left_mean_squared * left_count_accum - 2. * left_mean * left_sum_accum))
//LOG((right_sum_squared_accum + right_mean_squared * right_count_accum - 2. * right_mean * right_sum_accum))
//LOG(best_idx << "; " << best_val << "; " << best_gain)
                    }
                }
            }
        }

        *idx = best_idx;
        *val = best_val;
        *gain = best_gain;

        sd_.clear();
    }

};


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
    };

protected:
    // vector of length VEC_LEN: leaf [node_id, class, data, ...] or intermediate node [node_id, class, idx, val, left, right]
    vector<double> tree_;

public:
    dtree() : tree_() {}
    virtual ~dtree() {}


    void tofile(const char* fname) {
        FILE* fd = fopen(fname, "wb+");
        for (int i = 0; i < tree_.size(); ++i) {
            fwrite(&tree_[i], sizeof(double), 1, fd);
        }

        fclose(fd);
    }

    void fromfile(const char* fname) {
        tree_.clear();

        FILE* fd = fopen(fname, "rb");
        while(!feof(fd)) {
            double d;
            fread(&d, sizeof(double), 1, fd);
            tree_.push_back(d);
        }
        fclose(fd);
    }

    void print() {
        int num = tree_.size() / dtree::VEC_LEN;

        //cout.precision(15);

        cout << "// DTree array" << endl;
        cout << "int dtree_array_size = " << tree_.size() << ";" << endl;
        cout << "double dtree_array[] = {" << endl;
        for (int i = 0; i < num; ++i) {
            cout //<< std::fixed
                 << tree_[i * dtree::VEC_LEN + 0] << ","
                 << tree_[i * dtree::VEC_LEN + 1] << ","
                 << tree_[i * dtree::VEC_LEN + 2] << ","
                 << tree_[i * dtree::VEC_LEN + 3] << ","
                 << tree_[i * dtree::VEC_LEN + 4] << ","
                 << tree_[i * dtree::VEC_LEN + 5] << "," << endl;
        }
        cout << "};" << endl;
        cout << "// END of DTree array" << endl;
    }

    double predict(const double* x) {
        if (!tree_.size())
            return 0.;

        double val = 0.;
        int start_idx = 0;

        while (true) {
            double* node = &tree_[start_idx];

            if (dtree::LEAF == node[CLS_IDX]) {
                val = node[DATA_IDX];
                break;
            }
            else {
                double val = get_x_val(x, (int)node[IDX_IDX]);

                if (val <= node[VAL_IDX]) {
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


class dtree_learner : public dtree {
    map<int, dtree_node> nodes_;

    int COLUMN_NUMBER;
    int K;
    int LNUM;


    int add_node(map<int, dtree_node>& nodes) {
        int nodes_num = tree_.size() / dtree::VEC_LEN;
        int new_id = nodes_num;

        tree_.push_back(new_id);
        tree_.push_back(dtree::SPLITTING);
        tree_.push_back(-1);    // idx
        tree_.push_back(-1);    // val
        tree_.push_back(-1);    // left
        tree_.push_back(-1);    // right

        // and node for this new splitting node
        int indices[K];
        random::get_k_of_n(K, COLUMN_NUMBER, &indices[0]);

        dtree_node root(K, LNUM);
        root.set_node_vector_idx(new_id);
        root.start_splitting(indices);

        nodes.insert(pair<int, dtree_node>(new_id, root));

        return new_id;
    }

    bool get_node(const double* x, dtree_node** node) {
        int nodes_num = tree_.size() / VEC_LEN;
        if (!nodes_num) {
            add_node(nodes_);
        }

        bool result = true;
        int start_idx = 0;

        while (true) {
            double *vec = &tree_[start_idx];

            if (dtree::SPLITTING == vec[CLS_IDX]) {
                // found, get the object
                map<int, dtree_node>::iterator it = nodes_.find(vec[ID_IDX]);
                *node = &it->second;
                break;
            }
            else if (dtree::LEAF == vec[CLS_IDX]) {
                // nothing to return for learning
                result = false;
                break;
            }
            else {
                // go down
                int idx = (int)vec[IDX_IDX];
                double val = vec[VAL_IDX];
                double x_val = get_x_val(x, idx);

                if (x_val <= val) {
                    start_idx = (int)vec[LEFT_IDX] * dtree::VEC_LEN;
                }
                else {
                    start_idx = (int)vec[RIGHT_IDX] * dtree::VEC_LEN;
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

    void process_fit(const double* x, double y) {
        dtree_node* node = NULL;
        if (get_node(x, &node)) {
            node->process_splitting(x, y);
        }
    }

    void stop_splitting() {
        map<int, dtree_node> new_nodes;

        for (map<int, dtree_node>::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
            int idx;
            double val;
            double gain;

            dtree_node& node = it->second;

            node.stop_splitting(&idx, &val, &gain);
            if (-1 == idx) {
                // cannot split, make a leaf
                int id = it->first;
                int start_idx = id * dtree::VEC_LEN;
                double* vec = &tree_[start_idx];

                vec[CLS_IDX] = dtree::LEAF;
                vec[DATA_IDX] = node.get_mean();
                vec[5] = node.get_count();          // for testing only
            }
            else {
                // make an intermediate node
                int id = it->first;
                int start_idx = id * dtree::VEC_LEN;
                double* vec = &tree_[start_idx];

                vec[CLS_IDX] = dtree::NON_LEAF;
                vec[IDX_IDX] = idx;
                vec[VAL_IDX] = val;

                int new_left_id = add_node(new_nodes);
                int new_right_id = add_node(new_nodes);

                // it's possible having invalidated iterators here, so I go via tree_ directly
                tree_[start_idx + LEFT_IDX] = new_left_id;
                tree_[start_idx + RIGHT_IDX] = new_right_id;
            }
        }

        nodes_.swap(new_nodes);
    }

    bool need_splitting() {
        return nodes_.size();
    }
};







/////////////////////////////////////////////////////////////////////////
// Testing
/////////////////////////////////////////////////////////////////////////



void test1() {

    cout << "TEST 1" << endl;

    int N = 10;
    int C = 25;

    double X[] = {
        14,10,31,23,2,8,4172,8644,12520,20406,21328,21633,1801941,3099127,9445805,9445815,9445819,9446101,9448453,9448463,9448530,9448903,9448939,9448986,9449183,
        14,10,31,23,2,7,2508,10751,12502,17621,21552,21637,1801941,8959319,9443055,9445815,9445819,9446489,9448453,9448463,9448597,9448904,9448918,9448973,9449191,
        14,10,31,23,-1,2,7,2508,10751,12502,21032,21356,21637,1801941,5676658,9444253,9445815,9445819,9448453,9448463,9448640,9448904,9448933,9449059,9449191,
        14,10,31,23,-1,2,7,12090,12520,20406,21328,21633,1801941,4432375,9439459,9445815,9445819,9447157,9448453,9448463,9448694,9448901,9448947,9449038,9449175,
        14,10,31,23,2,7,2508,10751,12502,17265,21552,21637,1801941,6174798,9442553,9445815,9445819,9446682,9448453,9448463,9448640,9448904,9448933,9449059,9449191,
        14,10,31,23,2,7,2755,8417,12500,20406,21328,21633,1801941,7672303,9441850,9445815,9445819,9446095,9448452,9448459,9448533,9448903,9448939,9448973,9449183,
        14,10,31,23,-1,2,7,12090,12520,20406,21328,21633,1801941,3547319,9444512,9445815,9445819,9447033,9448453,9448463,9448694,9448901,9448947,9449038,9449175,
        14,10,31,23,2,7,1762,8417,12500,20406,21328,21633,1801941,7755280,9444294,9445815,9445819,9446095,9448452,9448459,9448533,9448903,9448939,9448973,9449183,
        14,10,31,23,-1,-1,2,7,2508,10751,12502,17426,21452,21637,1801941,5415650,9437724,9445815,9445821,9448453,9448463,9448902,9448923,9449116,9449195,
        14,10,31,23,-1,2,7,12090,12520,20406,21328,21633,1801941,3193482,9442752,9445815,9445819,9447150,9448453,9448463,9448694,9448901,9448947,9449038,9449175,
    };
    double Y[] = {0., 1., 1., 1., 1., 1., 1., 1., 1., 1.};

    dtree_learner t(TOTAL_FEATURES, 3500, 1);
    t.start_fit();

    while (true) {
        for (int i = 0; i < N; ++i) {
            t.process_fit(&X[i*C], Y[i]);
        }
        t.stop_splitting();
        if (!t.need_splitting()) {
            break;
        }
    }

    double* v = &X[0*C];
    double p = t.predict(v);
    LOG("predict(0) == " << p)

    v = &X[1*C];
    p = t.predict(v);
    LOG("predict(1) == " << p)

    t.print();

    t.tofile("c:\\Temp\\rf.b");
}


int main() {

    test1();

    return 0;
}

//
// C stype interface for Python
//

extern "C" {

    void predict(void* ptree, double* x, double* p) {
        *p = static_cast<dtree*>(ptree)->predict(x);
    }

    void* alloc_tree_learner(int columns, int k, int lnum) {
        return new dtree_learner(TOTAL_FEATURES, k, lnum);
    }

    void* alloc_tree() {
        return new dtree();
    }

    void free_tree(void* ptree) {
        delete (dtree*)ptree;
    }

    void fromfile_tree(void* ptree, const char* fname) {
        static_cast<dtree*>(ptree)->fromfile(fname);
    }

    void tofile_tree(void* ptree, const char* fname) {
        static_cast<dtree*>(ptree)->tofile(fname);
    }

    void print_tree(void* ptree) {
        static_cast<dtree*>(ptree)->print();
    }

    void start_fit_tree(void* ptree) {
        static_cast<dtree_learner*>(ptree)->start_fit();
    }

    void fit_tree(void* ptree, double* x, double y) {
        static_cast<dtree_learner*>(ptree)->process_fit(x, y);
    }

    void stop_fit_tree(void* ptree) {
        static_cast<dtree_learner*>(ptree)->stop_splitting();
    }

    int end_of_splitting(void* ptree) {
        return (int)(false == static_cast<dtree_learner*>(ptree)->need_splitting());
    }

};   // END of extern "C"









