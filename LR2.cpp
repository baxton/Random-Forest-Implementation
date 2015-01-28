

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <algorithm>

using namespace std;

#define QUICK

typedef unsigned long long ULONG;       // I have to care about sizes as I'm on 32 bit system

// vector format
// click,  v1, v2, v3, v4, v5, v6, v7
//
const int VEC_LEN = 8;
const int VEC_LEN_BYTES = VEC_LEN * sizeof(float);

const int COLUMNS = VEC_LEN;    // including zero term
//int g_columns = VEC_LEN + 7 * 3; //COLUMNS;
int g_columns = COLUMNS;

const double l = 0.;


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
            return ::rand();
#else
            return (int)::rand() * ((int)RAND_MAX + 1) + (int)::rand();
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


    // returns random numbers from [0..1]
    static void rand(double* numbers, int size) {
        double max = -1.;
        for (int i = 0; i < size; ++i) {
            double r = (double)randint();
            numbers[i] = r;
            if (max < r)
                max = r;
        }
        if (max > 0.) { // sanity check
            for (int i = 0; i < size; ++i) {
                numbers[i] /= max;
            }
        }
    }

};



///////////////////////////////////////////////////////////////////////
// LinAlg
///////////////////////////////////////////////////////////////////////

double dot(const double* v1, const double* v2, int size) {
    double r = 0.;
    for (int i = 0; i < size; ++i) {
        r += v1[i] * v2[i];
    }
    return r;
}

void mul_and_add(double scalar, const double* v, double* r, int size) {
    for (int i = 0; i < size; ++i) {
        r[i] += v[i] * scalar;
    }
}


///////////////////////////////////////////////////////////////////////
// LogReg
///////////////////////////////////////////////////////////////////////

double sigmoid(double x) {
    x = x <= 50 ? (x >= -50 ? x : -50) : 50;
    return 1. / (1. + ::exp(-x));
}

double logistic_h(const double* theta, const double* x, int columns) {
    double r = dot(x, theta, columns);
    return sigmoid(r);
}




// for logistic cos func: sigmoid( h(X) )
// processes 1 sample per call
double logistic_cost(const double* theta, const double* x, double* grad_x, double y, int columns) {
    // calc logistic val
    double h = logistic_h(theta, x, columns);

    // calc cost part
    double p1 = h > 0. ? h : 0.0000000001;
    double p2 = (1. - h) > 0. ? (1. - h) : 0.0000000001;
    double cost = -1 * (y * ::log(p1) + (1. - y) * ::log(p2));

    // calc gradient part
    if (grad_x) {
        double delta = h - y;
        mul_and_add(delta, x, grad_x, columns);
    }

    return cost;
}




///////////////////////////////////////////////////////////////////////
// read files stuff
///////////////////////////////////////////////////////////////////////

const ULONG ALLIGN = 512;
const int BUFFER_LEN_BYTES = 120000000;
char buffer[ALLIGN + BUFFER_LEN_BYTES];
char* p = buffer;

const int CV_SIZE = 8000000;
const int CV_NUM = CV_SIZE / 4;

#if defined QUICK
    const int CROSS_VAL_NUM = CV_NUM;
#else
    const int CROSS_VAL_NUM = CV_NUM;
#endif
int cross_val_ids[CROSS_VAL_NUM];
double cross_val_E = 0.;
double cross_val_SME = 0.;
int cur_cross_idx = 0;

const int N = 100;

#if defined QUICK
    const int FILES_NUM = 2;
#else
    const int FILES_NUM = 14;
#endif

FILE* fds[FILES_NUM];
const char* file_names[/*FILES_NUM*/] = {
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.0",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.1",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.2",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.3",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.4",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.5",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.6",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.7",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.8",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.9",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.10",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.11",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.12",
"C:\\Temp\\kaggle\\CTR\\data\\lr_train\\lr_train.b.13",
};


//
// problem specific - it knows about files and vector's format
//
double read_file(FILE* fd, const double* theta, double* grad, int columns, int* M, int* cur_id) {
    double E = 0.;

    fseek(fd, 0, SEEK_END);
    size_t size = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    size_t ret = fread(p, size, 1, fd);

    float* vectors = (float*)p;
    int vectors_num = size / VEC_LEN_BYTES;

    //double x[VEC_LEN];
    double x[VEC_LEN + 7 * 3];

    //g_columns = VEC_LEN + 7 * 3;

    for (int v = 0; v < vectors_num; ++v) {
        int idx = v * VEC_LEN;

        // convert to double from float
        for (int i = 0; i < VEC_LEN; ++i) {
            x[i] = (double)vectors[idx + i];
            x[i] = x[i] > 0 ? log(x[i]) : 0;
        }

//        x[3] = x[4] ? x[3] / x[4] : .0;
//        x[4] = x[5] ? x[4] / x[5] : .0;
//        x[5] = x[6] ? x[5] / x[6] : .0;
//        x[6] = x[7] ? x[6] / x[7] : .0;

/*
        for (int i = 0; i < VEC_LEN-1; ++i) {
            int j = i + 1;
            x[VEC_LEN + i*3+0] = x[j] * x[j];
            x[VEC_LEN + i*3+1] = x[j] * x[j] * x[j];
            x[VEC_LEN + i*3+2] = x[j] * x[j] * x[j] * x[j];

            //x[j] = sqrt(x[j]);
        }
*/
        double y = x[0];
        x[0] = 1.;  // zero term

        if (*cur_id == cross_val_ids[cur_cross_idx]) {
//cout << "# CV id: " << cross_val_ids[cur_cross_idx] << endl;
            ++cur_cross_idx;
            cross_val_E += logistic_cost(theta, x, NULL, y, columns);
        }
        else {
            E += logistic_cost(theta, x, grad, y, columns);
        }

        // set next cur ID
        *cur_id += 1;
    }

    *M = vectors_num;

    //usleep(1);

    return E;
}


//
// calculates cost and fills in gradient in one go
//
double cost(const double* theta, double* grad, int columns, int* M) {
    double E = 0.;

    for (int i = 0; i < columns; ++i) {
        grad[i] = 0.;
    }

    *M = 0;



    // start checking for ID is for cross varsalidation from the begining
    cur_cross_idx = 0;
    cross_val_E = 0.;

    // start counting current varsector ID
    int cur_id = 0;

    // go throuh the entire train set
    for (int f = 0; f < FILES_NUM; ++f) {
        int tmp;
        E += read_file(fds[f], theta, grad, columns, &tmp, &cur_id);
        *M += tmp;
    }

    // calc grad and apply regularization
    double reg = 0.;
    grad[0] /= *M;
    for (int i = 1; i < columns; ++i) {
        grad[i] = (grad[i] + l * theta[i]) / *M;
        reg += theta[i] * theta[i];
    }
    reg *= l/(*M * 2.);

    return E / *M + reg;
}



///////////////////////////////////////////////////////////////////////
// utils
///////////////////////////////////////////////////////////////////////

void print_theta(const double* theta, int size, double cost) {
    cout << "// Theta for cost " << cost << endl;
    cout << "double theta[" << size << "] = {";
    for (int i = 0; i < size; ++i) {
        cout << std::setw(19) << std::setprecision(16) << theta[i] << ", ";
    }
    cout << "};" << endl;
}

///////////////////////////////////////////////////////////////////////
// gradient discent
///////////////////////////////////////////////////////////////////////

typedef double (*FUNC)(const double*, double*, int, int*);

static void minimize_gc(double* theta, int columns, FUNC func, int max_iterations) {
    double cost = 0.;
    double grad[columns];

    double e = 0.0001;
    double a = 22.;

    int M = 0;

    double local_theta[columns];

    cost = func(theta, grad, columns, &M);

    int cur_iter = 0;

    while (cost > e && cur_iter < max_iterations) {
        ++cur_iter;

        // save theta
        for (int i = 0; i < columns; ++i) {
            local_theta[i] = theta[i];
        }

        // update theta
        for (int i = 0; i < columns; ++i) {
            theta[i] = theta[i] - a * grad[i];
        }

        double new_cost = func(theta, grad, columns, &M);

        if (cost <= new_cost) {
            a -= a / 5.;
            //a /= 2.;
            // restore theta
            for (int i = 0; i < columns; ++i) {
                theta[i] = local_theta[i];
            }
            // reset current iteration
            --cur_iter;
        }
        else {
            cost = new_cost;

//          cout << "# theta: " << theta[0] << ", " << theta[1] << ", " << theta[2] << "..." << endl;
            cout << "# alpha " << a << " grad: " << grad[0] << ", " << grad[1] << ", " << grad[2] << "..." << endl;
            cout << "# CV " << (cross_val_E / M) << endl;   // intermediate val
            print_theta(theta, columns, cost);
            cout << "# iteration " << std::setw(19) << std::setprecision(16) << cur_iter << " cost " << cost << endl;

            if (cur_cross_idx < 10000)
                cout << "# ERROR CV idx " << cur_cross_idx << endl;
        }
    }

    cout << "# num of iterations " << cur_iter << "; final cost " << cost << endl;
}




int main() {

    // allign buffer
    while (0 != ((ULONG)p % ALLIGN)) {
        ++p;
    }

    // open data files
    for (int f = 0; f < FILES_NUM; ++f) {
        fds[f] = fopen(file_names[f], "rb");
        if (!fds[f]) {
            cout << "# ERROR cannot open data file " << f << ": " << errno << endl;
            return 0;
        }
    }

    random::seed();

    // prepare CV ids
#if defined QUICK
    const int LINES_IN_TRAIN = 6000000;
#else
    const int LINES_IN_TRAIN = 40428967;
#endif

    //random::get_k_of_n(CROSS_VAL_NUM, LINES_IN_TRAIN, &cross_val_ids[0]);
    //std::sort(&cross_val_ids[0], &cross_val_ids[CROSS_VAL_NUM]);
    FILE* fcv = fopen("C:\\Temp\\kaggle\\CTR\\cv_click_ids.b", "rb");
    fread(&cross_val_ids[0], CV_SIZE, 1, fcv);
    fclose(fcv);

    //

    //double theta[g_columns];
    //random::rand(theta, g_columns);
    //
    double theta[29] = { -3.939460551955113,   4.562200696553635,  -3.664489109755185,   4.817769544714159,
                         4.723329792925568,  -4.033096586970479,   5.062432897824738,   3.497314841928399,
                         -5.615707359460818,   4.497059200803301,    1.09288393935726, -0.9097795785900078,
                         1.039575005126297,   1.536249626306091,  -1.726144771705812,  0.4279856438712299,
                         1.047898864907344,   -1.73849936354751,  0.4262981091067785,    1.04993014379362,
                         -0.9129812686187934,   1.034680164095185,   1.530510194297934,  -1.749494984907827,
                         0.4153619858817644,   1.040346151469149,  -1.936188631239613,  0.4751589971718277,   1.129217198720281, };
    cout << "# init theta: " << theta[0] << ", " << theta[1] << ", " << theta[2] << "..." << endl;


    minimize_gc(theta, g_columns, cost, N);

    // print the result
    cout << "// Theta for LogReg, CV " << cross_val_E << endl;
    cout << "double theta[" << g_columns << "] = {";
    for (int i = 0; i < g_columns; ++i) {
        cout << std::setw(19) << std::setprecision(16) << theta[i] << ", ";
    }
    cout << "};" << endl;

    return 0;
}


