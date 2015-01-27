

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <unistd.h>

using namespace std;

typedef unsigned long long ULONG;       // I have to care about sizes as I'm on 32 bit system

// vector format
// click,  v1, v1**2, v1**3,  v2, v2**2, v2**3,  v3, v3**2, v3**3
//
const int VEC_LEN = 10;
const int VEC_LEN_BYTES = VEC_LEN * sizeof(float);

const int COLUMNS = VEC_LEN;    // including zero term

const double l = .5;


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

    double l = 1.;

    double r = 1. / (1. + ::exp(-x * l));
    if (r == 1.)
        r = .999999999;
    else if (r == 0.)
        r = .000000001;

    return r;
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
    double cost = y * ::log(p1) + (1. - y) * ::log(p2);

    // calc gradient part
    double delta = h - y;
    mul_and_add(delta, x, grad_x, columns);

    return cost;
}




///////////////////////////////////////////////////////////////////////
// read files stuff
///////////////////////////////////////////////////////////////////////

const ULONG ALLIGN = 512;
const int BUFFER_LEN_BYTES = 120000000;
char buffer[ALLIGN + BUFFER_LEN_BYTES];
char* p = buffer;

const int N = 200;

const int FILES_NUM = 14;
//const int FILES_NUM = 1;
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
double read_file(FILE* fd, const double* theta, double* grad, int columns, int* M) {
    double E = 0.;

    fseek(fd, 0, SEEK_END);
    size_t size = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    size_t ret = fread(p, size, 1, fd);

    float* vectors = (float*)p;
    int vectors_num = size / VEC_LEN_BYTES;

    double x[VEC_LEN];

    for (int v = 0; v < vectors_num; ++v) {
        int idx = v * VEC_LEN;

        // convert to double from float
        for (int i = 0; i < VEC_LEN; ++i)
            x[i] = (double)vectors[idx + i];

        x[1] = sqrt(x[1]);
        x[4] = sqrt(x[4]);
        x[7] = sqrt(x[7]);
        //x[4] = ;
        //x[5] = ;
        //x[6] = ;

        double y = x[0];
        x[0] = 1.;  // zero term

        E += logistic_cost(theta, x, grad, y, columns);
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

    // go throuh the entire train set
    for (int f = 0; f < FILES_NUM; ++f) {
        int tmp;
        E += read_file(fds[f], theta, grad, columns, &tmp);
        *M += tmp;
    }

    double reg = 0.;

    for (int i = 0; i < columns; ++i) {
        grad[i] /= *M;

        if (i > 0)
            reg += theta[i] * theta[i];
    }

    reg *= l/(*M * 2.);

    return -1 * E / *M + reg;
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
        theta[0] = theta[0] - a * grad[0];
        for (int i = 1; i < columns; ++i) {
            theta[i] = theta[i] - a * (grad[i] + l * theta[i] / M);
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
            print_theta(theta, columns, cost);
            cout << "# iteration " << std::setw(19) << std::setprecision(16) << cur_iter << " cost " << cost << endl;
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

    int columns = COLUMNS;
    //int columns = 4;

    //double theta[columns];
    //random::rand(theta, columns);
    // Theta for cost 0.3819295622215437
    double theta[10] = { -6.621295529420721,   4.174826949593883,  -5.397414750887257,   4.107372047196714,   4.554937421642614,   -5.76491522658397,   4.391173394717607,   4.450525199056153,  -5.737414862957012,   4.378752054066173, };
    cout << "# init theta: " << theta[0] << ", " << theta[1] << ", " << theta[2] << "..." << endl;


    minimize_gc(theta, columns, cost, N);

    // print the result
    cout << "// Theta for LogReg" << endl;
    cout << "double theta[" << columns << "] = {";
    for (int i = 0; i < columns; ++i) {
        cout << std::setw(19) << std::setprecision(16) << theta[i] << ", ";
    }
    cout << "};" << endl;

    return 0;
}


