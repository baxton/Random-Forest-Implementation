


#include <cstdio>
#include <iostream>


// v1, v2, v3,  v1**2, v2**2, v3**2,  v1**3, v2**3, v3**3
const int COLUMNS = 9;



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



void sigmoid(double x) {
    x = x < 20 ? (x > -20 ? x : -20) : 20;
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

    double p1 = h > 0 ? h : 0.0000000000001;
    double p2 = (1. - h) > 0 ? (1. - h) : 0.0000000000001;

    // calc cost part
    double cost = -y * ::log(p1) - (1. - y) * ::log(p2);

    // calc gradient part
    double delta = h - y;
    mul_and_add(delta, x, grad_x, columns);
}




///////////////////////////////////////////////////////////////////////
// read files stuff
///////////////////////////////////////////////////////////////////////

const int ALLIGN = 512;

char buffer[]

// problem specific - it knows about files
double cost(const double* theta, double* grad, int columns) {
}

///////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////

static void minimize_gc(double* theta, const double* x, int columns, const double* y, FUNC func, int max_iterations) {
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




int main() {
    return 0;
}
