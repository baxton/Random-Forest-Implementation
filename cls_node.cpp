

//
// g++ cls_node.cpp -o cls_node.exe
//

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <vector>
#include <limits>
#include <set>
#include <ctime>

using namespace std;

#define X_LEN        10000
#define VEC_LEN      (X_LEN + 1 + 1)
#define DTW_VEC_LEN  (X_LEN + 1)

#define ALIGNMENT __attribute__((aligned(4)))

#define DIST_IDX 0
#define CLS_IDX  1
#define X_IDX    2



float DTW [DTW_VEC_LEN * DTW_VEC_LEN] ALIGNMENT;





//
// Lin algebra
//
struct linalg {

    static void sub(const float* __restrict__ v1, const float* __restrict__ v2, float* r, int size) {
        for (int i = 0; i < size; ++i) {
            r[i] = v1[i] - v2[i];
        }
    }

    // dot product
    static float dot(const float* __restrict__ v1, const float* __restrict__ v2, int size) {
        float r = 0.;
        for (int i = 0; i < size; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }

    static float norm(const float* __restrict__ v, int size) {
        float d = dot(v, v, size);
        return sqrt(d);
    }



    // distances

    static float cosine(const float* __restrict__ v1, const float* __restrict__ v2, int size) {
        // as per scipy
        float d = dot(v1, v2, size);
        float n1 = norm(v1, size);
        float n2 = norm(v2, size);

        return 1.f - d / (n1 * n2);
    }

}; // linalg




template<class T>
void line2vec(const string& line, vector<T>& vec) {
    char sep = ',';

    vec.clear();

    size_t old_pos = 0;
    size_t pos = line.find(sep, old_pos);

    int idx = 0;

    while (std::string::npos != pos) {
        string ss = line.substr(old_pos, pos - old_pos);

        if ("." == ss) {
            vec.push_back(0);
        }
        else {
            vec.push_back( atof(ss.c_str()) );
        }

        old_pos = pos + 1;
        pos = line.find(sep, old_pos);

        ++idx;
    }

    if (old_pos != std::string::npos && old_pos < line.length()) {
        string ss = line.substr(old_pos);

        if ("." == ss) {
            vec.push_back(0);
        }
        else {
            vec.push_back( atof(ss.c_str()) );
        }
    }
}




void help();
bool read_data(vector<float>& data, const string& fname);
float predict(const vector<float>& train, vector<float>& vec, int w);

bool equal(float f1, float f2, float e=0.01);

float DTWDistance(const float* s1, const float* s2, int w);
void fill(float v);





int main(int argc, const char** argv) {

    if (2 != argc) {
        help();
        return -1;
    }

    string fname = argv[1];
    cout << "#train data file " << fname << endl;


    vector<float> train;
    train.reserve(1024 * 1024 * sizeof(float));

    if (!read_data(train, fname)) {
        help();
        return -1;
    }

    string name;

    string line;
    while (std::getline(cin, line)) {

        if ('#' == line[0]) {
            // comment
            cout << line << endl;
        }
        else if ('+' == line[0]) {
            // name of file
            name = line.substr(1);
        }
        else {
            // data of a sensor
            // 1st send it further for processing
            //cout << name << endl;
            //cout << line << endl;

            // and process it here too
            vector<float> tmp;
            line2vec(line, tmp);

            time_t start = clock();
            float p = predict(train, tmp, 3);

            //vector<float> trr(VEC_LEN, 0.0);
            //vector<float> trr2(VEC_LEN, 1000.0);
            //float p = DTWDistance(&trr[X_IDX], &tmp[X_IDX], 3);
            //float p2 = DTWDistance(&trr2[X_IDX], &tmp[X_IDX], 3);

            time_t finish = clock();
            cout << "#time " << ((float)(finish - start) / CLOCKS_PER_SEC) << endl;

            cout << "#" << name << " : " << p << endl;
        }

    }


    return 0;
}

float DTWDistance(const float* s1, const float* s2, int w) {

    fill(numeric_limits<float>::infinity());

    DTW[0] = 0.0f;

    for (int i = 0; i < X_LEN; ++i) {
        int begin = (i - w) > 0 ? i - w : 0;
        int end = (i + w) < X_LEN ? i + w : X_LEN;

        for (int j = begin; j < end; ++j) {

            int dtwI = i + 1;
            int dtwJ = j + 1;

            float dist = (s1[i] - s2[j]) * (s1[i] - s2[j]);

            float min = DTW[(dtwI-1) * DTW_VEC_LEN + dtwJ] < DTW[dtwI * DTW_VEC_LEN + (dtwJ-1)] ? DTW[(dtwI-1) * DTW_VEC_LEN + dtwJ] : DTW[dtwI * DTW_VEC_LEN + (dtwJ-1)];
            min = min < DTW[(dtwI-1) * DTW_VEC_LEN + (dtwJ-1)] ? min : DTW[(dtwI-1) * DTW_VEC_LEN + (dtwJ-1)];

            DTW[dtwI * DTW_VEC_LEN + dtwJ] = dist + min;

        }
    }

    return sqrt( DTW[DTW_VEC_LEN*DTW_VEC_LEN - 1] );
}

void fill(float v) {

    int size_1 = (DTW_VEC_LEN-1)*(DTW_VEC_LEN-1);
    int size_2 = DTW_VEC_LEN * DTW_VEC_LEN;

    int i = 0;

    for (; i < size_1; i += 8) {
        DTW[i+0] = v;
        DTW[i+1] = v;
        DTW[i+2] = v;
        DTW[i+3] = v;
        DTW[i+4] = v;
        DTW[i+5] = v;
        DTW[i+6] = v;
        DTW[i+7] = v;
    }

    for (; i < size_2; ++i) {
        DTW[i] = v;
    }
}


bool equal(float f1, float f2, float e) {
    float d = f1 - f2;
    return -e <= d && d <= e;
}

void help() {
    cout << "cls_node - one node of classification\n"
         << "Usage: cat data.txt | cls_node train_data1.bin | cls_node train_data2.bin > result.txt"
         << endl;
}

bool read_data(vector<float>& data, const string& fname) {
    try {
        ifstream fin;
        fin.open(fname.c_str(), ifstream::in | ifstream::binary);

        vector<float> tmp;
        double d;
        while (fin.read((char*)&d, sizeof(double))) {
            tmp.push_back((float)d);
        }

        data.swap(tmp);
    }
    catch(const std::exception& ex) {
        cout << "#ERROR: [" << ex.what() << "] from " << fname << endl;
        return false;
    }
    catch(...) {
        cout << "#ERROR: unknown exception on reading data from " << fname << endl;
        return false;
    }

    return true;
}

struct distance {
    float dist;
    int id;
};

bool operator< (const distance& cd1, const distance& cd2) {
    return cd1.id < cd2.id;
}


float predict(const vector<float>& train, vector<float>& vec, int w) {

    int N = train.size() / VEC_LEN;
    cout << "# number of train rows: " << N << endl;

    float* initial_vec = &train[0];


    float dist_to_1 = DTWDistance(&initial_vec[X_IDX], &vec[X_IDX], w);
    float delta = 50.;
    float LOW = dist_to_1 - delta;
    float HIGH = dist_to_1 + delta;

    set<distance> visited;


    float min_dist = numeric_limits<float>::infinity();
    int min_idx = -1;

    while (min_dist == numeric_limits<float>::infinity() && LOW > 0.) {

        for (int i = 1; i < N; ++i) {
            set<distance>::iterator it = visited.find(i);
            if (visited.end() == it) {
            }
            else {
            }

                float dist = DTWDistance(&train[i * VEC_LEN + X_IDX], &vec[X_IDX + 0], w);
        }
    }




    for (int i = 1; i < N; ++i) {

        float dist = DTWDistance(&train[i * VEC_LEN + X_IDX], &vec[X_IDX + 0], w);

        if (min_dist > dist) {
            min_dist = dist;
            min_idx = i;

            //cout << "# >>" << min_dist << "; cls " << train[min_idx * VEC_LEN + CLS_IDX] << " vs " << vec[CLS_IDX] << endl;
        }

        //if (0 == (i % 100)) {
        //    cout << "# processed " << i << " rows" << endl;
        //}
    }

    float p = train[min_idx * VEC_LEN + CLS_IDX];

    return p;
}





