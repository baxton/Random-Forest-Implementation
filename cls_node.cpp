

//
// g++ cls_node.cpp -o cls_node.exe
// g++ cls_node.cpp -o cls_node_p.exe
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
#include <algorithm>


using namespace std;

#define X_LEN        10000
#define VEC_LEN      (X_LEN + 1 + 2)
#define DTW_VEC_LEN  (X_LEN + 1)

#define ALIGNMENT __attribute__((aligned(4)))

#define CLS_IDX   0
#define DIST1_IDX 1
#define DIST2_IDX 2
#define X_IDX     3


// for people
#define DELTA       500.
#define MAX_RADIUS  500000.


// Model
const int files_num = 14;
const char* fnames[files_num] = {
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_0.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_1.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_2.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_3.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_4.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_5.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_6.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_7.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_8.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_9.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_10.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_11.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_12.b",
    "C:\\Temp\\kaggle\\epilepsy\\data\\prepared_sub1\\full_train_13.b",
    };



// Pre-allocated arrays
float DTW [DTW_VEC_LEN * DTW_VEC_LEN] ALIGNMENT;

float dot_1[X_LEN] ALIGNMENT;
float dot_2[X_LEN] ALIGNMENT;


// DTW window width
int W = 50;



//
// Lin algebra
//
struct linalg {

    static void set(float* __restrict__ v, float scalar, int size) {
        for (int i = 0; i < size; ++i)
            v[i] = scalar;
    }



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
bool read_data(vector<float>& data, const char* fname);
float predict(const vector<float>& train, vector<float>& vec, int w);

bool equal(float f1, float f2, float e=0.01);

float DTWDistance(const float* s1, const float* s2, int w);
void fill(float v);



int main(int argc, const char** argv) {

    linalg::set(dot_1, 0.f, X_LEN);
    linalg::set(dot_2, 5000.f, X_LEN);
    cout << "#2 dots are ready, size " << X_LEN << endl;


    vector<float> train;
    train.reserve(5 * 1024 * 1024 * sizeof(float));

    for (int i = 0; i < files_num; ++i) {
        if (!read_data(train, fnames[i])) {
            help();
            return -1;
        }
    }

    int N = train.size() / VEC_LEN;
    int ones = 0;
    int zeros = 0;
    for (int i = 0; i < N; ++i) {
        if (train[i*VEC_LEN + CLS_IDX])
            ++ones;
        else
            ++zeros;
    }
    cout << "#STAT: ones " << ones << "; zeros " << zeros << endl;

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

            //if (0. == tmp[CLS_IDX])
            //    continue;

            cout << "#new vector [" << tmp[CLS_IDX] << ", " << tmp[DIST1_IDX] << ", " << tmp[DIST2_IDX] << ", " << tmp[X_IDX+0] << ", " << tmp[X_IDX+1] << ", "
                 << tmp[X_IDX+2] << ", " << tmp[X_IDX+3] << ", ...]"
                 << endl;

            time_t start = clock();
            float p = predict(train, tmp, W);
            time_t finish = clock();

            cout << "#" << name << " : " << p << endl;
            cout << "#time " << ((float)(finish - start) / CLOCKS_PER_SEC) << endl;
        }

    }


    return 0;
}

int DTWDistance(const float* s1, const float* s2, int w, float giveup_threshold, float* ret) {

    fill(numeric_limits<float>::infinity());

    DTW[0] = 0.f;

    bool giveup = false;

    float global_min_dist = numeric_limits<float>::infinity();

    for (int i = 0; i < X_LEN && !giveup; ++i) {
        int begin = (i - w) > 0 ? i - w : 0;
        int end = (i + w) < X_LEN ? i + w : X_LEN;

        float min_dist = numeric_limits<float>::infinity();

        for (int j = begin; j < end; ++j) {

            int dtwI = i + 1;
            int dtwJ = j + 1;

            float dist = (s1[i] - s2[j]) * (s1[i] - s2[j]);

            float min = DTW[(dtwI-1) * DTW_VEC_LEN + dtwJ] < DTW[dtwI * DTW_VEC_LEN + (dtwJ-1)] ? DTW[(dtwI-1) * DTW_VEC_LEN + dtwJ] : DTW[dtwI * DTW_VEC_LEN + (dtwJ-1)];
            min = min < DTW[(dtwI-1) * DTW_VEC_LEN + (dtwJ-1)] ? min : DTW[(dtwI-1) * DTW_VEC_LEN + (dtwJ-1)];

            DTW[dtwI * DTW_VEC_LEN + dtwJ] = dist + min;

            if (min_dist > DTW[dtwI * DTW_VEC_LEN + dtwJ]) {
                min_dist = DTW[dtwI * DTW_VEC_LEN + dtwJ];
                global_min_dist = min_dist;

                if (min_dist > giveup_threshold) {
                    giveup = true;
                    break;
                }
            }
        }
    }

    *ret = giveup ? global_min_dist : sqrt( DTW[DTW_VEC_LEN*DTW_VEC_LEN - 1] );

    return !giveup;
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
    cout << "#cls_node - one node of classification\n"
         << "#Usage: cat data.txt | cls_node train_data1.bin | cls_node train_data2.bin > result.txt"
         << endl;
}

bool read_data(vector<float>& data, const char* fname) {
    try {
        ifstream fin;
        fin.open(fname, ifstream::in | ifstream::binary);

        double d;
        while (fin.read((char*)&d, sizeof(double))) {
            data.push_back((float)d);
        }
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

bool operator< (const struct distance& cd1, const struct distance& cd2) {
    return cd1.id < cd2.id;
}


float predict(const vector<float>& train, vector<float>& vec, int w) {

    const int NO_DIST = -1.;

    int N = train.size() / VEC_LEN;
    cout << "# number of train rows: " << N << endl;

    float dist_to_1;
    float dist_to_2;

    int r = DTWDistance(dot_1, &vec[X_IDX], 500, numeric_limits<float>::infinity(), &dist_to_1);
        r = DTWDistance(dot_2, &vec[X_IDX], 500, numeric_limits<float>::infinity(), &dist_to_2);

    cout << "#distances: " << dist_to_1 << "; " << dist_to_2 << "; " << endl;


    float delta = DELTA;
    float LOW1 = (dist_to_1 - delta) > 0.f ? dist_to_1 - delta : 0.f;
    float HIGH1 = dist_to_1 + delta;
    float LOW2 = (dist_to_2 - delta) > 0.f ? dist_to_2 - delta : 0.f;
    float HIGH2 = dist_to_2 + delta;
    float radius = delta;

    //set<struct distance> visited;
    vector<float> visited(N, numeric_limits<float>::infinity());

    int checked = 0;

    float dtw_giveup_threshold = numeric_limits<float>::infinity();

    float min_dist = numeric_limits<float>::infinity();
    int min_idx = -1;
/*
    while (min_dist == numeric_limits<float>::infinity() && radius < MAX_RADIUS) {

        for (int i = 1; i < N; ++i) {
            if (visited[i] == numeric_limits<float>::infinity() &&
                (LOW1 <= train[i * VEC_LEN + DIST1_IDX] && train[i * VEC_LEN + DIST1_IDX] <= HIGH1) &&
                (LOW2 <= train[i * VEC_LEN + DIST2_IDX] && train[i * VEC_LEN + DIST2_IDX] <= HIGH2)) {

                float dist;
                int r = DTWDistance(&train[i * VEC_LEN + X_IDX], &vec[X_IDX], w, dtw_giveup_threshold, &dist);
                visited[i] = dist;
                if (dtw_giveup_threshold > dist)
                    dtw_giveup_threshold = dist;

                //if (!r)
                //    cout << "# gave up as threshold is " << dtw_giveup_threshold << " vs " << dist << endl;

                ++checked;

                if (min_dist > dist && dist <= radius) {
                    min_dist = dist;
                    min_idx = i;
                    cout << "# >>" << min_dist << "; cls " << train[min_idx * VEC_LEN + CLS_IDX] << " vs " << vec[CLS_IDX]
                         << " (" << train[min_idx * VEC_LEN + DIST1_IDX] << "; " << train[min_idx * VEC_LEN + DIST2_IDX] << "; " << ")"
                         << endl;
                }
            }
            else {
                if (visited[i] <= radius && min_dist > visited[i]) {
                    min_dist = visited[i];
                    min_idx = i;
                    cout << "# >>" << min_dist << "; cls " << train[min_idx * VEC_LEN + CLS_IDX] << " vs " << vec[CLS_IDX]
                         << " (" << train[min_idx * VEC_LEN + DIST1_IDX] << "; " << train[min_idx * VEC_LEN + DIST2_IDX] << "; " << ")"
                         << endl;

                }
            }


        }

        radius += delta;
        LOW1 = (LOW1 - delta) > 0.f ? LOW1 - delta : 0.f;
        HIGH1 = HIGH1 + delta;
        LOW2 = (LOW2 - delta) > 0.f ? LOW2 - delta : 0.f;
        HIGH2 = HIGH2 + delta;

        cout << "#radius: " << radius << "; checked: " << checked << endl;
    }
*/
    cout << "#checked: " << checked << endl;

    if (min_dist == numeric_limits<float>::infinity()) {
        cout << "#full scan" << endl;

        for (int i = 1; i < N; ++i) {
            float dist = numeric_limits<float>::infinity();

            if (visited[i] == numeric_limits<float>::infinity()) {
                int r = DTWDistance(&train[i * VEC_LEN + X_IDX], &vec[X_IDX + 0], w, dtw_giveup_threshold, &dist);
                visited[i] = dist;
                if (dtw_giveup_threshold > dist)
                    dtw_giveup_threshold = dist;

                if (train[i * VEC_LEN + CLS_IDX] == 1) {
                    cout << "# ONE dist: " << dist << (r ? "" : " (gave up)") << " distances " << train[i * VEC_LEN + DIST1_IDX] << "; " << train[i * VEC_LEN + DIST2_IDX] << endl;
                }

                //if (!r)
                //    cout << "# gave up as threshold is " << dtw_giveup_threshold << " vs " << dist << endl;

                ++checked;
            }
            else {
                dist = visited[i];
            }

            if (min_dist > dist) {
                min_dist = dist;
                min_idx = i;
                cout << "# >>" << min_dist << "; cls " << train[min_idx * VEC_LEN + CLS_IDX] << " vs " << vec[CLS_IDX]
                     << " (" << train[min_idx * VEC_LEN + DIST1_IDX] << "; " << train[min_idx * VEC_LEN + DIST2_IDX] << "; " << ")"
                     << endl;

            }

            //if (0 == (i % 100)) {
            //    cout << "# processed " << i << " rows" << endl;
            //}
        }

        cout << "#checked: " << checked << endl;
    }

    float p = train[min_idx * VEC_LEN + CLS_IDX];

    return p;
}





