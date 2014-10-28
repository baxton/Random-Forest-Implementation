


//
// g++ -O3 dtw_fast.cpp -shared -o dtw_fast.dll
//


#include <iostream>
#include <limits>
#include <cmath>
#include <vector>

//#include <emmintrin.h>


using namespace std;


//#define X_LEN        (600 - 59 - 59)
//#define DTW_VEC_LEN  (X_LEN + 1)

#define ALIGNMENT __attribute__((aligned(4)))



/****
DTW = sp.zeros((X_LEN+1, X_LEN+1 ), dtype=np.float32)


def DTWDistance(s1, s2, w):

    w = max(w, abs(len(s1)-len(s2)))

    DTW.fill(float('inf'))
    DTW[0,0] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2

            dtwI = i + 1
            dtwJ = j + 1

            if i == 9999:
                z=0

            DTW[dtwI, dtwJ] = dist + min(DTW[dtwI-1,dtwJ], DTW[dtwI,dtwJ-1], DTW[dtwI-1,dtwJ-1])

    return sp.sqrt(DTW[len(s1), len(s2)])
****/

//float DTW [DTW_VEC_LEN * DTW_VEC_LEN] ALIGNMENT;
/*
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
*/

extern "C" {

__declspec(dllexport)
int DTWDistance(const float* s1, const float* s2, int size1, int size2, int w, float giveup_threshold, float* ret) {

    int DTW_ROWS = size1 + 1;
    int DTW_VEC_LEN = size2 + 1;

    //cout << "RET: " << *ret << endl;
    //cout << "S1[0]: " << s1[0] << endl;

    //cout << size1 << endl << size2 << endl << DTW_ROWS << endl << DTW_VEC_LEN << endl << w << endl;

    vector<float> DTW(DTW_ROWS * DTW_VEC_LEN, numeric_limits<float>::infinity());
    //cout << "DTW size: " << DTW.size() << endl;

    DTW[0] = 0.f;

    float global_min_dist = numeric_limits<float>::infinity();
    bool giveup = false;

    for (int i = 0; i < size1 && !giveup; ++i) {
        int begin = (i - w) > 0 ? i - w : 0;
        int end = (i + w) < size2 ? i + w : size2;

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
            }
        }

        if (min_dist > giveup_threshold) {
            giveup = true;
            //cout << "giveup" << endl;
        }
    }

    *ret = giveup ? global_min_dist : DTW[(DTW_ROWS-1) * DTW_VEC_LEN + (DTW_VEC_LEN - 1)];

    return !giveup;
}





}






