#ifndef __MYML_CORE__
#define __MYML_CORE__



#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>

#define epsilon 0.00000001
using namespace std;
random_device rdev;
mt19937 rEngine(rdev());
uniform_real_distribution<> uni_gen(-1, 1);


namespace km {

	void fill(double* arr, double val, int len) {
		for (int i = 0; i < len; i++) {
			arr[i] = val;
		}
	}

	void fill_random(double* arr, int len) {
		for (int i = 0; i < len; i++) {
			arr[i] = uni_gen(rEngine);
		}
	}
	void fill_gaussian_dist(double* arr, double stddev, int len) {
		normal_distribution<> gaussian_dist(0.0, stddev);
		for (int i = 0; i < len; i++) {
			arr[i] = gaussian_dist(rEngine);
		}
	}

	void scalar_mult(double* dest, double scalar, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] *= scalar;
		}
	}
	void copy(double* dest, double* src, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
	}

	double vector_mean(double** src, int nb_v, int vector_len) {
		double mean = 0.0;
		for (int m = 0; m < nb_v; m++) {
			for (int i = 0; i < vector_len; i++) {
				mean += src[m][i];
			}
		}
		mean = mean / ((double)nb_v * (double)vector_len);
		return mean;
	}

	double vector_variance(double** src, double mean, int nb_v, int vector_len) {

		double variance = 0.0;
		for (int m = 0; m < nb_v; m++) {
			for (int i = 0; i < vector_len; i++) {
				variance += (src[m][i] - mean) * (src[m][i] - mean);
			}
		}
		variance = variance / ((double)nb_v * (double)vector_len);
		return variance;
	}

	double softmax_loss(double* pred, double* label, int len) {
		double loss = 0.0;
		for (int i = 0; i < len; i++) {
			loss += (-1)*log(pred[i])*label[i];
		}
		return loss;
	}

	void softmax_cross_entropy_derivative(double* dest, double* pred, double* label, int len) {
		for (int i = 0; i < len; i++) {
			dest[i] = pred[i] - label[i];
		}
	}

	void weight_normalization(double* weight, int len, double variance) {
		double mean = 0.0;
		for (int i = 0; i < len; i++) {
			mean += weight[i];
		}
		mean = mean / (double)len;
		double stddev = 0.0;
		for (int i = 0; i < len; i++) {
			stddev += (weight[i] - mean)*(weight[i] - mean);
		}
		stddev = sqrt((stddev / (double)len) + epsilon);

		double denumerator = stddev / sqrt(variance);
		for (int i = 0; i < len; i++) {
			weight[i] = (weight[i] - mean) / denumerator;
		}
	}
}

#endif // !__MYML_CORE__
