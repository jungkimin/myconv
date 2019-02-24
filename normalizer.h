#pragma once
#ifndef __NORMALIZER__
#define __NORMALIZER__
#include "myml_core.h"
class batch_normalizer_ver_B {//feature map 을 통합한 mean, variance 사용, gamma 와 beta는 feature map의 channel 마다 따로
public:
	int nb_group;
	int element_size;
	int batch_size;

	double denumerator;
	double* gamma;
	double* beta;

	double* gradients_gamma;
	double* gradients_beta;

	double mean;
	double variance;

	double dLossdmean;
	double dLossdvariance;

	double stddev;
	double** normalized_container;

	int step;


	batch_normalizer_ver_B(int element_sz, int nb_g) {
		create(element_sz, nb_g);
	}
	void create(int element_sz, int nb_g) {
		nb_group = nb_g;
		batch_size = -1;
		element_size = element_sz;
		gamma = new double[nb_group];
		beta = new double[nb_group];
		gradients_beta = new double[nb_group];
		gradients_gamma = new double[nb_group];
		for (int i = 0; i < nb_group; i++) {
			gamma[i] = 1.0;
			beta[i] = 0.0;
		}
	}

	void alloc_storage(int new_batch_size) {
		if (batch_size > -1) {
			for (int m = 0; m < batch_size; m++) {
				delete[] normalized_container[m];
			}
			delete[] normalized_container;
		}
		batch_size = new_batch_size;
		denumerator = (double)batch_size * (double)element_size  * (double)nb_group;
		normalized_container = new double*[batch_size];
		for (int m = 0; m < batch_size; m++) {
			normalized_container[m] = new double[element_size * nb_group];
		}
	}

	~batch_normalizer_ver_B() {
		if (batch_size > -1) {
			for (int m = 0; m < batch_size; m++) {
				delete[] normalized_container[m];
			}
			delete[] normalized_container;
		}
		delete[] gamma;
		delete[] beta;
	}

	void back_propagation(double** inner_deltaflow, double** un_normalized_container) { // dLoss/dztelda
		dLossdmean = 0.0;
		dLossdvariance = 0.0;
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					gradients_beta[g] += inner_deltaflow[m][step + i];
					gradients_gamma[g] += inner_deltaflow[m][step + i] * normalized_container[m][step + i];
					inner_deltaflow[m][step + i] *= gamma[g];
					dLossdmean += -pow(variance, -0.5)*inner_deltaflow[m][step + i];
					dLossdvariance += (0.5)*(mean - un_normalized_container[m][step + i])*pow(variance, -1.5) * inner_deltaflow[m][step + i];
				}
			}
		}

		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					inner_deltaflow[m][step + i] = (inner_deltaflow[m][step + i] / stddev) + dLossdvariance * (2.0 / denumerator)*(un_normalized_container[m][step + i] - mean) + (dLossdmean / denumerator);
				}
			}

		}
	}


	void normalize(double** destination, double** un_normalized_container) {
		mean = 0.0;
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					mean += un_normalized_container[m][step + i];
				}
			}
		}
		mean = mean / denumerator;
		variance = 0.0;
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {

					variance += (un_normalized_container[m][step + i] - mean)*(un_normalized_container[m][step + i] - mean);
				}
			}
		}

		variance = (variance / (denumerator)) + epsilon;
		stddev = sqrt(variance);
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					normalized_container[m][step + i] = (un_normalized_container[m][step + i] - mean) / stddev;
					destination[m][step + i] = gamma[g] * normalized_container[m][step + i] + beta[g];
				}
			}
		}

	}
};

class batch_normalizer {//feature map 의 channel 마다 mean과 variance를 따로 구함
public:
	int nb_group;
	int element_size;
	int batch_size;

	double denumerator;
	double* gamma;
	double* beta;

	double* gradients_gamma;
	double* gradients_beta;

	double* mean;
	double* variance;

	double* dLossdmean;
	double* dLossdvariance;

	double* stddev;
	double** normalized_container;

	int step;


	batch_normalizer(int element_sz, int nb_g) {
		create(element_sz, nb_g);
	}
	void create(int element_sz, int nb_g) {
		nb_group = nb_g;
		batch_size = -1;
		element_size = element_sz;
		gamma = new double[nb_group];
		beta = new double[nb_group];
		gradients_beta = new double[nb_group];
		gradients_gamma = new double[nb_group];
		for (int i = 0; i < nb_group; i++) {
			gamma[i] = 1.0;
			beta[i] = 0.0;
		}
		mean = new double[nb_group];
		variance = new double[nb_group];
		dLossdmean = new double[nb_group];
		dLossdvariance = new double[nb_group];
		stddev = new double[nb_group];
	}

	void alloc_storage(int new_batch_size) {
		if (batch_size > -1) {
			for (int m = 0; m < batch_size; m++) {
				delete[] normalized_container[m];
			}
			delete[] normalized_container;
		}
		batch_size = new_batch_size;
		denumerator = (double)batch_size * (double)element_size;
		normalized_container = new double*[batch_size];
		for (int m = 0; m < batch_size; m++) {
			normalized_container[m] = new double[element_size * nb_group];
		}
	}

	~batch_normalizer() {
		if (batch_size > -1) {
			for (int m = 0; m < batch_size; m++) {
				delete[] normalized_container[m];
			}
			delete[] normalized_container;
		}
		delete[] gamma;
		delete[] beta;
		delete[] mean;
		delete[] variance;
		delete[] stddev;
	}

	void back_propagation(double** inner_deltaflow, double** un_normalized_container) { // dLoss/dztelda

		for (int g = 0; g < nb_group; g++) {
			dLossdmean[g] = 0.0;
			dLossdvariance[g] = 0.0;
			step = g * element_size;

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					gradients_beta[g] += inner_deltaflow[m][step + i];
					gradients_gamma[g] += inner_deltaflow[m][step + i] * normalized_container[m][step + i];
					inner_deltaflow[m][step + i] *= gamma[g];
					dLossdmean[g] += -pow(variance[g], -0.5)*inner_deltaflow[m][g*element_size + i];
					dLossdvariance[g] += (0.5)*(mean[g] - un_normalized_container[m][step + i])*pow(variance[g], -1.5) * inner_deltaflow[m][step + i];
				}
			}

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					inner_deltaflow[m][step + i] = (inner_deltaflow[m][step + i] / stddev[g]) + dLossdvariance[g] * (2.0 / denumerator)*(un_normalized_container[m][step + i] - mean[g]) + (dLossdmean[g] / denumerator);
				}
			}

		}
	}


	void normalize(double** destination, double** un_normalized_container) {

		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			mean[g] = 0.0;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					mean[g] += un_normalized_container[m][step + i];
				}
			}
			mean[g] = mean[g] / denumerator;
		}

		for (int g = 0; g < nb_group; g++) {
			variance[g] = 0.0;
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {

					variance[g] += (un_normalized_container[m][step + i] - mean[g])*(un_normalized_container[m][step + i] - mean[g]);
				}
			}

			variance[g] = (variance[g] / (denumerator)) + epsilon;
			stddev[g] = sqrt(variance[g]);
		}

		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					normalized_container[m][step + i] = (un_normalized_container[m][step + i] - mean[g]) / stddev[g];
					destination[m][step + i] = gamma[g] * normalized_container[m][step + i] + beta[g];
				}
			}
		}

	}
};


#endif // ! __NORMALIZER