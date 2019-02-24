#ifndef __OPTIMIZER__
#define __OPTIMIZER__
#include "myml_model.h"
enum TERM { GRADIENT_DESCENT, MOMENTUM, RMSPROP, ADAPIVE_MOMENTUM };
class unit_manager {
public:
	UNIT u_type;
	TERM term;
	double beta1;
	double beta1_inv;
	double beta2;
	double beta2_inv;
	double curr_beta1;
	double curr_beta2;
	double denumerater1;
	double denumerater2;
	virtual void use_momentum(double beta) {}
	virtual void use_RMSprop(double beta) {}
	virtual void use_adaptive_momentum(double momentum_beta, double rms_beta) {}
	virtual void reset() {}
	virtual void update(const double& learning_rate) {}
	virtual ~unit_manager() {}
};
class bn_convolution_manager : public unit_manager {
public:
	bn_convolutions* trainee;

	double** momentum;
	double* momentum_bias;
	double** rms;
	double* rms_bias;
	int nb_filters;
	int filter_size;
	bool use_bias;
	double* momentum_beta;
	double* momenutm_gamma;
	double* rms_beta;
	double* rms_gamma;

	int nb_group;
	bn_convolution_manager(bn_convolutions* target) {
		u_type = BN_CONVOLUSIONS;
		trainee = target;
		nb_group = trainee->normalizer->nb_group;
		nb_filters = target->nb_dimension;
		filter_size = target->filter_size;
		use_bias = target->use_bias;
	}

	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = 0.0;
				}
			}
			for (int i = 0; i < nb_group; i++) {
				momenutm_gamma[i] = 0.0;
				momentum_beta[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = 0.0;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					rms[d][i] = 0.0;
				}
			}
			for (int i = 0; i < nb_group; i++) {
				rms_gamma[i] = 0.0;
				rms_beta[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					rms_bias[i] = 0.0;
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = 0.0;
					rms[d][i] = 0.0;
				}
			}
			for (int i = 0; i < nb_group; i++) {
				momenutm_gamma[i] = 0.0;
				momentum_beta[i] = 0.0;
				rms_gamma[i] = 0.0;
				rms_beta[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
		}
	}
	virtual void use_momentum(double mom_beta) {
		term = MOMENTUM;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			momentum[i] = new double[filter_size];
		}
		for (int i = 0; i < nb_group; i++) {
			momenutm_gamma = new double[nb_group];
			momentum_beta = new double[nb_group];
		}
		if (use_bias == true) {
			momentum_bias = new double[nb_filters];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta1;

		rms = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			rms[i] = new double[filter_size];
		}
		for (int i = 0; i < nb_group; i++) {
			rms_gamma = new double[nb_group];
			rms_beta = new double[nb_group];
		}
		if (use_bias == true) {
			rms_bias = new double[nb_filters];
		}
		curr_beta2 = 1.0;
		reset();
	}
	virtual void use_adaptive_momentum(double mom_beta, double RMSprop_beta) {
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;
		momentum = new double*[nb_filters];
		rms = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			momentum[i] = new double[filter_size];
			rms[i] = new double[filter_size];
		}
		for (int i = 0; i < nb_group; i++) {
			momenutm_gamma = new double[nb_group];
			momentum_beta = new double[nb_group];
			rms_gamma = new double[nb_group];
			rms_beta = new double[nb_group];
		}
		if (use_bias == true) {
			momentum_bias = new double[nb_filters];
			rms_bias = new double[nb_filters];
		}
		curr_beta2 = 1.0;
		curr_beta1 = 1.0;
		reset();
	}
	~bn_convolution_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			for (int i = 0; i < nb_filters; i++) {
				delete[] momentum[i];
			}
			delete[] momenutm_gamma;
			delete[] momentum_beta;
			delete[] momentum;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			for (int i = 0; i < nb_filters; i++) {
				delete[] rms[i];
			}
			delete[] rms_gamma;
			delete[] rms_beta;
			delete[] rms;
		}
	}
	virtual void update(const double& learning_rate) {
		if (term == MOMENTUM) {
			curr_beta1 *= beta1;
			denumerater1 = 1.0 - curr_beta1;

			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = beta1 * momentum[d][i] + beta1_inv * trainee->gradients_filter[d][i];
					trainee->filter[d][i] -= learning_rate * (momentum[d][i]/denumerater1);
				}
			}
			for (int i = 0; i < nb_group; i++) {
				momenutm_gamma[i] = beta1 * momenutm_gamma[i] + beta1_inv * trainee->normalizer->gradients_gamma[i];
				trainee->normalizer->gamma[i] -= learning_rate * (momenutm_gamma[i] / denumerater1);
				momentum_beta[i] = beta1 * momentum_beta[i] + beta1_inv * trainee->normalizer->gradients_beta[i];
				trainee->normalizer->gamma[i] -= learning_rate * (momentum_beta[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * (momentum_bias[i]/denumerater1);
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					rms[d][i] = beta2 * rms[d][i] + beta2_inv * (trainee->gradients_filter[d][i]) * (trainee->gradients_filter[d][i]);
					trainee->filter[d][i] -= (learning_rate / sqrt((rms[d][i]/denumerater2) + epsilon)) * (trainee->gradients_filter[d][i]);
				}
			}
			for (int i = 0; i < nb_group; i++) {
				rms_gamma[i] = beta2 * rms_gamma[i] + beta2_inv * trainee->normalizer->gradients_gamma[i] * trainee->normalizer->gradients_gamma[i];
				trainee->normalizer->gamma[i] -= (learning_rate / sqrt((rms_gamma[i]/denumerater2) + epsilon)) * (trainee->normalizer->gradients_gamma[i]);
				rms_beta[i] = beta2 * rms_beta[i] + beta2_inv * trainee->normalizer->gradients_beta[i] * trainee->normalizer->gradients_beta[i];
				trainee->normalizer->beta[i] -= (learning_rate / sqrt((rms_beta[i]/denumerater2) + epsilon)) * (trainee->normalizer->gradients_beta[i]);
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i]/denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			denumerater1 = 1.0 - curr_beta1;
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = beta1 * momentum[d][i] + beta1_inv * trainee->gradients_filter[d][i];
					rms[d][i] = beta2 * rms[d][i] + beta2_inv * (trainee->gradients_filter[d][i]) * (trainee->gradients_filter[d][i]);
					trainee->filter[d][i] -= (learning_rate / sqrt((rms[d][i] / denumerater2) + epsilon)) * (momentum[d][i] / denumerater1);
				}
			}
			for (int i = 0; i < nb_group; i++) {
				momenutm_gamma[i] = beta1 * momenutm_gamma[i] + beta1_inv * trainee->normalizer->gradients_gamma[i];
				rms_gamma[i] = beta2 * rms_gamma[i] + beta2_inv * trainee->normalizer->gradients_gamma[i] * trainee->normalizer->gradients_gamma[i];
				trainee->normalizer->gamma[i] -= (learning_rate / sqrt((rms_gamma[i] / denumerater2) + epsilon)) * (momenutm_gamma[i]/denumerater1);
				momentum_beta[i] = beta1 * momentum_beta[i] + beta1_inv * trainee->normalizer->gradients_beta[i];
				rms_beta[i] = beta2 * rms_beta[i] + beta2_inv * trainee->normalizer->gradients_beta[i] * trainee->normalizer->gradients_beta[i];
				trainee->normalizer->beta[i] -= (learning_rate / sqrt((rms_beta[i] / denumerater2) + epsilon)) * (momentum_beta[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
		}
		else {
			trainee->update(learning_rate);
		}
	}
};


class convolution_manager : public unit_manager {
public:
	convolutions* trainee;

	double** momentum;
	double* momentum_bias;
	double** rms;
	double* rms_bias;
	int nb_filters;
	int filter_size;
	bool use_bias;
	convolution_manager(convolutions* target) {
		u_type = CONVOLUSIONS;
		trainee = target;
		nb_filters = target->nb_dimension;
		filter_size = target->filter_size;
		use_bias = target->use_bias;
	}

	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = 0.0;
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = 0.0;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					rms[d][i] = 0.0;
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					rms_bias[i] = 0.0;
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = 0.0;
					rms[d][i] = 0.0;
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
		}
	}
	virtual void use_momentum(double mom_beta) {
		term = MOMENTUM;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			momentum[i] = new double[filter_size];
		}

		if (use_bias == true) {
			momentum_bias = new double[nb_filters];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta1;

		rms = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			rms[i] = new double[filter_size];
		}
		if (use_bias == true) {
			rms_bias = new double[nb_filters];
		}
		curr_beta2 = 1.0;
		reset();
	}
	virtual void use_adaptive_momentum(double mom_beta, double RMSprop_beta) {
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;
		momentum = new double*[nb_filters];
		rms = new double*[nb_filters];
		for (int i = 0; i < nb_filters; i++) {
			momentum[i] = new double[filter_size];
			rms[i] = new double[filter_size];
		}
		if (use_bias == true) {
			momentum_bias = new double[nb_filters];
			rms_bias = new double[nb_filters];
		}
		curr_beta2 = 1.0;
		curr_beta1 = 1.0;
		reset();
	}
	~convolution_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			for (int i = 0; i < nb_filters; i++) {
				delete[] momentum[i];
			}
			delete[] momentum;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			for (int i = 0; i < nb_filters; i++) {
				delete[] rms[i];
			}
			delete[] rms;
		}
	}
	virtual void update(const double& learning_rate) {
		if (term == MOMENTUM) {
			curr_beta1 *= beta1;
			denumerater1 = 1.0 - curr_beta1;

			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = beta1 * momentum[d][i] + beta1_inv * trainee->gradients_filter[d][i];
					trainee->filter[d][i] -= learning_rate * (momentum[d][i]/denumerater1);
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * (momentum_bias[i]/denumerater1);
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					rms[d][i] = beta2 * rms[d][i] + beta2_inv * (trainee->gradients_filter[d][i]) * (trainee->gradients_filter[d][i]);
					trainee->filter[d][i] -= (learning_rate / sqrt((rms[d][i]/denumerater2) + epsilon)) * (trainee->gradients_filter[d][i]);
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i]/denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			denumerater1 = 1.0 - curr_beta1;
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int d = 0; d < nb_filters; d++) {
				for (int i = 0; i < filter_size; i++) {
					momentum[d][i] = beta1 * momentum[d][i] + beta1_inv * trainee->gradients_filter[d][i];
					rms[d][i] = beta2 * rms[d][i] + beta2_inv * (trainee->gradients_filter[d][i]) * (trainee->gradients_filter[d][i]);
					trainee->filter[d][i] -= (learning_rate / sqrt((rms[d][i] / denumerater2) + epsilon)) * (momentum[d][i] / denumerater1);
				}
			}
			if (use_bias == true) {
				for (int i = 0; i < nb_filters; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
		}
		else {
			trainee->update(learning_rate);
		}
	}
};

class perceptron_manager : public unit_manager {
public:
	perceptrons* trainee;
	double* momentum;
	double* momentum_bias;
	double* rms;
	double* rms_bias;


	bool use_bias;
	int weight_len;
	int bias_len;
	perceptron_manager(){}
	perceptron_manager(perceptrons* target) {
		u_type = PERCEPTRONS;
		trainee = target;
		weight_len = trainee->weight_len;
		
		bias_len = trainee->bias_len;
		use_bias = trainee->use_bias;
		term = GRADIENT_DESCENT;
	}
	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = 0.0;
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
		}
	}
	virtual void use_momentum(double momentum_beta) {
		term = MOMENTUM;
		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double[trainee->weight_len];
		if (use_bias == true) {
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;

		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
		}
		curr_beta2 = 1.0;
		reset();
	}

	virtual void use_adaptive_momentum(double momentum_beta, double RMSprop_beta) {
		
		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		momentum = new double[trainee->weight_len];
		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		curr_beta2 = 1.0;
		reset();
	}

	virtual void update(const double& learning_rate) {

		if (term == MOMENTUM) {

			curr_beta1 *= beta1;
			denumerater1 = (1.0 - curr_beta1);
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				trainee->weight[i] -= learning_rate * momentum[i] / denumerater1;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * momentum_bias[i] / denumerater1;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * trainee->gradients_weight[i];
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			curr_beta2 *= beta2;
			denumerater1 = 1.0 - curr_beta1;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * (momentum[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
		}
		else {
			trainee->update(learning_rate);
		}
	}
	~perceptron_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			delete[] momentum;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			delete[] rms;
		}
	}
};


class bn_perceptron_manager : public unit_manager{
public:
	double* momentum;
	double* momentum_bias;
	double* rms;
	double* rms_bias;
	bool use_bias;
	int weight_len;
	int bias_len;
	double momentum_gamma;
	double momentum_beta;
	double rms_gamma;
	double rms_beta;
	bn_perceptrons* trainee;
	bn_perceptron_manager(bn_perceptrons* target) {
		trainee = target;
		weight_len = trainee->weight_len;
		bias_len = trainee->bias_len;
		use_bias = trainee->use_bias;
		u_type = BN_PERCEPTRONS;
	}
	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
				}
			}
			momentum_gamma = 0.0;
			momentum_beta = 0.0;
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = 0.0;
				}
			}
			rms_gamma = 0.0;
			rms_beta = 0.0;
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
			momentum_gamma = 0.0;
			momentum_beta = 0.0;
			rms_gamma = 0.0;
			rms_beta = 0.0;
		}
	}
	virtual void use_momentum(double mom_beta) {
		term = MOMENTUM;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double[trainee->weight_len];
		if (use_bias == true) {
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;

		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
		}
		curr_beta2 = 1.0;
		reset();
	}

	virtual void use_adaptive_momentum(double mom_beta, double RMSprop_beta) {
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		beta1 = mom_beta;
		beta1_inv = 1.0 - beta1;
		momentum = new double[trainee->weight_len];
		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		curr_beta2 = 1.0;
		reset();
	}

	virtual void update(const double& learning_rate) {

		if (term == MOMENTUM) {
			curr_beta1 *= beta1;
			denumerater1 = 1.0 - curr_beta1;

			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				trainee->weight[i] -= learning_rate * momentum[i] / denumerater1;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * momentum_bias[i] / denumerater1;
				}
			}
			momentum_gamma = beta1 * momentum_gamma + beta1_inv * trainee->normalizer->gradients_gamma[0];
			trainee->normalizer->gamma[0] -= learning_rate * momentum_gamma / denumerater1;
			momentum_beta = beta1 * momentum_beta + beta1_inv * trainee->normalizer->gradients_beta[0];
			trainee->normalizer->beta[0] -= learning_rate * momentum_beta / denumerater1;
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * trainee->gradients_weight[i];
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
			rms_gamma = beta2 * rms_gamma + beta2_inv * trainee->normalizer->gradients_gamma[0] * trainee->normalizer->gradients_gamma[0];
			trainee->normalizer->gamma[0] -= (learning_rate / sqrt((rms_gamma / denumerater2) + epsilon)) * trainee->normalizer->gradients_gamma[0];

			rms_beta = beta2 * rms_beta + beta2_inv * trainee->normalizer->gradients_beta[0] * trainee->normalizer->gradients_beta[0];
			trainee->normalizer->beta[0] -= (learning_rate / sqrt((rms_beta / denumerater2) + epsilon)) * trainee->normalizer->gradients_beta[0];
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			curr_beta2 *= beta2;
			denumerater1 = 1.0 - curr_beta1;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * (momentum[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
			momentum_gamma = beta1 * momentum_gamma + beta1_inv * trainee->normalizer->gradients_gamma[0];
			momentum_beta = beta1 * momentum_beta + beta1_inv * trainee->normalizer->gradients_beta[0];

			rms_gamma = beta2 * rms_gamma + beta2_inv * trainee->normalizer->gradients_gamma[0] * trainee->normalizer->gradients_gamma[0];
			rms_beta = beta2 * rms_beta + beta2_inv * trainee->normalizer->gradients_beta[0] * trainee->normalizer->gradients_beta[0];

			trainee->normalizer->gamma[0] -= (learning_rate / sqrt((rms_gamma / denumerater2) + epsilon)) * (momentum_gamma / denumerater1);
			trainee->normalizer->beta[0] -= (learning_rate / sqrt((rms_beta / denumerater2) + epsilon)) * (momentum_beta / denumerater1);

		}
		else {
			trainee->update(learning_rate);
		}
	}
	~bn_perceptron_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			delete[] momentum;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			delete[] rms;
		}
	}
};
class optimizer {

private:
	double learning_rate;
public:
	multi_layer_net* trainee;
	unit_manager** manager;
	int n_layer;
	TERM term;
	int time;
	optimizer(multi_layer_net& target) {
		trainee = &target;
		n_layer = trainee->n_layer;
		term = GRADIENT_DESCENT;
		manager = new unit_manager*[n_layer];
		for (int i = 0; i < n_layer; i++) {
			if (trainee->layer[i]->unit_type == PERCEPTRONS) {
				manager[i] = new perceptron_manager((perceptrons*)trainee->layer[i]);
			}
			else if (trainee->layer[i]->unit_type == BN_PERCEPTRONS) {
				manager[i] = new bn_perceptron_manager((bn_perceptrons*)trainee->layer[i]);
			}
			else if (trainee->layer[i]->unit_type == CONVOLUSIONS) {
				manager[i] = new convolution_manager((convolutions*)trainee->layer[i]);
			}
			else if (trainee->layer[i]->unit_type == BN_CONVOLUSIONS) {
				manager[i] = new bn_convolution_manager((bn_convolutions*)trainee->layer[i]);
			}
		}
	}
	void use_momentum(double beta) {
		term = MOMENTUM;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_momentum(beta);
		}
	}
	void use_RMSprop(double beta) {
		term = RMSPROP;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_RMSprop(beta);
		}
	}
	void use_adaptive_momentum(double momentum_beta, double rms_beta) {
		term = ADAPIVE_MOMENTUM;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_adaptive_momentum(momentum_beta, rms_beta);
		}
	}
	void reset() {
		time = 0;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->reset();
		}
	}
	void set_learning_rate(double _learing_rate) {
		learning_rate = _learing_rate / (double)trainee->batch_size;
	}
	void update(double _learning_rate) {
		_learning_rate /= (double)trainee->batch_size;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->update(_learning_rate);
		}
	}
	~optimizer() {
		for (int i = 0; i < n_layer; i++) {
			delete manager[i];
		}
		delete[] manager;
	}
};

#endif // !__OPTIMIZER__