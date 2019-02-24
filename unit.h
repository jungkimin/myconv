#ifndef __MYML_UNIT__
#define __MYML_UNIT__
#include "activation.h"
#include "normalizer.h"
enum UNIT { PERCEPTRONS, CONVOLUSIONS, BN_PERCEPTRONS, BN_CONVOLUSIONS };
class unit {
public:

	unit() {};
	UNIT unit_type;
	virtual void load_inputs(double** mini_batch_x) = 0;
	virtual void forward_propagation(double** forward) = 0;
	virtual void back_propagation(double** delta) = 0;
	virtual void update_preparation() = 0;
	virtual void update(double learning_rate) = 0;
	virtual void alloc_storage(int new_batch_size) = 0;
	Activation* activator;
	bool use_bias;
	int input_size;
	int output_size;
	int batch_size;
	double** input_container;
	double** entrance_port;
	virtual ~unit() {}
};

class perceptrons : public unit {
public:
	double* bias;
	double* weight;
	double** inner_deltaflow;

	double* gradients_bias;
	double* gradients_weight;
	int weight_len;
	int bias_len;

	double** pre_activated_container;

	perceptrons() {}
	perceptrons(int input_sz, int output_sz, ACTIVATION f = IDENTITY, bool use_Bias = true) {
		this->create(input_sz, output_sz, f, use_Bias);
	}

	void create(int input_sz, int output_sz, ACTIVATION f = IDENTITY, bool use_Bias = true) {
		use_bias = use_Bias;
		unit_type = PERCEPTRONS;
		output_size = output_sz;
		input_size = input_sz;
		batch_size = -1;
		bias_len = output_size;
		weight_len = input_size * output_size;

		Give_function(activator, f, output_size);

		weight = new double[weight_len];
		gradients_weight = new double[weight_len];
		km::fill_random(weight, weight_len);
		if (activator->formula == RELU || activator->formula == SWISH) {
			km::weight_normalization(weight, weight_len, 2.0 / (double)input_size);
		}
		else {
			km::weight_normalization(weight, weight_len, 1.0 / (double)(input_size + output_size));
		}
		if (use_bias == true) {
			bias = new double[bias_len];
			gradients_bias = new double[bias_len];
			km::fill_gaussian_dist(bias, 0.01, bias_len);
		}
	}

	virtual void alloc_storage(int new_batch_size) {
		if (batch_size > 0) {
			for (int i = 0; i < batch_size; i++) {
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];

			}
			delete[] inner_deltaflow;
			delete[] entrance_port;
			delete[] input_container;
			delete[] pre_activated_container;
		}

		batch_size = new_batch_size;
		activator->batch_size = batch_size;
		entrance_port = new double*[batch_size];
		pre_activated_container = new double*[batch_size];
		inner_deltaflow = new double*[batch_size];
		input_container = new double*[batch_size];

		for (int m = 0; m < batch_size; m++) {
			entrance_port[m] = new double[input_size];
			pre_activated_container[m] = new double[output_size];

			inner_deltaflow[m] = new double[output_size];
			input_container[m] = new double[input_size];
		}
	}

	void load_inputs(double** mini_batch_x) {

		for (int m = 0; m < batch_size; m++) {
			km::copy(input_container[m], mini_batch_x[m], input_size);
		}
	}

	void input_X_weight(double** container) { /*pre_activated_container <- matmul(input, weight)*/
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int w = 0; w < output_size; w++) {
					container[m][w] = bias[w];
				}
			}
		}
		else {
			for (int m = 0; m < batch_size; m++) {
				for (int w = 0; w < output_size; w++) {
					container[m][w] = 0.0;
				}
			}
		}

		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				for (int h = 0; h < input_size; h++) {
					container[m][w] += weight[w*input_size + h] * input_container[m][h];
				}
			}
		}
	}

	virtual void forward_propagation(double** posterior_layers_input_container) {
		input_X_weight(pre_activated_container);
		activator->batch_function(posterior_layers_input_container, pre_activated_container);
	}

	void get_gradients() {
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int w = 0; w < output_size; w++) {
					gradients_bias[w] += inner_deltaflow[m][w];
				}
			}
		}
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				for (int h = 0; h < input_size; h++) {
					gradients_weight[w*input_size + h] += inner_deltaflow[m][w] * input_container[m][h];
				}
			}
		}
	}

	void recitfying_delta() {
		for (int m = 0; m < batch_size; m++) {
			for (int h = 0; h < input_size; h++) {
				entrance_port[m][h] = 0.0;
				for (int w = 0; w < output_size; w++) {
					entrance_port[m][h] += inner_deltaflow[m][w] * weight[w*input_size + h];
				}
			}
		}
	}

	virtual void back_propagation(double** posterior_layers_entrance_port) {
		activator->batch_function_prime(inner_deltaflow, pre_activated_container); //inner_df[m][w] <= d(activated)/d(pre_activated)
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				inner_deltaflow[m][w] *= posterior_layers_entrance_port[m][w]; //inner_df[m][w] <= d(Loss)/d(ztelda[m][w])
			}
		}
		get_gradients();// get d(Loss)/d(Weight)
		recitfying_delta(); //entrance_port[m][w] <= d(Loss)/d(input[m][w])
	}

	virtual void update_preparation() {
		if (use_bias == true) {
			km::fill(gradients_bias, 0.0, bias_len);
		}
		km::fill(gradients_weight, 0.0, weight_len);
	}

	virtual void update(double learning_rate) {
		if (use_bias == true) {
			for (int i = 0; i < output_size; i++) {
				bias[i] -= gradients_bias[i] * learning_rate;
			}
		}
		for (int i = 0; i < weight_len; i++) {
			weight[i] -= gradients_weight[i] * learning_rate;
		}
	}

	virtual ~perceptrons() {
		if (batch_size > 0) {
			for (int i = 0; i < batch_size; i++) {
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];

			}
			delete[] inner_deltaflow;
			delete[] entrance_port;
			delete[] input_container;
			delete[] pre_activated_container;

		}
		delete activator;
		delete[] weight;
		delete[] gradients_weight;
		if (use_bias == true) {
			delete[] gradients_bias;
			delete[] bias;
		}
	}
};


class bn_perceptrons : public perceptrons {
public:

	batch_normalizer* normalizer;
	double** un_normalized_container;
	perceptrons* super;
	
	bn_perceptrons(int input_sz, int output_sz, ACTIVATION f = IDENTITY, bool use_Bias = true) {
		create(input_sz, output_sz, f, use_Bias);
	}

	void create(int input_sz, int output_sz, ACTIVATION f = IDENTITY, bool use_Bias = true) {
		super = (perceptrons*)this;
		super->create(input_sz, output_sz, f, use_Bias);
		normalizer = new batch_normalizer(output_size, 1);
		unit_type = BN_PERCEPTRONS;
	}

	virtual void alloc_storage(int new_batch_size) {
		if (batch_size > 0) {
			for (int i = 0; i < batch_size; i++) {
				delete[] un_normalized_container[i];
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];

			}
			delete[] inner_deltaflow;
			delete[] entrance_port;
			delete[] input_container;
			delete[] pre_activated_container;
			delete[] un_normalized_container;
		}
		batch_size = new_batch_size;
		activator->batch_size = batch_size;
		entrance_port = new double*[batch_size];
		pre_activated_container = new double*[batch_size];
		inner_deltaflow = new double*[batch_size];
		input_container = new double*[batch_size];
		un_normalized_container = new double*[batch_size];
		for (int m = 0; m < batch_size; m++) {
			entrance_port[m] = new double[input_size];
			pre_activated_container[m] = new double[output_size];
			un_normalized_container[m] = new double[output_size];
			inner_deltaflow[m] = new double[output_size];
			input_container[m] = new double[input_size];
		}
		normalizer->alloc_storage(batch_size);
	}

	virtual void forward_propagation(double** posterior_layers_input_container) {
		input_X_weight(un_normalized_container);
		normalizer->normalize(pre_activated_container, un_normalized_container);
		activator->batch_function(posterior_layers_input_container, pre_activated_container);
	}
	
	virtual void update_preparation() {
		if (use_bias == true) {
			for (int i = 0; i < output_size; i++) {
				gradients_bias[i] = 0.0;
			}
		}
		for (int i = 0; i < weight_len; i++) {
			gradients_weight[i] = 0.0;
		}
		for (int i = 0; i < normalizer->nb_group; i++) {
			normalizer->gradients_gamma[i] = 0.0;
			normalizer->gradients_beta[i] = 0.0;
		}
	}

	virtual void update(double learning_rate) {
		if (use_bias == true) {
			for (int i = 0; i < output_size; i++) {
				bias[i] -= gradients_bias[i] * learning_rate;
			}
		}
		for (int i = 0; i < weight_len; i++) {
			weight[i] -= gradients_weight[i] * learning_rate;
		}
		for (int i = 0; i < normalizer->nb_group; i++) {
			normalizer->gamma[i] -= normalizer->gradients_gamma[i] * learning_rate;
			normalizer->beta[i] -= normalizer->gradients_beta[i] * learning_rate;
		}
	}
	
	virtual void back_propagation(double** posterior_layers_entrance_port) {
		activator->batch_function_prime(inner_deltaflow, pre_activated_container);
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				inner_deltaflow[m][w] *= posterior_layers_entrance_port[m][w]; //inner_df[m][w] = d(Loss)/d(ztelda[m][w])
			}
		}
		normalizer->back_propagation(inner_deltaflow, un_normalized_container);
		get_gradients();
		recitfying_delta();
	}
	
	~bn_perceptrons() {
		if (batch_size > 0) {
			for (int i = 0; i < batch_size; i++) {
				delete[] un_normalized_container[i];
			}
			delete[] un_normalized_container;
		}
		delete normalizer;
	}
};

class conv_size {
public:
	int width_and_height;
	int nb_channel;
	int nb_dimension;
	conv_size() {}
	conv_size(int n_channel, int width_and_height, int n_dimension) {
		this->width_and_height = width_and_height;
		this->nb_channel = n_channel;
		this->nb_dimension = n_dimension;
	}
};

class Filter_width {
public:
	int wd;
	Filter_width(int filter_wd) {
		wd = filter_wd;
	}
};

class Stride {
public:
	int stride;
	Stride(int sd) {
		stride = sd;
	}
};
class convolutions : public unit {
public:

	double** filter;
	double* bias;
	double** gradients_filter;
	double* gradients_bias;

	int nb_channel;
	int nb_dimension;
	int input_width;
	int output_width;

	int pad;
	int filter_size; // a filter's n of params
	int filter_width;
	int stride;
	int batch_size;

	double** pre_activated_container;
	double** inner_deltaflow;

	convolutions() {}
	convolutions(int input_sz, int output_sz, conv_size format, Filter_width fw, Stride st, ACTIVATION f, bool use_Bias = true) {
		this->create(input_sz, output_sz, format, fw, st, f, use_Bias);
	}

	void create(int input_sz, int output_sz, conv_size format, Filter_width fw, Stride st, ACTIVATION f, bool use_Bias = true) {
		unit_type = CONVOLUSIONS;
		use_bias = use_Bias;
		batch_size = -1;
		if (fw.wd % 2 == 0) {
			cout << "filter_width size must be odd number" << endl;
		}
		output_size = output_sz;
		input_size = input_sz;
		input_width = format.width_and_height;
		nb_dimension = format.nb_dimension;
		nb_channel = format.nb_channel;

		filter_width = fw.wd;
		filter_size = filter_width * filter_width* nb_channel;
		pad = filter_width / 2;
		stride = st.stride;
		Give_function(activator, f, output_size);
		if (input_width % stride == 0) {
			output_width = (input_width / stride);
		}
		else {
			int step = (input_width / stride);
			output_width = (input_width / stride);
			if (step*stride + 1 <= input_width) {
				output_width += 1;
			}
		}
		cout << "output_width : " << output_width << " pad : " << pad << " stride : " << stride << endl;
		cout << "filter wd : " << filter_width << " filter size : " << filter_size << endl;
		cout << "dim : " << nb_dimension << " chann : " << nb_channel << endl;
		if (input_width*input_width*nb_channel != input_size) {
			cout << "input_size is something wrong" << endl;
		}
		if (output_width*output_width*nb_dimension != output_size) {

			cout << "output_size(" << output_width * output_width*nb_dimension << ")is something wrong" << endl;
		}

		filter = new double*[nb_dimension];
		gradients_filter = new double*[nb_dimension];
		for (int d = 0; d < nb_dimension; d++) {
			filter[d] = new double[filter_size];
			km::fill_random(filter[d], filter_size);
			km::weight_normalization(filter[d], filter_size, (2.0 / (double)input_size));
			gradients_filter[d] = new double[filter_size];
		}
		if (use_bias == true) {
			bias = new double[nb_dimension];
			gradients_bias = new double[nb_dimension];
			km::fill_gaussian_dist(bias, 0.00001, nb_dimension);
		}
	}

	virtual void update(double learning_rate) {
		for (int d = 0; d < nb_dimension; d++) {
			for (int i = 0; i < filter_size; i++) {
				filter[d][i] -= gradients_filter[d][i] * learning_rate;
			}
		}
		if (use_bias == true) {
			for (int d = 0; d < nb_dimension; d++) {
				bias[d] -= gradients_bias[d] * learning_rate;
			}
		}
	}

	virtual void update_preparation() {
		for (int d = 0; d < nb_dimension; d++) {
			for (int i = 0; i < filter_size; i++) {
				gradients_filter[d][i] = 0.0;
			}
		}
		if (use_bias == true) {
			for (int d = 0; d < nb_dimension; d++) {
				gradients_bias[d] = 0.0;
			}
		}
	}

	virtual ~convolutions() {
		if (batch_size > -1) {
			for (int i = 0; i < batch_size; i++) {
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];
			}
			delete[] entrance_port;
			delete[] pre_activated_container;
			delete[] inner_deltaflow;
			delete[] input_container;
		}
		for (int d = 0; d < nb_dimension; d++) {
			delete[] filter[d];
			delete[] gradients_filter[d];
		}
		delete[] filter;
		delete[] gradients_filter;
		if (use_bias == true) {
			delete[] bias;
			delete[] gradients_bias;
		}
	}

	virtual void alloc_storage(int new_batch_size) {

		if (batch_size > -1) {
			for (int i = 0; i < batch_size; i++) {
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];
			}
			delete[] entrance_port;
			delete[] pre_activated_container;
			delete[] inner_deltaflow;
			delete[] input_container;
		}

		batch_size = new_batch_size;
		activator->batch_size = batch_size;
		entrance_port = new double*[batch_size];
		pre_activated_container = new double*[batch_size];
		inner_deltaflow = new double*[batch_size];
		input_container = new double*[batch_size];

		for (int i = 0; i < batch_size; i++) {
			entrance_port[i] = new double[input_size];
			pre_activated_container[i] = new double[output_size];
			inner_deltaflow[i] = new double[output_size];
			input_container[i] = new double[input_size];
		}
	}

	virtual void load_inputs(double** x_data) {
		for (int m = 0; m < batch_size; m++) {
			km::copy(input_container[m], x_data[m], input_size);
		}
	}

	void input_X_filters(double** container) {
		int I;
		int J;
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int d = 0; d < nb_dimension; d++) {
					for (int i = 0; i < output_width*output_width; i++) {
						container[m][d*output_width*output_width + i] = bias[d];
					}
				}
			}
		}
		else {
			for (int m = 0; m < batch_size; m++) {
				for (int d = 0; d < nb_dimension; d++) {
					for (int i = 0; i < output_width*output_width; i++) {
						container[m][d*output_width*output_width + i] = 0.0;
					}
				}
			}
		}
		for (int m = 0; m < batch_size; m++) {
			for (int d = 0; d < nb_dimension; d++) {
				for (int c = 0; c < nb_channel; c++) {
					for (int i = 0; i < output_width; i++) {
						I = i * stride;
						for (int j = 0; j < output_width; j++) {
							J = j * stride;
							for (int p = 0; p < filter_width; p++) {
								if (I + p - pad < 0 || I + p - pad >= input_width) {
									continue;
								}
								for (int q = 0; q < filter_width; q++) {
									if (J + q - pad < 0 || J + q - pad >= input_width) {
										continue;
									}

									container[m][d* output_width* output_width + i * output_width + j] += filter[d][c*filter_width*filter_width + p * filter_width + q] *
										input_container[m][input_width*input_width*c + (I + p - pad)*input_width + (J + q - pad)];
								}
							}
						}
					}
				}
			}
		}
	}
	virtual void forward_propagation(double** next_layers_input_container) {
		input_X_filters(pre_activated_container);
		activator->batch_function(next_layers_input_container, pre_activated_container);
	}

	void get_gradients() {
		int I;
		int J;
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int d = 0; d < nb_dimension; d++) {
					for (int i = 0; i < output_width*output_width; i++) {
						gradients_bias[d] += inner_deltaflow[m][d*output_width*output_width + i];
					}
				}
			}
		}
		for (int m = 0; m < batch_size; m++) {
			for (int d = 0; d < nb_dimension; d++) {
				for (int c = 0; c < nb_channel; c++) {

					for (int i = 0; i < output_width; i++) {
						I = i * stride;
						for (int j = 0; j < output_width; j++) {
							J = j * stride;
							for (int p = 0; p < filter_width; p++) {
								if (I + p - pad < 0 || I + p - pad >= input_width) {
									continue;
								}
								for (int q = 0; q < filter_width; q++) {
									if (J + q - pad < 0 || J + q - pad >= input_width) {
										continue;
									}


									gradients_filter[d][c*filter_width*filter_width + p * filter_width + q]
										+= input_container[m][input_width*input_width*c + (I + p - pad)*input_width + (J + q - pad)] * inner_deltaflow[m][d* output_width* output_width + i * output_width + j];
								}
							}
						}
					}
				}
			}
		}
	}

	void recitfying_delta() {
		int I;
		int J;
		for (int m = 0; m < batch_size; m++) {
			km::fill(entrance_port[m], 0.0, input_size);
			for (int d = 0; d < nb_dimension; d++) {
				for (int c = 0; c < nb_channel; c++) {
					for (int i = 0; i < output_width; i++) {
						I = i * stride;
						for (int j = 0; j < output_width; j++) {
							J = j * stride;
							for (int p = 0; p < filter_width; p++) {
								if (I + p - pad < 0 || I + p - pad >= input_width) {
									continue;
								}
								for (int q = 0; q < filter_width; q++) {
									if (J + q - pad < 0 || J + q - pad >= input_width) {
										continue;
									}
									entrance_port[m][input_width*input_width*c + (I + p - pad)*input_width + (J + q - pad)]
										+= inner_deltaflow[m][d* output_width* output_width + i * output_width + j] * filter[d][c*filter_width*filter_width + p * filter_width + q];
								}
							}
						}
					}
				}
			}
		}
	}

	virtual void back_propagation(double** posterior_layers_entrance_port) {
		activator->batch_function_prime(inner_deltaflow, pre_activated_container);
		for (int m = 0; m < batch_size; m++) {
			for (int d = 0; d < nb_dimension; d++) {
				for (int i = 0; i < output_width*output_width; i++) {
					inner_deltaflow[m][d*output_width*output_width + i] *= posterior_layers_entrance_port[m][d*output_width*output_width + i];
				}
			}
		}
		get_gradients();
		recitfying_delta();
	}
};

class bn_convolutions : public convolutions {
public:
	batch_normalizer* normalizer;
	double** un_normalized_container;
	convolutions* super;
	bn_convolutions(int input_sz, int output_sz, conv_size format, Filter_width fw, Stride st, ACTIVATION f, bool use_Bias = true) {
		this->create(input_sz, output_sz, format, fw, st, f, use_Bias);
	}
	void create(int input_sz, int output_sz, conv_size format, Filter_width fw, Stride st, ACTIVATION f, bool use_Bias = true) {
		super = (convolutions*)this;
		super->create(input_sz, output_sz, format, fw, st, f, use_Bias);
		//normalizer = new batch_normalizer(output_size, 1);
		normalizer = new batch_normalizer(output_width*output_width, nb_dimension);
		unit_type = BN_CONVOLUSIONS;
	}

	virtual void forward_propagation(double** next_layers_input_container) {
		input_X_filters(un_normalized_container);
		normalizer->normalize(pre_activated_container, un_normalized_container);
		activator->batch_function(next_layers_input_container, pre_activated_container);
	}

	virtual void back_propagation(double** posterior_layers_entrance_port) {
		activator->batch_function_prime(inner_deltaflow, pre_activated_container);
		for (int m = 0; m < batch_size; m++) {
			for (int d = 0; d < nb_dimension; d++) {
				for (int i = 0; i < output_width*output_width; i++) {
					inner_deltaflow[m][d*output_width*output_width + i] *= posterior_layers_entrance_port[m][d*output_width*output_width + i];
				}
			}
		}
		normalizer->back_propagation(inner_deltaflow, un_normalized_container);
		get_gradients();
		recitfying_delta();
	}

	virtual void update_preparation() {
		for (int d = 0; d < nb_dimension; d++) {
			for (int i = 0; i < filter_size; i++) {
				gradients_filter[d][i] = 0.0;
			}
		}
		if (use_bias == true) {
			for (int d = 0; d < nb_dimension; d++) {
				gradients_bias[d] = 0.0;
			}
		}
		for (int i = 0; i < normalizer->nb_group; i++) {
			normalizer->gradients_gamma[i] = 0.0;
			normalizer->gradients_beta[i] = 0.0;
		}
	}
	virtual void update(double learning_rate) {
		for (int d = 0; d < nb_dimension; d++) {
			for (int i = 0; i < filter_size; i++) {
				filter[d][i] -= gradients_filter[d][i] * learning_rate;
			}
		}
		if (use_bias == true) {
			for (int d = 0; d < nb_dimension; d++) {
				bias[d] -= gradients_bias[d] * learning_rate;
			}
		}
		for (int i = 0; i < normalizer->nb_group; i++) {
			normalizer->gamma[i] -= normalizer->gradients_gamma[i] * learning_rate;
			normalizer->beta[i] -= normalizer->gradients_beta[i] * learning_rate;
		}
	}
	virtual void alloc_storage(int new_batch_size) {
		if (batch_size > -1) {
			for (int i = 0; i < batch_size; i++) {
				delete[] entrance_port[i];
				delete[] pre_activated_container[i];
				delete[] un_normalized_container[i];
				delete[] inner_deltaflow[i];
				delete[] input_container[i];
			}
			delete[] entrance_port;
			delete[] pre_activated_container;
			delete[] un_normalized_container;
			delete[] inner_deltaflow;
			delete[] input_container;
		}

		batch_size = new_batch_size;
		activator->batch_size = batch_size;
		normalizer->alloc_storage(batch_size);
		entrance_port = new double*[batch_size];
		pre_activated_container = new double*[batch_size];
		un_normalized_container = new double*[batch_size];
		inner_deltaflow = new double*[batch_size];
		input_container = new double*[batch_size];
		for (int i = 0; i < batch_size; i++) {
			entrance_port[i] = new double[input_size];
			pre_activated_container[i] = new double[output_size];
			un_normalized_container[i] = new double[output_size];
			inner_deltaflow[i] = new double[output_size];
			input_container[i] = new double[input_size];
		}
	}

	~bn_convolutions() {
		delete normalizer;
		for (int i = 0; i < batch_size; i++) {
			delete[] un_normalized_container[i];
		}
		delete[] un_normalized_container;
	}
};

#endif // !__MYML_UNIT__