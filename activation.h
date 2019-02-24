#ifndef __ACTIVATION__
#define __ACTIVATION__

#include "myml_core.h"
enum ACTIVATION { SIGMOID, RELU, TANH, IDENTITY, SWISH, SOFTMAX };
class Activation {
public:
	int len;
	int batch_size;
	ACTIVATION formula;
	Activation() {}
	Activation(int length) {
		len = length;
	}
	void set_batch(int batch_sz) {
		batch_size = batch_sz;
	}
	void print_what_it_is() {
		if (formula == SIGMOID) {
			std::cout << "SIGMOID";
		}
		else if (formula == RELU) {
			std::cout << "RELU";
		}
		else if (formula == TANH) {
			std::cout << "TANH";
		}
		else if (formula == IDENTITY) {
			std::cout << "IDENTITY";
		}
		else if (formula == SWISH) {
			std::cout << "SWISH";
		}
		else if (formula == SOFTMAX) {
			std::cout << "SOFTMAX";
		}
	}
	virtual void function(double* dest, double* src) = 0;
	virtual void function_prime(double* dest, double* src) = 0;

	virtual void batch_function(double** dest, double** src) = 0;
	virtual void batch_function_prime(double** dest, double** src) = 0;
};
class Sigmoid : public Activation {
public:
	double sigmoid;
	Sigmoid() {
		formula = SIGMOID;
	}
	Sigmoid(int length) : Activation(length) {
		formula = SIGMOID;
	}
	virtual void function(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				dest[i] = 1.0 / (1.0 + exp(-src[i]));
			}
			else {
				dest[i] = exp(src[i]) / (1.0 + exp(src[i]));
			}
		}
	}
	virtual void function_prime(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				sigmoid = 1.0 / (1.0 + exp(-src[i]));
			}
			else {
				sigmoid = exp(src[i]) / (1.0 + exp(src[i]));
			}
			dest[i] = (1 - sigmoid)*sigmoid;
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					dest[m][i] = 1.0 / (1.0 + exp(-src[m][i]));
				}
				else {
					dest[m][i] = exp(src[m][i]) / (1.0 + exp(src[m][i]));
				}
			}
		}
	}
	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					sigmoid = 1.0 / (1.0 + exp(-src[m][i]));
				}
				else {
					sigmoid = exp(src[m][i]) / (1.0 + exp(src[m][i]));
				}
				dest[m][i] = (1 - sigmoid)*sigmoid;
			}
		}
	}
};
class Tanh : public Activation {
public:
	double enx;
	double ex;
	Tanh() {
		formula = TANH;
	}
	Tanh(int length) : Activation(length) {
		formula = TANH;
	}
	virtual void function(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				enx = exp(-src[i]);
				ex = 1.0 / enx;
			}
			else {
				ex = exp(src[i]);
				enx = 1.0 / ex;
			}
			dest[i] = (ex - enx) / (ex + enx);
		}
	}
	virtual void function_prime(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				enx = exp(-src[i]);
				ex = 1.0 / enx;
			}
			else {
				ex = exp(src[i]);
				enx = 1.0 / ex;
			}
			dest[i] = 1.0 - ((ex - enx)*(ex - enx)) / ((ex + enx)*(ex + enx));
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					enx = exp(-src[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(src[m][i]);
					enx = 1.0 / ex;
				}
				dest[m][i] = (ex - enx) / (ex + enx);
			}
		}
	}
	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					enx = exp(-src[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(src[m][i]);
					enx = 1.0 / ex;
				}
				dest[m][i] = 1.0 - ((ex - enx)*(ex - enx)) / ((ex + enx)*(ex + enx));
			}
		}
	}
};
class Relu : public Activation {
public:
	Relu() {
		formula = RELU;
	}
	Relu(int length) : Activation(length) {
		formula = RELU;
	}
	virtual void function(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				dest[i] = src[i];
			}
			else {
				dest[i] = 0;
			}
		}
	}

	virtual void function_prime(double* dest, double* src) {

		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				dest[i] = 1;
			}
			else {
				dest[i] = 0;
			}
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					dest[m][i] = src[m][i];
				}
				else {
					dest[m][i] = 0;
				}
			}
		}
	}

	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					dest[m][i] = 1;
				}
				else {
					dest[m][i] = 0;
				}
			}
		}
	}
};

class Softmax : public Activation {
public:
	double max;
	Softmax() {
		formula = SOFTMAX;
	}
	Softmax(int length) : Activation(length) {
		formula = SOFTMAX;
	}
	virtual void function(double* dest, double* src) {
		max = src[0];
		for (int i = 1; i < len; i++) {
			if (max < src[i]) {
				max = src[i];
			}
		}
		double base = 0.0;
		for (int i = 0; i < len; i++) {
			dest[i] = exp(src[i] - max);
			base += dest[i];
		}
		for (int i = 0; i < len; i++) {
			dest[i] = dest[i] / base;
		}
	}
	virtual void function_prime(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			dest[i] = 1.0;
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			max = src[m][0];
			for (int i = 1; i < len; i++) {
				if (max < src[m][i]) {
					max = src[m][i];
				}
			}
			double base = 0.0;
			for (int i = 0; i < len; i++) {
				dest[m][i] = exp(src[m][i] - max);
				base += dest[m][i];
			}
			for (int i = 0; i < len; i++) {
				dest[m][i] = dest[m][i] / base;
			}
		}
	}
	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				dest[m][i] = 1.0;
			}
		}
	}
};

class Identity : public Activation {
public:
	Identity() {
		formula = IDENTITY;
	}
	Identity(int length) : Activation(length) {
		formula = IDENTITY;
	}
	virtual void function(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
	}
	virtual void function_prime(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			dest[i] = 1;
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				dest[m][i] = src[m][i];
			}
		}
	}
	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				dest[m][i] = 1;
			}
		}
	}
};

class Swish : public Activation {
public:
	double sigmoid;
	Swish() {
		formula = SWISH;
	}
	Swish(int length) : Activation(length) {
		formula = SWISH;
	}
	virtual void function(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				dest[i] = src[i] / (1.0 + exp(-src[i]));
			}
			else {
				dest[i] = (src[i] * exp(src[i])) / (1.0 + exp(src[i]));
			}
		}
	}
	virtual void function_prime(double* dest, double* src) {
		for (int i = 0; i < len; i++) {
			if (src[i] >= 0) {
				sigmoid = 1.0 / (1.0 + exp(-src[i]));
			}
			else {
				sigmoid = exp(src[i]) / (1.0 + exp(src[i]));
			}
			dest[i] = sigmoid + src[i] * sigmoid*(1.0 - sigmoid);
		}
	}

	virtual void batch_function(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					dest[m][i] = src[m][i] / (1.0 + exp(-src[m][i]));
				}
				else {
					dest[m][i] = (src[m][i] * exp(src[m][i])) / (1.0 + exp(src[m][i]));
				}
			}
		}
	}
	virtual void batch_function_prime(double** dest, double** src) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (src[m][i] >= 0) {
					sigmoid = 1.0 / (1.0 + exp(-src[m][i]));
				}
				else {
					sigmoid = exp(src[m][i]) / (1.0 + exp(src[m][i]));
				}
				dest[m][i] = sigmoid + src[m][i] * sigmoid*(1.0 - sigmoid);
			}
		}
	}
};

void Give_function(Activation*& activator, ACTIVATION f, int len) {

	if (f == SIGMOID) {
		activator = new Sigmoid(len);
	}
	else if (f == RELU) {
		activator = new Relu(len);
	}
	else if (f == TANH) {
		activator = new Tanh(len);
	}
	else if (f == SOFTMAX) {
		activator = new Softmax(len);
	}
	else if (f == IDENTITY) {
		activator = new Identity(len);
	}
	else if (f == SWISH) {
		activator = new Swish(len);
	}
}
#endif // !__ACTIVATION__