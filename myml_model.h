#ifndef __MYML_MODEL__
#define __MYML_MODEL__

#include "unit.h"
enum MODE {CLASSIFIER, LOGISTIC_REGRESSER};
class multi_layer_net {
public:

	unit** layer;
	int end_layer;
	int n_layer;
	int batch_size;
	int output_size;
	double** output;
	double** dLoss;
	MODE mode;
	multi_layer_net() { batch_size = -1; }

	multi_layer_net(int n) {
		n_layer = n;
		batch_size = -1;
		end_layer = n_layer - 1;
		layer = new unit*[n_layer];
	}
	void set_batch_size(int batch_sz) {

		this->batch_size = batch_sz;
		output_size = layer[end_layer]->output_size;

		cout << "class : " << output_size << endl;

		output = new double*[batch_size];
		dLoss = new double*[batch_size];
		for (int m = 0; m < batch_size; m++) {
			output[m] = new double[output_size];
			dLoss[m] = new double[output_size];

		}
		for (int i = 0; i < n_layer; i++) {
			layer[i]->alloc_storage(batch_size);
		}
	}

	void get_dCELoss(double** batch_y) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < output_size; i++) {

				dLoss[m][i] = output[m][i] - batch_y[m][i];
			}
		}
	}

	void update(double learning_rate) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->update(learning_rate);
		}
	}

	void gradients_init() {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->update_preparation();
		}
	}

	void batch_forward(double** batch_x) {
		layer[0]->load_inputs(batch_x);
		for (int i = 0; i < end_layer; i++) {
			layer[i]->forward_propagation(layer[i + 1]->input_container);
		}
		layer[end_layer]->forward_propagation(output);
	}

	void batch_backward() {
		layer[end_layer]->back_propagation(dLoss);
		for (int i = end_layer - 1; i >= 0; i--) {
			layer[i]->back_propagation(layer[i + 1]->entrance_port);
		}
	}

	void get_gradients(double** batch_x, double** batch_y) {

		gradients_init();

		batch_forward(batch_x);

		get_dCELoss(batch_y);
		batch_backward();

		//divide_gradient_by_batch_size();

	}

	double accuracy(double** batch_x, double** batch_y) {

		double acc = 0.0;
		batch_forward(batch_x);
		int pred;
		int ans;
		for (int m = 0; m < batch_size; m++) {
			pred = 0;
			ans = 0;
			for (int i = 1; i < output_size; i++) {
	
				if (output[m][i] > output[m][pred]) {
					pred = i;
				}
			}
			if (batch_y[m][pred] > 0.99) {
				acc += 1.0;
			}
		}
		acc = acc / (double)batch_size;
		return acc;
	}
	double loss(double** batch_x, double** batch_y) {

		double loss = 0.0;
		batch_forward(batch_x);
		for (int m = 0; m < batch_size; m++) {
			loss += km::softmax_loss(output[m], batch_y[m], output_size);
		}
		loss = loss / (double)batch_size;
		return loss;
	}

	~multi_layer_net() {

		for (int i = 0; i < n_layer; i++) {
			delete layer[i];
		}
		delete[] layer;
		if (batch_size > -1) {
			for (int m = 0; m < batch_size; m++) {
				delete[] output[m];
				delete[] dLoss[m];
			}
			delete[] dLoss;
			delete[] output;
		}
	}
};
#endif // !__MYML_MODEL__