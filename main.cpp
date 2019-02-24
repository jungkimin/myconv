#include "myml_model.h"
#include "mnist_data_frame.h"
#include "optimizer.h"


/*
 *  created by kimin jeong, chungang university, south korea.
 *  2019-02-24.
 *
 *  unit의 weight 초기화 시 xavier initialization 이 적용됩니다.
 *
 *  optimizer class를 통해 adam, rmsprop, momentum 을 사용할 수 있습니다.
 *
 *  bn_perceptrons, bn_convolution을 통해 batch normalization을 사용할 수 있습니다.
 *
 */

int main() {
	DataSet mnist("C:\\ai_data\\mnist_train_100.csv");
	mnist.scaling();
	multi_layer_net mlp(5);

	//mlp.layer[0] = new bn_perceptrons(784, 128, RELU);
	//mlp.layer[1] = new bn_perceptrons(128, 64, RELU);
	//mlp.layer[2] = new bn_perceptrons(64, 32, RELU);
	//mlp.layer[3] = new bn_perceptrons(32, 16, RELU);
	//mlp.layer[4] = new bn_perceptrons(16, 10, SOFTMAX);

	int dim1 = 6;
	int dim2 = 6;
	int dim3 = 8;
	int dim4 = 10;

	mlp.layer[0] = new convolutions(784, 196 * dim1, conv_size(1, 28, dim1), Filter_width(5), Stride(2), RELU, true);
	mlp.layer[1] = new bn_convolutions(196 * dim1, 49 * dim2, conv_size(dim1, 14, dim2), Filter_width(5), Stride(2), RELU, true);
	mlp.layer[2] = new convolutions(49 * dim2, 16 * dim3, conv_size(dim2, 7, dim3), Filter_width(5), Stride(2), RELU, true);
	mlp.layer[3] = new bn_convolutions(16 * dim3, 4 * dim4, conv_size(dim3, 4, dim4), Filter_width(5), Stride(2), RELU, true);
	mlp.layer[4] = new perceptrons(4 * dim4, 10, SOFTMAX, true);

	int batch_size = 16;
	mlp.set_batch_size(batch_size);

	int num_iter = mnist.size / batch_size;

	mnist.set_batch_size(batch_size);
	optimizer Optimizer(mlp);
	//Optimizer.use_momentum(0.9);
	//Optimizer.use_RMSprop(0.999);
	Optimizer.use_adaptive_momentum(0.9, 0.999);

	double lr = 0;
	double avg_cost = 0.0;
	double avg_accuracy = 0.0;
	while (true) {
		cout << "learning_rate : ";
		cin >> lr;
		cout << "epoch (exit if epoch < 0): ";
		int epoch;
		cin >> epoch;
		if (epoch < 0) {
			break;
		}
		for (int i = 0; i < epoch; i++) {
			for (int it = 0; it < num_iter; it++) {
				mnist.next_batch();
				mlp.get_gradients(mnist.batch_x, mnist.batch_y);
				Optimizer.update(lr);
			}
		}
		avg_cost = 0.0;
		avg_accuracy = 0.0;
		for (int it = 0; it < num_iter; it++) {
			mnist.next_batch();
			mlp.get_gradients(mnist.batch_x, mnist.batch_y);
			Optimizer.update(lr);
			avg_cost += mlp.loss(mnist.batch_x, mnist.batch_y);
			avg_accuracy += mlp.accuracy(mnist.batch_x, mnist.batch_y);
		}
		cout << "avg_loss : " << avg_cost/(double)num_iter << endl;
		cout << "avg_accuracy : " << avg_accuracy / (double)num_iter << endl;
	}
}