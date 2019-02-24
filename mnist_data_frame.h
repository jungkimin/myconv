#pragma once
#include <string>
#include <fstream>
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;
class DataSet {
public:
	int n_rows;
	int n_cols;
	int n_class;
	double** data;
	double** label_onehot_vector;
	int size;
	double** batch_x;
	double** batch_y;
	int batch_size;
	int nb_batch;
	int curr_order;
	void set_batch_size(int batch_sz) {
		curr_order = -1;
		batch_size = batch_sz;
		nb_batch = size / batch_size;
		batch_x = new double*[batch_size];
		batch_y = new double*[batch_size];
		for (int i = 0; i < batch_size; i++) {
			batch_x[i] = new double[n_cols];
			batch_y[i] = new double[n_class];
		}
		next_batch();
	}
	~DataSet() {
		for (int i = 0; i < batch_size; i++) {
			delete[] batch_x[i];
			delete[] batch_y[i];
		}
		delete[] batch_x;
		delete[] batch_y;
		for (int i = 0; i < size; i++) {
			delete[] data[i];
			delete[] label_onehot_vector[i];
		}
		delete[] data;
		delete[] label_onehot_vector;

	}
	void next_batch() {
		curr_order++;
		int step = curr_order * batch_size;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < n_cols; i++) {
				batch_x[m][i] = data[step + m][i];
			}
			for (int i = 0; i < n_class; i++) {
				batch_y[m][i] = label_onehot_vector[step + m][i];
			}
		}
		if (curr_order >= nb_batch - 1) {
			curr_order = -1;
		}
	}
	void print_batch() {
		int step = (curr_order)* batch_size;
		if (curr_order == -1) {
			step = (nb_batch - 1)* batch_size;
		}
		for (int m = 0; m < batch_size; m++) {
			cout << step + m << " ";
			for (int i = 0; i < n_class; i++) {
				cout << batch_y[m][i] << " ";
			}
			cout << endl;
		}
	}
	DataSet() {}
	DataSet(string path) {

		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			n_rows = 0;
			n_cols = 0;
			n_class = 10;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}

				if (n_rows == 0) {
					for (int i = 0; i < (int)line.length(); i++) {
						if (line[i] == ',') {
							n_cols++;
						}
					}
				}
				n_rows++;
			}
			fin.close();
			fin.open(path);
			data = new double*[n_rows];
			label_onehot_vector = new double*[n_rows];

			int LINE = 0;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				data[LINE] = new double[n_cols];
				label_onehot_vector[LINE] = new double[n_class];
				for (int a = 0; a < n_class; a++) {
					label_onehot_vector[LINE][a] = 0.0;
				}
				string value_buffer;
				size = n_rows;
				int count = -1;
				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						if (count == -1) {
							int idx = atoi(value_buffer.c_str());
							label_onehot_vector[LINE][idx] = 1.0;
							value_buffer.clear();
						}
						else {
							data[LINE][count] = atof(value_buffer.c_str());
							value_buffer.clear();
						}
						count++;
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
				LINE++;
			}
		}
		fin.close();
	}
	void scaling() {
		for (int n = 0; n < n_rows; n++) {
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					data[n][i * 28 + j] = (data[n][i * 28 + j] / (double)255) - 0.5;
				}
			}
		}
	}
	void print() {
		for (int n = 0; n < n_rows; n++) {
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					if (data[n][i * 28 + j] > 0.1) {
						cout << "бс";
					}
					else {
						cout << "бр";
					}
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}

};