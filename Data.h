#pragma once
#include <fstream>
#include <iomanip>
#include "armadillo"

class Data {
	void save(arma::vec& data, std::string path, std::string filename) {
		std::ofstream file("./data/" + path + "/" + filename + ".txt");
		file << std::fixed << std::setprecision(8);
		for (int i = 0; i < data.n_rows; i++) {
			file << data(i);
			if (i < data.n_rows - 1) {
				file << " ";
			}
		}
		file.close();
	}
	void save(arma::field<arma::mat>& data, std::string path, std::string filename) {
		std::ofstream file("./data/" + path + "/" + filename + ".txt");
		for (int i = 0; i < data.n_elem; i++) {
			for (int j = 0; j < data(i).n_rows; j++) {
				for (int k = 0; k < data(i).n_cols; k++) {
					file << data(i)(j, k);
					if (j < data(i).n_rows - 1 || k < data(i).n_cols - 1) {
						file << " ";
					}
				}
			}
			if (i < data.n_elem - 1) {
				file << "\n";
			}
		}
		file.close();
	}
public:
	arma::vec x_p;
	arma::vec y_p;
	arma::vec x_u;
	arma::vec y_u;
	arma::vec x_v;
	arma::vec y_v;
	arma::vec t;
	arma::field<arma::mat> p;
	arma::field<arma::mat> u;
	arma::field<arma::mat> v;
	arma::field<arma::mat> u_f;
	arma::field<arma::mat> v_f;
	void save(std::string path) {
		save(x_p, path, "x_p");
		save(y_p, path, "y_p");
		save(x_u, path, "x_u");
		save(y_u, path, "y_u");
		save(x_v, path, "x_v");
		save(y_v, path, "y_v");
		save(t, path, "t");
		save(p, path, "p");
		save(u, path, "u");
		save(v, path, "v");
	}
};
