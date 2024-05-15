#pragma once
#define _USE_MATH_DEFINES
#include "Data.h"

class UnsteadyLidDrivenCavityFlow {
	double rho;
	double L;
	double mu;
	double u_max;
	double omega;
	double T;
	std::string grid;
	int Nx;
	int Ny;
	int Nt;
	double dx;
	double dy;
	double dt;
	double alpha_p;
	double alpha_u;
	double alpha_v;
	double tol;
	double u_L(double t) {
		return u_max * sin(omega * t);
	}
	void initialize(Data& data) {
		if (grid == "staggered") {
			data.x_p = arma::linspace(0.5 * dx, L - 0.5 * dx, Nx);
			data.y_p = arma::linspace(0.5 * dy, L - 0.5 * dy, Ny);
			data.x_u = arma::linspace(0.0, L, Nx + 1);
			data.y_u = arma::linspace(0.5 * dy, L - 0.5 * dy, Ny);
			data.x_v = arma::linspace(0.5 * dx, L - 0.5 * dx, Nx);
			data.y_v = arma::linspace(0.0, L, Ny + 1);
			data.t = arma::linspace(0.0, T, Nt + 1);
			data.p = arma::field<arma::mat>(Nt + 1);
			data.p(0) = arma::zeros(Nx, Ny);
			data.u = arma::field<arma::mat>(Nt + 1);
			data.u(0) = arma::zeros(Nx + 1, Ny);
			data.v = arma::field<arma::mat>(Nt + 1);
			data.v(0) = arma::zeros(Nx, Ny + 1);
		}
		else if (grid == "collocated") {
			data.x_p = arma::linspace(0.5 * dx, L - 0.5 * dx, Nx);
			data.y_p = arma::linspace(0.5 * dy, L - 0.5 * dy, Ny);
			data.x_u = arma::linspace(0.5 * dx, L - 0.5 * dx, Nx);
			data.y_u = arma::linspace(0.5 * dy, L - 0.5 * dy, Ny);
			data.x_v = arma::linspace(0.5 * dx, L - 0.5 * dx, Nx);
			data.y_v = arma::linspace(0.5 * dy, L - 0.5 * dy, Ny);
			data.t = arma::linspace(0.0, T, Nt + 1);
			data.p = arma::field<arma::mat>(Nt + 1);
			data.p(0) = arma::zeros(Nx, Ny);
			data.u = arma::field<arma::mat>(Nt + 1);
			data.u(0) = arma::zeros(Nx, Ny);
			data.v = arma::field<arma::mat>(Nt + 1);
			data.v(0) = arma::zeros(Nx, Ny);
			data.u_f = arma::field<arma::mat>(Nt + 1);
			data.u_f(0) = arma::zeros(Nx + 1, Ny);
			data.v_f = arma::field<arma::mat>(Nt + 1);
			data.v_f(0) = arma::zeros(Nx, Ny + 1);
		}
	}
	void SIMPLE(Data& data, int i) {
		if (grid == "staggered") {
			data.p(i + 1) = data.p(i);
			data.u(i + 1) = data.u(i);
			data.v(i + 1) = data.v(i);
			int count = 0;
			while (true) {
				count++;
				arma::mat u_star = arma::zeros(Nx + 1, Ny);
				arma::mat A_W_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_S_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_P_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_N_u = arma::zeros(Nx + 1, Ny);
				arma::mat A_E_u = arma::zeros(Nx + 1, Ny);
				arma::mat Q_P_u = arma::zeros(Nx + 1, Ny);
				for (int j = 0; j < Nx + 1; j++) {
					for (int k = 0; k < Ny; k++) {
						if (j == 0 || j == Nx) {
							A_P_u(j, k) = 1.0;
						}
						else {
							if (i == 0) {
								A_P_u(j, k) += rho * dx * dy / dt;
								Q_P_u(j, k) += rho * data.u(i)(j, k) * dx * dy / dt;
							}
							else {
								A_P_u(j, k) += 3 * rho * dx * dy / (2 * dt);
								Q_P_u(j, k) += rho * (4 * data.u(i)(j, k) - data.u(i - 1)(j, k)) * dx * dy / (2 * dt);
							}
							A_P_u(j, k) += rho * (data.u(i + 1)(j, k) + data.u(i + 1)(j + 1, k)) / 4 * dy;
							A_E_u(j, k) += rho * (data.u(i + 1)(j, k) + data.u(i + 1)(j + 1, k)) / 4 * dy;
							A_W_u(j, k) += -rho * (data.u(i + 1)(j - 1, k) + data.u(i + 1)(j, k)) / 4 * dy;
							A_P_u(j, k) += -rho * (data.u(i + 1)(j - 1, k) + data.u(i + 1)(j, k)) / 4 * dy;
							if (k == Ny - 1) {
								Q_P_u(j, k) += -rho * u_L(data.t(i + 1)) * (data.v(i + 1)(j - 1, k + 1) + data.v(i + 1)(j, k + 1)) / 2 * dx;
							}
							else {
								A_P_u(j, k) += rho * (data.v(i + 1)(j - 1, k + 1) + data.v(i + 1)(j, k + 1)) / 4 * dx;
								A_N_u(j, k) += rho * (data.v(i + 1)(j - 1, k + 1) + data.v(i + 1)(j, k + 1)) / 4 * dx;
							}
							if (k > 0) {
								A_S_u(j, k) += -rho * (data.v(i + 1)(j - 1, k) + data.v(i + 1)(j, k)) / 4 * dx;
								A_P_u(j, k) += -rho * (data.v(i + 1)(j - 1, k) + data.v(i + 1)(j, k)) / 4 * dx;
							}
							A_P_u(j, k) += mu * dy / dx;
							A_E_u(j, k) += -mu * dy / dx;
							A_W_u(j, k) += -mu * dy / dx;
							A_P_u(j, k) += mu * dy / dx;
							if (k == Ny - 1) {
								A_P_u(j, k) += 2 * mu * dx / dy;
								Q_P_u(j, k) += 2 * mu * dx / dy * u_L(data.t(i + 1));
							}
							else {
								A_P_u(j, k) += mu * dx / dy;
								A_N_u(j, k) += -mu * dx / dy;
							}
							if (k == 0) {
								A_P_u(j, k) += 2 * mu * dx / dy;
							}
							else {
								A_S_u(j, k) += -mu * dx / dy;
								A_P_u(j, k) += mu * dx / dy;
							}
							Q_P_u(j, k) += (data.p(i + 1)(j - 1, k) - data.p(i + 1)(j, k)) * dy;
						}
					}
				}
				arma::mat A_u;
				arma::vec b_u;
				assemble(A_u, b_u, A_W_u, A_S_u, A_P_u, A_N_u, A_E_u, Q_P_u);
				b_u = arma::solve(A_u, b_u);
				reshape(u_star, b_u);
				arma::mat v_star = arma::zeros(Nx, Ny + 1);
				arma::mat A_W_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_S_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_P_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_N_v = arma::zeros(Nx, Ny + 1);
				arma::mat A_E_v = arma::zeros(Nx, Ny + 1);
				arma::mat Q_P_v = arma::zeros(Nx, Ny + 1);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny + 1; k++) {
						if (k == 0 || k == Ny) {
							A_P_v(j, k) = 1.0;
						}
						else {
							if (i == 0) {
								A_P_v(j, k) += rho * dx * dy / dt;
								Q_P_v(j, k) += rho * data.v(i)(j, k) * dx * dy / dt;
							}
							else {
								A_P_v(j, k) += 3 * rho * dx * dy / (2 * dt);
								Q_P_v(j, k) += rho * (4 * data.v(i)(j, k) - data.v(i - 1)(j, k)) * dx * dy / (2 * dt);
							}
							if (j < Nx - 1) {
								A_P_v(j, k) += rho * (data.u(i + 1)(j + 1, k - 1) + data.u(i + 1)(j + 1, k)) / 4 * dy;
								A_E_v(j, k) += rho * (data.u(i + 1)(j + 1, k - 1) + data.u(i + 1)(j + 1, k)) / 4 * dy;
							}
							if (j > 0) {
								A_W_v(j, k) += -rho * (data.u(i + 1)(j, k - 1) + data.u(i + 1)(j, k)) / 4 * dy;
								A_P_v(j, k) += -rho * (data.u(i + 1)(j, k - 1) + data.u(i + 1)(j, k)) / 4 * dy;
							}
							A_P_v(j, k) += rho * (data.v(i + 1)(j, k) + data.v(i + 1)(j, k + 1)) / 4 * dx;
							A_N_v(j, k) += rho * (data.v(i + 1)(j, k) + data.v(i + 1)(j, k + 1)) / 4 * dx;
							A_S_v(j, k) += -rho * (data.v(i + 1)(j, k - 1) + data.v(i + 1)(j, k)) / 4 * dx;
							A_P_v(j, k) += -rho * (data.v(i + 1)(j, k - 1) + data.v(i + 1)(j, k)) / 4 * dx;
							if (j == Nx - 1) {
								A_P_v(j, k) += 2 * mu * dy / dx;
							}
							else {
								A_P_v(j, k) += mu * dy / dx;
								A_E_v(j, k) += -mu * dy / dx;
							}
							if (j == 0) {
								A_P_v(j, k) += 2 * mu * dy / dx;
							}
							else {
								A_W_v(j, k) += -mu * dy / dx;
								A_P_v(j, k) += mu * dy / dx;
							}
							A_P_v(j, k) += mu * dx / dy;
							A_N_v(j, k) += -mu * dx / dy;
							A_S_v(j, k) += -mu * dx / dy;
							A_P_v(j, k) += mu * dx / dy;
							Q_P_v(j, k) += (data.p(i + 1)(j, k - 1) - data.p(i + 1)(j, k)) * dx;
						}
					}
				}
				arma::mat A_v;
				arma::vec b_v;
				assemble(A_v, b_v, A_W_v, A_S_v, A_P_v, A_N_v, A_E_v, Q_P_v);
				b_v = arma::solve(A_v, b_v);
				reshape(v_star, b_v);
				arma::mat p_prime = arma::zeros(Nx, Ny);
				arma::mat A_W_p = arma::zeros(Nx, Ny);
				arma::mat A_S_p = arma::zeros(Nx, Ny);
				arma::mat A_P_p = arma::zeros(Nx, Ny);
				arma::mat A_N_p = arma::zeros(Nx, Ny);
				arma::mat A_E_p = arma::zeros(Nx, Ny);
				arma::mat Q_P_p = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (j == 0 && k == 0) {
							A_P_p(j, k) = 1.0;
						}
						else {
							if (j < Nx - 1) {
								A_P_p(j, k) += pow(dy, 2) / A_P_u(j + 1, k);
								A_E_p(j, k) += -pow(dy, 2) / A_P_u(j + 1, k);
							}
							Q_P_p(j, k) += -u_star(j + 1, k) * dy;
							if (j > 0) {
								A_W_p(j, k) += -pow(dy, 2) / A_P_u(j, k);
								A_P_p(j, k) += pow(dy, 2) / A_P_u(j, k);
							}
							Q_P_p(j, k) += u_star(j, k) * dy;
							if (k < Ny - 1) {
								A_P_p(j, k) += pow(dx, 2) / A_P_v(j, k + 1);
								A_N_p(j, k) += -pow(dx, 2) / A_P_v(j, k + 1);
							}
							Q_P_p(j, k) += -v_star(j, k + 1) * dx;
							if (k > 0) {
								A_S_p(j, k) += -pow(dx, 2) / A_P_v(j, k);
								A_P_p(j, k) += pow(dx, 2) / A_P_v(j, k);
							}
							Q_P_p(j, k) += v_star(j, k) * dx;
						}
					}
				}
				arma::mat A_p;
				arma::vec b_p;
				assemble(A_p, b_p, A_W_p, A_S_p, A_P_p, A_N_p, A_E_p, Q_P_p);
				b_p = arma::solve(A_p, b_p);
				reshape(p_prime, b_p);
				arma::mat u_prime = arma::zeros(Nx + 1, Ny);
				for (int j = 1; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						u_prime(j, k) += dy / A_P_u(j, k) * (p_prime(j - 1, k) - p_prime(j, k));
					}
				}
				arma::mat v_prime = arma::zeros(Nx, Ny + 1);
				for (int j = 0; j < Nx; j++) {
					for (int k = 1; k < Ny; k++) {
						v_prime(j, k) += dx / A_P_v(j, k) * (p_prime(j, k - 1) - p_prime(j, k));
					}
				}
				arma::mat u = u_star + u_prime;
				arma::mat v = v_star + v_prime;
				arma::mat p_new = data.p(i + 1) + alpha_p * p_prime;
				arma::mat u_new = alpha_u * u + (1 - alpha_u) * data.u(i + 1);
				arma::mat v_new = alpha_v * v + (1 - alpha_v) * data.v(i + 1);
				double res_p = arma::abs(data.p(i + 1) - p_new).max();
				double res_u = arma::abs(data.u(i + 1) - u_new).max();
				double res_v = arma::abs(data.v(i + 1) - v_new).max();
				std::cout << "grid=" << grid << " time=" << data.t(i + 1) << " count=" << count << " res_p=" << res_p << " res_u=" << res_u << " res_v=" << res_v << std::endl;
				data.p(i + 1) = p_new;
				data.u(i + 1) = u_new;
				data.v(i + 1) = v_new;
				if (res_p < tol && res_u < tol && res_v < tol) {
					break;
				}
			}
		}
		else if (grid == "collocated") {
			data.p(i + 1) = data.p(i);
			data.u(i + 1) = data.u(i);
			data.v(i + 1) = data.v(i);
			data.u_f(i + 1) = data.u_f(i);
			data.v_f(i + 1) = data.v_f(i);
			int count = 0;
			while (true) {
				count++;
				arma::mat u_star = arma::zeros(Nx, Ny);
				arma::mat A_W_u = arma::zeros(Nx, Ny);
				arma::mat A_S_u = arma::zeros(Nx, Ny);
				arma::mat A_P_u = arma::zeros(Nx, Ny);
				arma::mat A_N_u = arma::zeros(Nx, Ny);
				arma::mat A_E_u = arma::zeros(Nx, Ny);
				arma::mat Q_P_u = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (i == 0) {
							A_P_u(j, k) += rho * dx * dy / dt;
							Q_P_u(j, k) += rho * data.u(i)(j, k) * dx * dy / dt;
						}
						else {
							A_P_u(j, k) += 3 * rho * dx * dy / (2 * dt);
							Q_P_u(j, k) += rho * (4 * data.u(i)(j, k) - data.u(i - 1)(j, k)) * dx * dy / (2 * dt);
						}
						if (j < Nx - 1) {
							A_P_u(j, k) += rho * data.u_f(i + 1)(j + 1, k) * dy / 2;
							A_E_u(j, k) += rho * data.u_f(i + 1)(j + 1, k) * dy / 2;
						}
						if (j > 0) {
							A_W_u(j, k) += -rho * data.u_f(i + 1)(j, k) * dy / 2;
							A_P_u(j, k) += -rho * data.u_f(i + 1)(j, k) * dy / 2;
						}
						if (k == Ny - 1) {
							Q_P_u(j, k) += -rho * u_L(data.t(i + 1)) * data.v_f(i + 1)(j, k + 1) * dx;
						}
						else {
							A_P_u(j, k) += rho * data.v_f(i + 1)(j, k + 1) * dx / 2;
							A_N_u(j, k) += rho * data.v_f(i + 1)(j, k + 1) * dx / 2;
						}
						if (k > 0) {
							A_S_u(j, k) += -rho * data.v_f(i + 1)(j, k) * dx / 2;
							A_P_u(j, k) += -rho * data.v_f(i + 1)(j, k) * dx / 2;
						}
						if (j == Nx - 1) {
							A_P_u(j, k) += 2 * mu * dy / dx;
						}
						else {
							A_P_u(j, k) += mu * dy / dx;
							A_E_u(j, k) += -mu * dy / dx;
						}
						if (j == 0) {
							A_P_u(j, k) += 2 * mu * dy / dx;
						}
						else {
							A_W_u(j, k) += -mu * dy / dx;
							A_P_u(j, k) += mu * dy / dx;
						}
						if (k == Ny - 1) {
							A_P_u(j, k) += 2 * mu * dx / dy;
							Q_P_u(j, k) += 2 * mu * u_L(data.t(i + 1)) * dx / dy;
						}
						else {
							A_P_u(j, k) += mu * dx / dy;
							A_N_u(j, k) += -mu * dx / dy;
						}
						if (k == 0) {
							A_P_u(j, k) += 2 * mu * dx / dy;
						}
						else {
							A_S_u(j, k) += -mu * dx / dy;
							A_P_u(j, k) += mu * dx / dy;
						}
						if (j == 0) {
							Q_P_u(j, k) += (data.p(i + 1)(j, k) - data.p(i + 1)(j + 1, k)) / 2 * dy;
						}
						else if (j == Nx - 1) {
							Q_P_u(j, k) += (data.p(i + 1)(j - 1, k) - data.p(i + 1)(j, k)) / 2 * dy;
						}
						else {
							Q_P_u(j, k) += (data.p(i + 1)(j - 1, k) - data.p(i + 1)(j + 1, k)) / 2 * dy;
						}
					}
				}
				arma::mat A_u;
				arma::vec b_u;
				assemble(A_u, b_u, A_W_u, A_S_u, A_P_u, A_N_u, A_E_u, Q_P_u);
				b_u = arma::solve(A_u, b_u);
				reshape(u_star, b_u);
				arma::mat v_star = arma::zeros(Nx, Ny);
				arma::mat A_W_v = arma::zeros(Nx, Ny);
				arma::mat A_S_v = arma::zeros(Nx, Ny);
				arma::mat A_P_v = arma::zeros(Nx, Ny);
				arma::mat A_N_v = arma::zeros(Nx, Ny);
				arma::mat A_E_v = arma::zeros(Nx, Ny);
				arma::mat Q_P_v = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (i == 0) {
							A_P_v(j, k) += rho * dx * dy / dt;
							Q_P_v(j, k) += rho * data.v(i)(j, k) * dx * dy / dt;
						}
						else {
							A_P_v(j, k) += 3 * rho * dx * dy / (2 * dt);
							Q_P_v(j, k) += rho * (4 * data.v(i)(j, k) - data.v(i - 1)(j, k)) * dx * dy / (2 * dt);
						}
						if (j < Nx - 1) {
							A_P_v(j, k) += rho * data.u_f(i + 1)(j + 1, k) * dy / 2;
							A_E_v(j, k) += rho * data.u_f(i + 1)(j + 1, k) * dy / 2;
						}
						if (j > 0) {
							A_W_v(j, k) += -rho * data.u_f(i + 1)(j, k) * dy / 2;
							A_P_v(j, k) += -rho * data.u_f(i + 1)(j, k) * dy / 2;
						}
						if (k < Ny - 1) {
							A_P_v(j, k) += rho * data.v_f(i + 1)(j, k + 1) * dx / 2;
							A_N_v(j, k) += rho * data.v_f(i + 1)(j, k + 1) * dx / 2;
						}
						if (k > 0) {
							A_S_v(j, k) += -rho * data.v_f(i + 1)(j, k) * dx / 2;
							A_P_v(j, k) += -rho * data.v_f(i + 1)(j, k) * dx / 2;
						}
						if (j == Nx - 1) {
							A_P_v(j, k) += 2 * mu * dy / dx;
						}
						else {
							A_P_v(j, k) += mu * dy / dx;
							A_E_v(j, k) += -mu * dy / dx;
						}
						if (j == 0) {
							A_P_v(j, k) += 2 * mu * dy / dx;
						}
						else {
							A_W_v(j, k) += -mu * dy / dx;
							A_P_v(j, k) += mu * dy / dx;
						}
						if (k == Ny - 1) {
							A_P_v(j, k) += 2 * mu * dx / dy;
						}
						else {
							A_P_v(j, k) += mu * dx / dy;
							A_N_v(j, k) += -mu * dx / dy;
						}
						if (k == 0) {
							A_P_v(j, k) += 2 * mu * dx / dy;
						}
						else {
							A_S_v(j, k) += -mu * dx / dy;
							A_P_v(j, k) += mu * dx / dy;
						}
						if (k == 0) {
							Q_P_v(j, k) += (data.p(i + 1)(j, k) - data.p(i + 1)(j, k + 1)) / 2 * dx;
						}
						else if (k == Ny - 1) {
							Q_P_v(j, k) += (data.p(i + 1)(j, k - 1) - data.p(i + 1)(j, k)) / 2 * dx;
						}
						else {
							Q_P_v(j, k) += (data.p(i + 1)(j, k - 1) - data.p(i + 1)(j, k + 1)) / 2 * dx;
						}
					}
				}
				arma::mat A_v;
				arma::vec b_v;
				assemble(A_v, b_v, A_W_v, A_S_v, A_P_v, A_N_v, A_E_v, Q_P_v);
				b_v = arma::solve(A_v, b_v);
				reshape(v_star, b_v);
				arma::mat u_f_star = arma::zeros(Nx + 1, Ny);
				for (int j = 1; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (j == 1) {
							u_f_star(j, k) += (u_star(j - 1, k) + u_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 1, k)) / dx - ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 1, k)) / (2 * dx) + (data.p(i + 1)(j + 1, k) - data.p(i + 1)(j - 1, k)) / (2 * dx)) / 2);
						}
						else if (j == Nx - 1) {
							u_f_star(j, k) += (u_star(j - 1, k) + u_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 1, k)) / dx - ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 2, k)) / (2 * dx) + (data.p(i + 1)(j, k) - data.p(i + 1)(j - 1, k)) / (2 * dx)) / 2);
						}
						else {
							u_f_star(j, k) += (u_star(j - 1, k) + u_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 1, k)) / dx - ((data.p(i + 1)(j, k) - data.p(i + 1)(j - 2, k)) / (2 * dx) + (data.p(i + 1)(j + 1, k) - data.p(i + 1)(j - 1, k)) / (2 * dx)) / 2);
						}
						if (i == 0) {
							u_f_star(j, k) += rho * dx * dy / (2 * dt) * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * (data.u_f(i)(j, k) - (data.u(i)(j - 1, k) + data.u(i)(j, k)) / 2);
						}
						else {
							u_f_star(j, k) += rho * dx * dy / dt * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * (data.u_f(i)(j, k) - (data.u(i)(j - 1, k) + data.u(i)(j, k)) / 2) - rho * dx * dy / (4 * dt) * (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * (data.u_f(i - 1)(j, k) - (data.u(i - 1)(j - 1, k) + data.u(i - 1)(j, k)) / 2);
						}
					}
				}
				arma::mat v_f_star = arma::zeros(Nx, Ny + 1);
				for (int j = 0; j < Nx; j++) {
					for (int k = 1; k < Ny; k++) {
						if (k == 1) {
							v_f_star(j, k) += (v_star(j, k - 1) + v_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 1)) / dy - ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 1)) / (2 * dy) + (data.p(i + 1)(j, k + 1) - data.p(i + 1)(j, k - 1)) / (2 * dy)) / 2);
						}
						else if (k == Ny - 1) {
							v_f_star(j, k) += (v_star(j, k - 1) + v_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 1)) / dy - ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 2)) / (2 * dy) + (data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 1)) / (2 * dy)) / 2);
						}
						else {
							v_f_star(j, k) += (v_star(j, k - 1) + v_star(j, k)) / 2 - dx * dy / 2 * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 1)) / dy - ((data.p(i + 1)(j, k) - data.p(i + 1)(j, k - 2)) / (2 * dy) + (data.p(i + 1)(j, k + 1) - data.p(i + 1)(j, k - 1)) / (2 * dy)) / 2);
						}
						if (i == 0) {
							v_f_star(j, k) += rho * dx * dy / (2 * dt) * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * (data.v_f(i)(j, k) - (data.v(i)(j, k - 1) + data.v(i)(j, k)) / 2);
						}
						else {
							v_f_star(j, k) += rho * dx * dy / dt * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * (data.v_f(i)(j, k) - (data.v(i)(j, k - 1) + data.v(i)(j, k)) / 2) - rho * dx * dy / (4 * dt) * (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * (data.v_f(i - 1)(j, k) - (data.v(i - 1)(j, k - 1) + data.v(i - 1)(j, k)) / 2);
						}
					}
				}
				arma::mat p_prime = arma::zeros(Nx, Ny);
				arma::mat A_W_p = arma::zeros(Nx, Ny);
				arma::mat A_S_p = arma::zeros(Nx, Ny);
				arma::mat A_P_p = arma::zeros(Nx, Ny);
				arma::mat A_N_p = arma::zeros(Nx, Ny);
				arma::mat A_E_p = arma::zeros(Nx, Ny);
				arma::mat Q_P_p = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (j == 0 && k == 0) {
							A_P_p(j, k) = 1.0;
						}
						else {
							if (j < Nx - 1) {
								A_P_p(j, k) += (1 / A_P_u(j, k) + 1 / A_P_u(j + 1, k)) / 2 * pow(dy, 2);
								A_E_p(j, k) += -(1 / A_P_u(j, k) + 1 / A_P_u(j + 1, k)) / 2 * pow(dy, 2);
							}
							Q_P_p(j, k) += -u_f_star(j + 1, k) * dy;
							if (j > 0) {
								A_W_p(j, k) += -(1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) / 2 * pow(dy, 2);
								A_P_p(j, k) += (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) / 2 * pow(dy, 2);
							}
							Q_P_p(j, k) += u_f_star(j, k) * dy;
							if (k < Ny - 1) {
								A_P_p(j, k) += (1 / A_P_v(j, k) + 1 / A_P_v(j, k + 1)) / 2 * pow(dx, 2);
								A_N_p(j, k) += -(1 / A_P_v(j, k) + 1 / A_P_v(j, k + 1)) / 2 * pow(dx, 2);
							}
							Q_P_p(j, k) += -v_f_star(j, k + 1) * dx;
							if (k > 0) {
								A_S_p(j, k) += -(1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) / 2 * pow(dx, 2);
								A_P_p(j, k) += (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) / 2 * pow(dx, 2);
							}
							Q_P_p(j, k) += v_f_star(j, k) * dx;
						}
					}
				}
				arma::mat A_p;
				arma::vec b_p;
				assemble(A_p, b_p, A_W_p, A_S_p, A_P_p, A_N_p, A_E_p, Q_P_p);
				b_p = arma::solve(A_p, b_p);
				reshape(p_prime, b_p);
				arma::mat u_prime = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (j == 0) {
							u_prime(j, k) = (p_prime(j, k) - p_prime(j + 1, k)) / (2 * A_P_u(j, k)) * dy;
						}
						else if (j == Nx - 1) {
							u_prime(j, k) = (p_prime(j - 1, k) - p_prime(j, k)) / (2 * A_P_u(j, k)) * dy;
						}
						else {
							u_prime(j, k) = (p_prime(j - 1, k) - p_prime(j + 1, k)) / (2 * A_P_u(j, k)) * dy;
						}
					}
				}
				arma::mat v_prime = arma::zeros(Nx, Ny);
				for (int j = 0; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						if (k == 0) {
							v_prime(j, k) = (p_prime(j, k) - p_prime(j, k + 1)) / (2 * A_P_v(j, k)) * dx;
						}
						else if (k == Ny - 1) {
							v_prime(j, k) = (p_prime(j, k - 1) - p_prime(j, k)) / (2 * A_P_v(j, k)) * dx;
						}
						else {
							v_prime(j, k) = (p_prime(j, k - 1) - p_prime(j, k + 1)) / (2 * A_P_v(j, k)) * dx;
						}
					}
				}
				arma::mat u_f_prime = arma::zeros(Nx + 1, Ny);
				for (int j = 1; j < Nx; j++) {
					for (int k = 0; k < Ny; k++) {
						u_f_prime(j, k) = (1 / A_P_u(j - 1, k) + 1 / A_P_u(j, k)) * (p_prime(j - 1, k) - p_prime(j, k)) / 2 * dy;
					}
				}
				arma::mat v_f_prime = arma::zeros(Nx, Ny + 1);
				for (int j = 0; j < Nx; j++) {
					for (int k = 1; k < Ny; k++) {
						v_f_prime(j, k) = (1 / A_P_v(j, k - 1) + 1 / A_P_v(j, k)) * (p_prime(j, k - 1) - p_prime(j, k)) / 2 * dx;
					}
				}
				arma::mat u = u_star + u_prime;
				arma::mat v = v_star + v_prime;
				arma::mat u_f = u_f_star + u_f_prime;
				arma::mat v_f = v_f_star + v_f_prime;
				arma::mat p_new = data.p(i + 1) + alpha_p * p_prime;
				arma::mat u_new = alpha_u * u + (1 - alpha_u) * data.u(i + 1);
				arma::mat v_new = alpha_v * v + (1 - alpha_v) * data.v(i + 1);
				arma::mat u_f_new = alpha_u * u_f + (1 - alpha_u) * data.u_f(i + 1);
				arma::mat v_f_new = alpha_v * v_f + (1 - alpha_v) * data.v_f(i + 1);
				double res_p = arma::abs(data.p(i + 1) - p_new).max();
				double res_u = arma::abs(data.u(i + 1) - u_new).max();
				double res_v = arma::abs(data.v(i + 1) - v_new).max();
				std::cout << "grid=" << grid << " time=" << data.t(i + 1) << " count=" << count << " res_p=" << res_p << " res_u=" << res_u << " res_v=" << res_v << std::endl;
				data.p(i + 1) = p_new;
				data.u(i + 1) = u_new;
				data.v(i + 1) = v_new;
				data.u_f(i + 1) = u_f_new;
				data.v_f(i + 1) = v_f_new;
				if (res_p < tol && res_u < tol && res_v < tol) {
					break;
				}
			}
		}
	}
	void assemble(arma::mat& A, arma::vec& b, arma::mat& A_W, arma::mat& A_S, arma::mat& A_P, arma::mat& A_N, arma::mat& A_E, arma::mat& Q_P) {
		A = arma::zeros(A_P.n_rows * A_P.n_cols, A_P.n_rows * A_P.n_cols);
		for (int i = 0; i < A_P.n_rows; i++) {
			for (int j = 0; j < A_P.n_cols; j++) {
				A(i * A_P.n_cols + j, i * A_P.n_cols + j) = A_P(i, j);
				if (i < A_P.n_rows - 1) {
					A(i * A_P.n_cols + j, (i + 1) * A_P.n_cols + j) = A_E(i, j);
				}
				if (i > 0) {
					A(i * A_P.n_cols + j, (i - 1) * A_P.n_cols + j) = A_W(i, j);
				}
				if (j < A_P.n_cols - 1) {
					A(i * A_P.n_cols + j, i * A_P.n_cols + (j + 1)) = A_N(i, j);
				}
				if (j > 0) {
					A(i * A_P.n_cols + j, i * A_P.n_cols + (j - 1)) = A_S(i, j);
				}
			}
		}
		b = arma::zeros(A_P.n_rows * A_P.n_cols);
		for (int i = 0; i < A_P.n_rows; i++) {
			for (int j = 0; j < A_P.n_cols; j++) {
				b(i * A_P.n_cols + j) = Q_P(i, j);
			}
		}
	}
	void reshape(arma::mat& phi, arma::vec& b) {
		for (int i = 0; i < phi.n_rows; i++) {
			for (int j = 0; j < phi.n_cols; j++) {
				phi(i, j) = b(i * phi.n_cols + j);
			}
		}
	}
public:
	UnsteadyLidDrivenCavityFlow(double rho, double L, double mu, double u_max, double P, double T, std::string grid, int Nx, int Ny, int Nt, double alpha_p, double alpha_u, double alpha_v, double tol) :rho(rho), L(L), mu(mu), u_max(u_max), omega(2 * M_PI / P), T(T), grid(grid), Nx(Nx), Ny(Ny), Nt(Nt), dx(L / Nx), dy(L / Ny), dt(T / Nt), alpha_p(alpha_p), alpha_u(alpha_u), alpha_v(alpha_v), tol(tol) {}
	void solve(Data& data) {
		initialize(data);
		for (int i = 0; i < Nt; i++) {
			SIMPLE(data, i);
		}
	}
};
