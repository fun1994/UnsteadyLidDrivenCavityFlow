#include "UnsteadyLidDrivenCavityFlow.h"

void test(std::string grid) {
	UnsteadyLidDrivenCavityFlow ULDCF(1.0, 1.0, 0.001, 1.0, 10.0, 50.0, grid, 64, 64, 5000, 0.8, 0.5, 0.5, 1e-8);
	Data data;
	ULDCF.solve(data);
	data.save(grid);
}

int main() {
	test("staggered");
	test("collocated");
	return 0;
}
