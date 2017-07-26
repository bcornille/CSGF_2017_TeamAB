#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <iostream>
#include <fstream>

KOKKOS_INLINE_FUNCTION
unsigned char CalculateColor(int dwell, double distance, int max_iterations, double pixel_size) {
	if (distance <= 0.5*pixel_size) {
		return pow( distance / (0.5*pixel_size), 1.0 / 3.0) * 255.0;
	} else {
		return 255;
	}
}

KOKKOS_INLINE_FUNCTION
unsigned char CalculatePixel(double x0, double y0, double pixel_size) {
	const int max_iterations = 1000;
	const double escape_radius = 4.0;

	double x = 0.0, y = 0.0;
	double dx = 0.0, dy = 0.0;
	int dwell = 0;
	double distance = 0.0;
	while ( (x*x + y*y) < escape_radius*escape_radius && dwell < max_iterations) {
		const double x_new = x*x - y*y + x0;
		const double y_new = 2.0*x*y + y0;
		const double temp_dx = 2.0*(x*dx - y*dy) + 1.0;
		dy = 2.0*(x*dy + y*dx) + 1.0;
		dx = temp_dx;
		x = x_new;
		y = y_new;
		dwell++;
	}

	if ((x*x + y*y) >= escape_radius*escape_radius) {
		const double mag_z = sqrt(x*x + y*y);
		const double mag_dz = sqrt(dx*dx + dy*dy);
		distance = log(mag_z*mag_z) * mag_z / mag_dz;
	}

	return CalculateColor(dwell, distance, max_iterations, pixel_size);
}

int main(int argc, char **argv) {

	Kokkos::initialize(argc, argv);

	const double center_x = -0.75;
	const double center_y = 0.0;
	const double length_x = 2.75;
	const double length_y = 2.0;

	const double x_min = center_x - length_x/2.0;
	const double y_min = center_y - length_y/2.0;
	const double pixel_size = 0.001;
	const int pixels_x = length_x / pixel_size;
	const int pixels_y = length_y / pixel_size;

	Kokkos::View<unsigned char*> pixels("pixels", pixels_x*pixels_y);

	Kokkos::parallel_for(pixels_x*pixels_y, KOKKOS_LAMBDA(int i) {
			const int n_y = i / pixels_x;
			const double y = y_min + n_y * pixel_size;
			const int n_x = i % pixels_x;
			const double x = x_min + n_x * pixel_size;
			pixels[i] = CalculatePixel(x, y, pixel_size);
			}
			); 

	auto pixels_mirror = Kokkos::create_mirror_view(pixels);
	deep_copy(pixels_mirror, pixels);

	std::ofstream output_file("mandelbrot.pgm", std::ios::out | std::ofstream::binary);
	output_file << "P5\n" << pixels_x << " " << pixels_y << " 255\n";
	for (int i = 0; i < pixels_x*pixels_y; i++ ) {
		output_file << pixels_mirror[i];
	}
	output_file.close();

	Kokkos::finalize();
}
