#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

struct pixel_t {
	unsigned char R;
	unsigned char G;
	unsigned char B;
	KOKKOS_INLINE_FUNCTION
	pixel_t(unsigned char _R = 0, unsigned char _G = 0, unsigned char _B = 0) :
		R(_R), G(_G), B(_B) {}
	KOKKOS_INLINE_FUNCTION
	pixel_t(float H, float S, float V) {
		float C = (V*S);
		float H_p = 360.0*H/60.0;
		float X = C * (1.0 - fabs((fmod(H_p,2.0f) - 1.0)));

		if(H_p >= 0.0 && H_p <= 1.0) {
			R = 255.0*C;
			G = 255.0*X;
			B = 0.0;
		} else if(H_p > 1.0 && H_p <= 2.0) {
			R = 255.0*X;
			G = 255.0*C;
			B = 0.0;
		} else if(H_p > 2.0 && H_p <= 3.0) {
			R = 0.0;
			G = 255.0*C;
			B = 255.0*X;
		} else if(H_p > 3.0 && H_p <= 4.0) {
			R = 0.0;
			G = 255.0*X;
			B = 255.0*C;
		} else if(H_p > 4.0 && H_p <= 5.0) {
			R = 255.0*X;
			G = 0.0;
			B = 255.0*C;
		} else if(H_p > 5.0 && H_p <= 6.0) {
			R = 255.0*C;
			G = 0.0;
			B = 255.0*X;
		}
	}
};

std::ostream& operator << (std::ostream &out, const pixel_t& pixel) {
	out << pixel.R << pixel.G << pixel.B;
	return out;
}

KOKKOS_INLINE_FUNCTION
float CalculateHue(float dwell_scaled) {
  float H;

  // Remap the scaled dwell onto Hue
  if(dwell_scaled < 0.5) {
    dwell_scaled = 1.0 - 2.0*dwell_scaled;
    H = 1.0 - dwell_scaled;
  } else {
    dwell_scaled = 1.5*dwell_scaled - 0.5;
    H = dwell_scaled;
  }

  // Go around color wheel 10x to cover range 1->max_iterations
  H *= 10.0;
  H -= floor(H);

  return H;
}

KOKKOS_INLINE_FUNCTION
float CalculateSaturation(float dwell_scaled) {
  float S;

  // Remap the scaled dwell onto Saturation
  S = sqrt(dwell_scaled);
  S -= floor(S);

  return S;
}

KOKKOS_INLINE_FUNCTION
pixel_t CalculateColor(int dwell, float distance, float mag_z, float escape_radius,
		int max_iterations, float pixel_spacing) {
	float H, S, V;
	// Point is within Mandelbrot set, color white
	if(dwell >= max_iterations) {
		H = 0.0;
		S = 0.0;
		V = 1.0;
		return pixel_t(H,S,V);
	}

	// log scale distance
	const float dist_scaled = log2(distance / pixel_spacing / 2.0);

	// Convert scaled distance to Value in 8 intervals
	if (dist_scaled > 0.0) {
		V = 1.0;
	} else if (dist_scaled > -8.0) {
		V = (8.0 + dist_scaled) / 8.0;
	} else {
		V = 0.0;
	}

	// Lighten every other stripe
	if(dwell%2) {
		V *= 0.95;
	}

	// log scale dwell
	float dwell_scaled = log((float)dwell)/log((float)max_iterations);

	// Calculate current and next dwell band values
	H = CalculateHue(dwell_scaled);
	S = CalculateSaturation(dwell_scaled);

	// Convert to RGB and set pixel value
	return pixel_t(H, S, V);
}

KOKKOS_INLINE_FUNCTION
pixel_t CalculatePixel(float x0, float y0, float pixel_size) {
	const int max_iterations = 10000;
	const float escape_radius = 1 << 18;

	float x = 0.0, y = 0.0;
	float dx = 0.0, dy = 0.0;
	int dwell = 0;
	float distance = 0.0;
	float continuous_dwell = 0.0;
	while ( (x*x + y*y) < escape_radius*escape_radius && dwell < max_iterations) {
		const float x_new = x*x - y*y + x0;
		const float y_new = 2.0*x*y + y0;
		const float temp_dx = 2.0*(x*dx - y*dy) + 1.0;
		dy = 2.0*(x*dy + y*dx) + 1.0;
		dx = temp_dx;
		x = x_new;
		y = y_new;
		dwell++;
	}

	const float mag_z = sqrt(x*x + y*y);

	if ((x*x + y*y) >= escape_radius*escape_radius) {
		const float mag_dz = sqrt(dx*dx + dy*dy);
		distance = log(mag_z*mag_z) * mag_z / mag_dz;
	}

	return CalculateColor(dwell, distance, mag_z, escape_radius, max_iterations, pixel_size);
}

int main(int argc, char **argv) {

	Kokkos::initialize(argc, argv);
	{
		const float center_x = -0.75;
		const float center_y = 0.0;
		const float length_x = 2.75;
		const float length_y = 2.0;

		const float x_min = center_x - length_x/2.0;
		const float y_min = center_y - length_y/2.0;
		const float pixel_size = 0.001;
		const int pixels_x = length_x / pixel_size;
		const int pixels_y = length_y / pixel_size;

		Kokkos::View<pixel_t*> pixels("pixels", pixels_x*pixels_y);

		Kokkos::parallel_for(pixels_x*pixels_y, KOKKOS_LAMBDA(int i) {
				const int n_y = i / pixels_x;
				const float y = y_min + n_y * pixel_size;
				const int n_x = i % pixels_x;
				const float x = x_min + n_x * pixel_size;
				pixels[i] = CalculatePixel(x, y, pixel_size);
				}
				); 

		auto pixels_mirror = Kokkos::create_mirror_view(pixels);
		deep_copy(pixels_mirror, pixels);

		std::ofstream output_file("mandelbrot.pgm", std::ios::out | std::ofstream::binary);
		output_file << "P6\n" << pixels_x << " " << pixels_y << " 255\n";
		for (int i = 0; i < pixels_x*pixels_y; i++ ) {
			output_file << pixels_mirror[i];
		}
		output_file.close();

	}
	Kokkos::finalize();
}
