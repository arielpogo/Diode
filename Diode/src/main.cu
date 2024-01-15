#include "core.h"
#include "camera.h"

__device__ camera* d_cam;
__device__ object* global_objects;
__device__ color_255* d_result;
#define NUM_OBJECTS 3

__global__ void init_kernel(time_t random_seed, int height, int width, double aspect_ratio) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	//create rng engine
	rng = new thrust::minstd_rand((uint32_t) random_seed);
	random = new thrust::uniform_real_distribution<float>(0.0f, 1.0f);
	printf("RNG Engine initialized\n");

	//set up camera
	d_cam = new camera;
	d_cam->image_height = height;
	d_cam->image_width = width;
	d_cam->hfov = 90;
	d_cam->samples_per_pixel = 10;
	d_cam->max_bounces = 10;
	d_cam->lookfrom = point3(0, 2, -5);
	d_cam->lookat = point3(0, 0, 0);
	d_cam->aspect_ratio = aspect_ratio;
	d_cam->initialize();
	printf("Camera initialized\n");

	//create the global objects
	global_objects = new object[3]{
		object(vec3(0.5f, 0.5f, 0.5f), vec3(0.0, -100.5, 0), material::lamertian, shape::sphere, 100.0f),
		object(vec3(0.5f, 0.5f, 0.5f), vec3(-1.0, 0.5, 0), material::lamertian, shape::sphere, 0.5f),
		object(vec3(0.5f, 0.5f, 0.5f), vec3(1.0, 0.0, 0), material::metal, shape::sphere, 0.6f)
	};
	printf("Global objects initialized\n");

	d_result = new color_255[d_cam->image_height * d_cam->image_width];
	printf("Result array initialized\n");

	printf("Successful initialization!\n");
}

__global__ void cleanup_kernel() {
	delete rng;
	delete random;
	delete d_cam;
	delete[] global_objects;

	printf("Successful clean up!\n");
} 

__host__ int main(int argc, char* argv[]) {
	int height_parameter = 0;
	int ratio_parameter = 0;
	std::string filename = "image";
	int block_size_parameter = 32; //default

	//Handle command line arguments
	if (argc > 1) { //if any arguments (beyond executable name)
		for (int i = 1; i < argc; i++) {
			char* str = argv[i];

			if (str[0] == '-') { //if this is an argument
				switch (str[1]) {

				case 'i':
					std::clog << "-i: display this menu\n-d: enable debug\n-o <name>: specify output file name\n-h <int>: specifiy image height\n-r <int> specify preset ratio (4 = 4:3, 16 = 16:9, 10 = 16:10, 1 = 1:1)" << std::endl;
					return 0;
					break;

				case 'd': //enable debug
					debug = true;
					std::clog << "Debug enabled." << std::endl;
					break;

				case 'b': //block size
					i++; //go to parameter
					if (i < argc) str = argv[i];
					else continue;

					if (str[0] != '-') block_size_parameter = atoi(str); //if there is a parameter, assign it
					else i--; //otherwise go back to this argument, so the next one is handled

					break;

				case 'o': //set output file name
					i++; //go to the parameter

					if (i < argc) str = argv[i];
					else continue;

					if (str[0] != '-') filename.assign(str); //if there is a parameter, assign it to the file name
					else i--; //otherwise go back to this argument, so the next one is handled
					break;

				case 'h': //set image height
					i++; //go to the parameter

					if (i < argc) str = argv[i];
					else continue;

					if (str[0] != '-') height_parameter = atoi(str); //if there is a parameter, assign it
					else i--; //otherwise go back to this argument, so the next one is handled
					break;

				case 'r':
					i++; //go to the parameter

					if (i < argc) str = argv[i];
					else continue;

					if (str[0] != '-') ratio_parameter = atoi(str); //if there is a parameter, assign it to the file name
					else i--; //otherwise go back to this argument, so the next one is handled
					break;
				}
			}
		}
	}

	output_file.open(filename + ".ppm");
	if (!output_file.is_open()) {
		std::cerr << "Error: " << filename << ".ppm" << " could not be opened/created." << std::endl;
		return 1;
	}
	
	////////////////////////////////
	//                            //
	//       Initialization       //
	//                            //
	////////////////////////////////

	int height = (height_parameter > 0) ? height_parameter : camera::DEFAULT_HEIGHT;
	double aspect_ratio;
	//pick aspect ratio
	switch (ratio_parameter) {
	case 10:
		aspect_ratio = 16.0 / 10.0;
		break;
	case 1:
		aspect_ratio = 1.0;
		break;
	case 4:
		aspect_ratio = 4.0 / 3.0;
		break;
	default:
		aspect_ratio = 16.0 / 9.0;
		break;
	}

	int width = static_cast<int>(aspect_ratio * height);
	width = (width < 1) ? 1 : width; //no 0px width image

	const int num_pixels = height * width;

	init_kernel<<<1,1>>> (time(nullptr), height, width, aspect_ratio);

	color_255* h_result = new color_255[num_pixels];
	CHKALLOC(h_result, num_pixels * sizeof(color_255));

	int num_blocks = (int)std::ceil(num_pixels / (double)block_size_parameter);
	int shm_size = 1024 * 48; //48KB
	//render_kernel<<<num_blocks, block_size_parameter, shm_size>>> ();

	////////////////////////////////
	//                            //
	//          Cleanup           //
	//                            //
	////////////////////////////////

cleanup:
	delete h_result;

	output_file.close();

	cleanup_kernel<<<1, 1>>> ();

	return 0;
}