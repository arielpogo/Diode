#include "Core.h"

int main(int argc, char* argv[]) {
	int height_parameter = 0;
	int ratio_parameter = 0;
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

					if (str[0] != '-') height_parameter = atoi(str); //if there is a parameter, assign it to the file name
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