#include "vec.h"
#include <iostream>

int32_t vals[] = {
	1, 2, 3, 4
};

int main(int argc, char const *argv[])
{
	int4 v = int4::load(vals);

	v = v + v;

	for (int i = 0; i < 4; i++)
		std::cout << v[i] << " ";
	std::cout << std::endl;

	return 0;
}