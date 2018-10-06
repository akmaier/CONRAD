/*
<%
setup_pybind11(cfg)
%>
*/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>


namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<float> square(py::array_t<float> input)
{
	py::array_t<float> rtn({input.shape(0), input.shape(1), input.shape(2)});

	std::cout << "Hello from C++" << std::endl;
	auto a = input.template unchecked<3>();
	auto b = rtn.template mutable_unchecked<3>();

	for(int z = 0; z < a.shape(0); ++z) {
		for ( int y = 0; y < a.shape(1); y++ ) {
			for ( int x = 0; x < a.shape(2); x++ ) {
				b( z,y,x ) = a( z,y,x ) * a( z,y,x );
			}
		}
	}
	return rtn;
}

PYBIND11_MODULE( square, m )
{
	m.def("square", &square, "a"_a = "Numpy array to square");
}
