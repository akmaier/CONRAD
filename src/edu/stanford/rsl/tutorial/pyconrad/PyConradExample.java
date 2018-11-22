/*
 * Copyright (C) 2010-2018 Stefan Seitz
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.pyconrad;

import edu.stanford.rsl.conrad.pyconrad.PyConrad;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class PyConradExample {

	public static void main(String[] args) {
		
		// Execute Python functions
		PyConrad pyconrad = new PyConrad();
		pyconrad.exec("print('Python rocks!')");
		pyconrad.exec("print(args[0])", "hallo");
		long two = (long) pyconrad.eval("1+1");
		System.out.println(two);
		
		// Grids are automatically converted to numpy and vice-versa
        // (can be deactivated by pyconrad.setAutoConvertConradGrids(false))
		pyconrad.exec("import numpy as np");
		Grid3D random = (Grid3D) pyconrad.eval("np.random.rand(20,30,10)");
		random.show();
		
		Grid3D randomPlusOne = (Grid3D) pyconrad.eval("args[0]+1", random);
		randomPlusOne.show();
		
		// Execute C++ functions (automatic compilation if necessary, see square.cpp)
		pyconrad.exec("cpp_module = __import__('cppimport').imp('src.edu.stanford.rsl.tutorial.pyconrad.square')");
		Grid3D squared = (Grid3D) pyconrad.eval("cpp_module.square(args[0])", randomPlusOne);
		squared.show();
		
		// Different instances have separated variables
		PyConrad otherPyConrad = new PyConrad();
		otherPyConrad.exec("x=1");
		otherPyConrad.exec("print('x is in locals(): ' + str('x' in locals()))");
		pyconrad.exec("print('x is in locals(): ' + str('x' in locals()))");
	}

}
