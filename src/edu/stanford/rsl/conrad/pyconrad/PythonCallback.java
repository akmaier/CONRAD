/*
 * Copyright (C) 2010-2018 Stefan Seitz
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.pyconrad;

import java.lang.Object;
import edu.stanford.rsl.conrad.pyconrad.PyConrad;

public interface PythonCallback {
	
	public void exec(PyConrad object, String code, Object... args );
	public Object eval(PyConrad object, String code, Object... args );

}
