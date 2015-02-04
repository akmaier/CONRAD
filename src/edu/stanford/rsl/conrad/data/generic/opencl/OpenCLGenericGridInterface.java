/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.opencl;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;
import edu.stanford.rsl.conrad.data.generic.GenericGrid;
import edu.stanford.rsl.conrad.data.generic.GenericGridOperatorInterface;
import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;

public interface OpenCLGenericGridInterface<T extends Gridable<T>> {
	
	OpenCLMemoryDelegate getDelegate();
	
	void notifyAfterWrite();
	
	void notifyBeforeRead();
	
	GenericGridOperatorInterface<T> getGridOperator();
	
	GenericGridOperatorInterface<T> selectGridOperator(GenericGrid<T> ... grids);
	
	void activateCL();
	
	void deactivateCL();

}
