/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;

public interface OpenCLGridInterface {

	/*
	 *  Getter for the delegate
	 */
	OpenCLMemoryDelegate getDelegate();
	
	/*
	 *  initialization of delegate (is Grid dependent, thus leave to the grid)
	 */
	public void initializeDelegate(CLContext context, CLDevice device);
	
	/*
	 *  release the grids GPU resources
	 */
	public void release();
	
}