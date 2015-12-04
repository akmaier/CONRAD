/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.opencl.delegates.OpenCLNumericMemoryDelegateLinear;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLGrid2D extends Grid2D implements OpenCLGridInterface {
	
	OpenCLMemoryDelegate delegate;
	
	// ****************************************************************************************	
	// ************************ Constructors and copying  *************************************
	// ****************************************************************************************
	
	public OpenCLGrid2D(Grid2D input, CLContext context, CLDevice device){
		super(input);
		this.initializeDelegate(context, device);
		this.numericGridOperator = OpenCLGridOperators.getInstance();
	}

	
	public OpenCLGrid2D(Grid2D input) {
		this(input, OpenCLUtil.getStaticContext(), OpenCLUtil.getStaticContext().getMaxFlopsDevice());
	}
	
	
	@Override
	public OpenCLGrid2D clone(){
		notifyBeforeRead();
		return new OpenCLGrid2D(this, delegate.getCLContext(), delegate.getCLDevice());
	}
	
	// ****************************************************************************************	
	// ************************ Overriding superclass methods *********************************
	// ****************************************************************************************
	
	@Override
	public void notifyBeforeRead() {
		getDelegate().prepareForHostOperation();
	}
	
	@Override
	public void notifyAfterWrite() {
		getDelegate().notifyHostChange();
	}
	
	@Override
	public NumericGridOperator getGridOperator() {
		return this.numericGridOperator;
	}
	
	
	// ****************************************************************************************	
	// ************************ Implementing interface methods *********************************
	// ****************************************************************************************
	
	@Override
	public void initializeDelegate(CLContext context, CLDevice device) {
		if(buffer==null) throw new NullPointerException("Host buffer needs to be initialized before the OpenCL delegate can be created");
		delegate = new OpenCLNumericMemoryDelegateLinear(buffer, context, device);
	}
	
	@Override
	public OpenCLMemoryDelegate getDelegate() {
		return delegate;
	}
	
	@Override
	public void release() {
		getDelegate().release();
	}
}
