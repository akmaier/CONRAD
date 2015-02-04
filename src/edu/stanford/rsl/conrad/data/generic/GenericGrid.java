/*
 * Copyright (C) 2010-2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic;

import java.util.Iterator;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.Grid;
import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;
import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;
import edu.stanford.rsl.conrad.data.generic.iterators.GenericPointwiseIteratorND;
import edu.stanford.rsl.conrad.data.generic.opencl.OpenCLGenericGridInterface;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;



/**
 *
 */
public abstract class GenericGrid<T extends Gridable<T>> extends Grid implements Iterable<T>, OpenCLGenericGridInterface<T>{
	
	protected OpenCLMemoryDelegate delegate;
	
	protected boolean openCLactive;
	
	/**
	 * @return Returns the object at position idx
	 */
	public abstract T getValue(int... idx);
	
	/**
	 * @return Sets the object at position idx
	 */
	public abstract void setValue(T val, int... idx);
	
	@Override
	public Iterator<T> iterator() {
		return new GenericPointwiseIteratorND<T>(this);
	}
	
	@Override
	public abstract GenericGridOperatorInterface<T> getGridOperator();
	
	
	@Override
	public GenericGridOperatorInterface<T> selectGridOperator(GenericGrid<T> ... grids){
		boolean useCL = true;
		for (GenericGrid<T> grid : grids){
			if (!grid.openCLactive){
				useCL = false;
			}
		}
		return selectGridOperator(useCL);
	};
	
	public abstract GenericGridOperatorInterface<T> selectGridOperator(boolean useOpenCLOperator);
	
	@Override
	public OpenCLMemoryDelegate getDelegate() {
		return delegate;
	}
	
	@Override
	public void notifyAfterWrite(){
		if(openCLactive) delegate.notifyHostChange();
	}
	
	@Override
	public void notifyBeforeRead(){
		if(openCLactive) delegate.prepareForHostOperation();
	}
	
	@Override
	public void activateCL(){
		initializeDelegate();
		openCLactive = true;
	}
	
	@Override
	public void deactivateCL(){
		notifyBeforeRead();
		if (delegate != null){
			delegate.release();
			delegate = null;
		}
		openCLactive = false;
	}
	
	public  abstract void initializeDelegate(CLContext context, CLDevice device);
	
	public  void initializeDelegate(){
		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();
		initializeDelegate(context,device);
	}
	
}


