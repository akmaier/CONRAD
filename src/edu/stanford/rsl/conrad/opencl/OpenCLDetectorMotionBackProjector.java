package edu.stanford.rsl.conrad.opencl;

import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.XmlUtils;

public class OpenCLDetectorMotionBackProjector extends OpenCLBackProjector{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7764566238080291654L;
	
	AffineTransform[] affDetMotion;

	public OpenCLDetectorMotionBackProjector () {
		super();
		affDetMotion = null;
	}

	@Override
	public void configure() throws Exception{
		super.configure();
		affDetMotion = (AffineTransform[])XmlUtils.importFromXML();
		if (affDetMotion==null || affDetMotion.length != Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize())
			throw new Exception("Motion fields exceed the number of projections");
	}
	
	public void configure(AffineTransform[] transf) throws Exception{
		super.configure();
		if (transf==null || transf.length != Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize()){
			configured = false;
			throw new Exception("Motion fields exceed the number of projections");
		}
		this.affDetMotion = transf;
	}

	@Override
	protected synchronized void initProjectionMatrix(int projectionNumber){
		// load projection Matrix for current Projection.
		// if detector motion field exists, apply to projection matrix to do correction
		SimpleMatrix pMat;
		if (affDetMotion== null || affDetMotion.length <= projectionNumber)
			pMat = getGeometry().getProjectionMatrix(projectionNumber).computeP();
		else{
			SimpleMatrix affMotionH = new SimpleMatrix(3,3);
			affMotionH.setSubMatrixValue(0, 0, affDetMotion[projectionNumber].getRotation(2));
			affMotionH.setSubColValue(0, 2, affDetMotion[projectionNumber].getTranslation(2));
			affMotionH.setElementValue(2,2,1);
			pMat = SimpleOperators.multiplyMatrixProd(affMotionH, getGeometry().getProjectionMatrix(projectionNumber).computeP());
		}
		
		float [] pMatFloat = new float[pMat.getCols() * pMat.getRows()];
		for (int j = 0; j< pMat.getRows(); j++) {
			for (int i = 0; i< pMat.getCols(); i++) {

				pMatFloat[(j * pMat.getCols()) + i] = (float) pMat.getElement(j, i);
			}
		}

		// Obtain the global pointer to the view matrix from
		// the module
		if (projectionMatrix == null)
			projectionMatrix = context.createFloatBuffer(pMatFloat.length, Mem.READ_ONLY);

		projectionMatrix.getBuffer().put(pMatFloat);
		projectionMatrix.getBuffer().rewind();
		commandQueue.putWriteBuffer(projectionMatrix, true).finish();
	}


	@Override
	public String getName() {
		return "OpenCL Backprojector With Detector Motion";
	}

	@Override
	public String getToolName(){
		return "OpenCL Backprojector With Detector Motion";
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/