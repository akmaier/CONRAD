package edu.stanford.rsl.conrad.opencl;

import java.util.ArrayList;

import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;

//TODO: Use our own matrices instead of Jama.Matrix




/**
 * Forward projection expects input of a volumetric phantom scaled to mass density. Projection result {@latex.inline $p(\\vec{x})$} is then the accumulated mass along the ray {@latex.inline $\\vec{x}$} which consists of the line segments {@latex.inline $x_i$} in {@latex.inline $[\\textnormal{cm}]$} with the mass densities {@latex.inline $\\mu_i$} in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^3}]$}.
 * The actual projection is then computed as:<br>
 * {@latex.inline $$p(\\vec{x}) = \\sum_{i} x_i \\cdot \\mu_i$$}<BR>
 * The projection values are then returned in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^2}]$}
 * @author akmaier, berger (refactored from CUDA)
 *
 */
public class OpenCLForwardProjectorWithMotion extends OpenCLForwardProjector implements GUIConfigurable, Citeable {

	/**
	 * The XML filename where the rigid motion parameters are stored
	 */
	private String rotationTranslationFilename = null;
	
	@SuppressWarnings("unchecked")
	public SimpleMatrix[] readInMotionMatrices() {

		SimpleMatrix[] motion = new SimpleMatrix[geometry.getNumProjectionMatrices()];
		// load data from XML file
		ArrayList<double[][][]> RotTrans = null;
		try {
			if (rotationTranslationFilename == null)
				rotationTranslationFilename = FileUtil.myFileChoose(".xml", false);
			
			RotTrans = (ArrayList<double[][][]>)XmlUtils.importFromXML(rotationTranslationFilename);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for (int i = 0; i < geometry.getNumProjectionMatrices(); i++) {
			motion[i] = new SimpleMatrix(4,4);
			SimpleMatrix rotation = new SimpleMatrix(RotTrans.get(0)[i]);
			SimpleMatrix translation = new SimpleMatrix(RotTrans.get(1)[i]);
			motion[i].setSubMatrixValue(0, 0, rotation.transposed());
			motion[i].setSubMatrixValue(0, 3, translation.multipliedBy(-1));
			motion[i].setRowValue(3, new SimpleVector(0,0,0,1));
		}
		
		return motion;
		
	}
	
	
	/**
	 * First incorporate the rigid object motion into the projection matrices, then 
	 * load the inverted projection matrices for all projections and reset the projection data.
	 * @param projectionNumber
	 */
	@Override
	protected void prepareAllProjections(){
		float [] cann = new float[3*4];
		float [] invAR = new float[3*3];
		float [] srcP = new float[3];
		
		if (gInvARmatrix == null)
			gInvARmatrix = context.createFloatBuffer(invAR.length*geometry.getNumProjectionMatrices(), Mem.READ_ONLY);
		if (gSrcPoint == null)
			gSrcPoint = context.createFloatBuffer(srcP.length*geometry.getNumProjectionMatrices(), Mem.READ_ONLY);
		
		// load the motion transforms
		// load data from XML file
		SimpleMatrix[] motion = readInMotionMatrices();
		
		
		for (int i=0; i < geometry.getNumProjectionMatrices(); ++i){
			SimpleMatrix projMat = geometry.getProjectionMatrix(i).computeP();
			double [][] mat = new double [3][4];
			SimpleOperators.multiplyMatrixProd(projMat, motion[i]).copyTo(mat);
			computeCanonicalProjectionMatrix(cann, invAR, srcP, new Jama.Matrix(mat));
			
			gInvARmatrix.getBuffer().put(invAR);
			gSrcPoint.getBuffer().put(srcP);
		}
		
		gInvARmatrix.getBuffer().rewind();
		gSrcPoint.getBuffer().rewind();
		
		commandQueue
		.putWriteBuffer(gSrcPoint, true)
		.putWriteBuffer(gInvARmatrix, true)
		.finish();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Martin Berger, Marco B�gel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/