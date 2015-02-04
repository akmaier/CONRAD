package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * @author jhchoi21
 *
 */

public class ViconAffineTransform{

	private SimpleMatrix scalingMatrix;
	private SimpleMatrix translationMatrix1;	
	private SimpleMatrix translationMatrix2;
	private SimpleMatrix translationMatrix3;
	private SimpleMatrix rotationAboutAxisMatrix;
	private SimpleMatrix rotationAboutOwnAxisMatrix;
	private SimpleMatrix affineTransformMatrix;
	//private SimpleMatrix rotationAboutAxisMatrixPatella;
	
	public ViconAffineTransform(){
		
	}
	
	public ViconAffineTransform(PointND[] beforePts, PointND[] afterPts) {
		
	}
	
	public ViconAffineTransform(PointND[] beforePts, PointND[] afterPts, double deg) {
		
		// shift starting vector to origin
		double[][] translationArray1 = 
				{{1, 0, 0, -beforePts[0].get(0)},				
				 {0, 1, 0, -beforePts[0].get(1)},
				 {0, 0, 1, -beforePts[0].get(2)},
				 {0, 0, 0, 1}};
	    translationMatrix1 = new SimpleMatrix(translationArray1);	    
		
		// rotate onto director of reference vector
		SimpleVector beforeVector = 	// XCAT
			new SimpleVector(beforePts[1].get(0)-beforePts[0].get(0), beforePts[1].get(1)-beforePts[0].get(1), beforePts[1].get(2)-beforePts[0].get(2));
		SimpleVector afterVector = 		// VICON
			new SimpleVector(afterPts[1].get(0)-afterPts[0].get(0), afterPts[1].get(1)-afterPts[0].get(1),afterPts[1].get(2)-afterPts[0].get(2));
		SimpleVector beforeVectorNormed = beforeVector.normalizedL2();
		SimpleVector afterVectorNormed = afterVector.normalizedL2();
		SimpleVector crossVector = General.crossProduct(beforeVectorNormed, afterVectorNormed);
		SimpleVector crossVectorNormed = crossVector.normalizedL2();		
		SimpleVector crossVectorCrossBeforeNormed = General.crossProduct(crossVectorNormed, beforeVectorNormed);
		SimpleVector crossVectorCrossAfterNormed = General.crossProduct(crossVectorNormed, afterVectorNormed);		
		
		double[][] subRotationArray1 = 
			{{beforeVectorNormed.getElement(0), beforeVectorNormed.getElement(1), beforeVectorNormed.getElement(2)},				
			 {crossVectorCrossBeforeNormed.getElement(0), crossVectorCrossBeforeNormed.getElement(1), crossVectorCrossBeforeNormed.getElement(2)},
			 {crossVectorNormed.getElement(0), crossVectorNormed.getElement(1), crossVectorNormed.getElement(2)}};
		SimpleMatrix subRotationMatrix1 = new SimpleMatrix(subRotationArray1);
		double[][] subRotationArray2 = 
			{{afterVectorNormed.getElement(0), crossVectorCrossAfterNormed.getElement(0), crossVectorNormed.getElement(0)},				
			 {afterVectorNormed.getElement(1), crossVectorCrossAfterNormed.getElement(1), crossVectorNormed.getElement(1)},
			 {afterVectorNormed.getElement(2), crossVectorCrossAfterNormed.getElement(2), crossVectorNormed.getElement(2)}};
		SimpleMatrix subRotationMatrix2 = new SimpleMatrix(subRotationArray2);
		SimpleMatrix rotationAboutAxisMatrixSub = SimpleOperators.multiplyMatrixProd(subRotationMatrix2, subRotationMatrix1);		
		rotationAboutAxisMatrix = new SimpleMatrix(4, 4);
		rotationAboutAxisMatrix.setElementValue(0, 3, 0);
		rotationAboutAxisMatrix.setElementValue(1, 3, 0);
		rotationAboutAxisMatrix.setElementValue(2, 3, 0);
		rotationAboutAxisMatrix.setElementValue(3, 3, 1);
		rotationAboutAxisMatrix.setElementValue(3, 0, 0);
		rotationAboutAxisMatrix.setElementValue(3, 1, 0);
		rotationAboutAxisMatrix.setElementValue(3, 2, 0);
		rotationAboutAxisMatrix.setSubMatrixValue(0, 0, rotationAboutAxisMatrixSub);
		
		// rotate about axis of two vectors
		//SimpleVector rotationAxisVector = General.crossProduct(beforeVector, afterVector);
		//double angleVectors = General.angle(beforeVector, afterVector);
		double angleVectors = General.toRadians(deg);
		SimpleMatrix rotationAboutOwnAxisMatrixSub = Rotations.createRotationMatrixAboutAxis(afterVectorNormed, angleVectors);
		rotationAboutOwnAxisMatrix = new SimpleMatrix(4, 4);
		rotationAboutOwnAxisMatrix.setElementValue(0, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(1, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(2, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 3, 1);
		rotationAboutOwnAxisMatrix.setElementValue(3, 0, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 1, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 2, 0);
		rotationAboutOwnAxisMatrix.setSubMatrixValue(0, 0, rotationAboutOwnAxisMatrixSub);	
		
		// scale to length of reference vector
		double scalingFactor = afterVector.normL2()/beforeVector.normL2();
		double[][] scalingArray = 
				{{scalingFactor , 0, 0, 0}, 
				{0, scalingFactor, 0, 0}, 
				{0, 0, scalingFactor, 0}, 
				{0, 0, 0, 1}};
		scalingMatrix = new SimpleMatrix(scalingArray);
		
		// shift to start point of reference vector
		double[][] translationArray2 = 
				{{1, 0, 0, afterPts[0].get(0)},				
				 {0, 1, 0, afterPts[0].get(1)},
				 {0, 0, 1, afterPts[0].get(2)},
				 {0, 0, 0, 1}};
	    translationMatrix2 = new SimpleMatrix(translationArray2);	 
	    
		affineTransformMatrix = new SimpleMatrix(4, 4);
	    //affineTransformMatrix = SimpleOperators.multiplyMatrixProd(translationMatrix2, SimpleOperators.multiplyMatrixProd(scalingMatrix, SimpleOperators.multiplyMatrixProd(rotationAboutAxisMatrix, translationMatrix1)));
		affineTransformMatrix = SimpleOperators.multiplyMatrixProd(translationMatrix2, SimpleOperators.multiplyMatrixProd(scalingMatrix, SimpleOperators.multiplyMatrixProd(rotationAboutOwnAxisMatrix, SimpleOperators.multiplyMatrixProd(rotationAboutAxisMatrix, translationMatrix1))));
	}
	
	public static void main(String[] args) {
	
		// read global configuration
		CONRAD.setup();
		
		PointND[] beforePts = new PointND[2];
		beforePts[0] = new PointND(273.8, 273.65,-649.57);
		beforePts[1] = new PointND(309.89, 287.18,-1033.07);

		// after affine transformation  
		PointND[] afterPts = new PointND[2];
		afterPts[0] = new PointND(-292.17, 27.22, 445.38);
		afterPts[1] = new PointND(-156.79, 41.466, 75.859);
		
//		PointND[] beforePts = new PointND[2];
//		beforePts[0] = new PointND(1, 1, 0);
//		beforePts[1] = new PointND(1, 2, 0);
//
//		// after affine transformation  
//		PointND[] afterPts = new PointND[2];
//		afterPts[0] = new PointND(2, 1, 5);
//		afterPts[1] = new PointND(4, 1, 0);

		ViconAffineTransform affineTrans = new ViconAffineTransform(beforePts, afterPts);
//		// test
//		PointND rlt = affineTrans.getTransformedPoints(beforePts[1]);
//		System.out.println(afterPts[1]);
//		System.out.println(rlt);		
		
	}	
	
	public ArrayList<PointND> getTransformedPoints(ArrayList<PointND> beforePts){
		
		ArrayList<PointND> afterPts = new ArrayList<PointND>();		
		SimpleVector beforeVector = new SimpleVector(0, 0, 0, 0);
		SimpleVector afterVector = new SimpleVector(0, 0, 0, 0);
		
		for (int i=0; i<beforePts.size(); i++){
						
			beforeVector.setElementValue(0, beforePts.get(i).get(0));
			beforeVector.setElementValue(1, beforePts.get(i).get(1));
			beforeVector.setElementValue(2, beforePts.get(i).get(2));
			beforeVector.setElementValue(3, 1);
					
			afterVector = SimpleOperators.multiply(affineTransformMatrix, beforeVector);			
			afterPts.add(i, new PointND(afterVector.getElement(0)/afterVector.getElement(3), afterVector.getElement(1)/afterVector.getElement(3), afterVector.getElement(2)/afterVector.getElement(3)));		
			
		}
				
		return afterPts;
		
	}	
	
	public PointND getTransformedPoints(PointND beforePts){
		
		PointND afterPts;		
		SimpleVector beforeVector = new SimpleVector(0, 0, 0, 0);
		SimpleVector afterVector = new SimpleVector(0, 0, 0, 0);
						
		beforeVector.setElementValue(0, beforePts.get(0));
		beforeVector.setElementValue(1, beforePts.get(1));
		beforeVector.setElementValue(2, beforePts.get(2));
		beforeVector.setElementValue(3, 1);
					
		afterVector = SimpleOperators.multiply(affineTransformMatrix, beforeVector);			
		afterPts = new PointND(afterVector.getElement(0)/afterVector.getElement(3), afterVector.getElement(1)/afterVector.getElement(3), afterVector.getElement(2)/afterVector.getElement(3));		
				
		return afterPts;
		
	}
	
	public SimpleMatrix getAffineTransformationMatrix (){
		return affineTransformMatrix;
	}
	
	public void setPatellaRotationMatrix(PointND KNE, PointND KJC, double deg){
				
		SimpleVector axisVector = new SimpleVector(KJC.get(0)-KNE.get(0), KJC.get(1)-KNE.get(1), KJC.get(2)-KNE.get(2));		
		SimpleVector axisVectorNormed = axisVector.normalizedL2();
		
		SimpleMatrix rotationAboutOwnAxisMatrix;
		double angle = General.toRadians(deg);
		SimpleMatrix rotationAboutOwnAxisMatrixSub = Rotations.createRotationMatrixAboutAxis(axisVectorNormed, angle);
		rotationAboutOwnAxisMatrix = new SimpleMatrix(4, 4);
		rotationAboutOwnAxisMatrix.setElementValue(0, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(1, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(2, 3, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 3, 1);
		rotationAboutOwnAxisMatrix.setElementValue(3, 0, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 1, 0);
		rotationAboutOwnAxisMatrix.setElementValue(3, 2, 0);
		rotationAboutOwnAxisMatrix.setSubMatrixValue(0, 0, rotationAboutOwnAxisMatrixSub);	
		
		affineTransformMatrix = SimpleOperators.multiplyMatrixProd(rotationAboutOwnAxisMatrix, affineTransformMatrix);
	}
	
	public SimpleMatrix getScalingMatrix() {
		return scalingMatrix;
	}
}

/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/