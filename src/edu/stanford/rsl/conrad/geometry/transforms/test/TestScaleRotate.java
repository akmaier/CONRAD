package edu.stanford.rsl.conrad.geometry.transforms.test;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.TestingTools;

/**
 * Class to test the accuracy of the {@link ScaleRotate} class
 * 
 * @see ScaleRotate
 * @author Rotimi X Ojo
 */
public class TestScaleRotate {

	@Test
	public void main() {
		System.out.println("Testing ScaleRotate class");	

		SimpleVector oldDir = TestingTools.randVector(3);
		SimpleMatrix randmatrix = TestingTools.randUpperTriangularMatrix(3, 3);
		ScaleRotate scr = new ScaleRotate(randmatrix);
		//Transform direction
		SimpleVector newDir = scr.transform(oldDir);;
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(randmatrix, oldDir),0));
		//Transform Point
		PointND oldP = new PointND(oldDir);
		PointND newP = scr.transform(oldP);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.multiply(randmatrix, oldDir),0));
		//Inverse Transform
		ScaleRotate inscr = (ScaleRotate) scr.inverse();
		//Invert new Direction		
		assert(SimpleOperators.equalElementWise(inscr.transform(newDir),oldDir,CONRAD.SMALL_VALUE));
		//Invert new Point
		assert(SimpleOperators.equalElementWise(inscr.transform(newP).getAbstractVector(),oldP.getAbstractVector(),CONRAD.SMALL_VALUE));
		
		
		System.out.println("Testing Complete");
		
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/