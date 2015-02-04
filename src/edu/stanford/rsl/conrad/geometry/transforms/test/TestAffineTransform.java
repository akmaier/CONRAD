/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.transforms.test;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.TestingTools;

/**
 * Class to test the accuracy of the {@link AffineTransform} class
 * 
 * @see AffineTransform
 * @author Rotimi X Ojo
 */
public class TestAffineTransform {

	@Test
	public void affineTest(){
		System.out.println("Testing Affine class");	
		SimpleVector translator = TestingTools.randVector(3);
		SimpleVector oldDir = TestingTools.randVector(3);
		SimpleMatrix randmatrix = TestingTools.randUpperTriangularMatrix(3, 3);
		AffineTransform aff = new AffineTransform(randmatrix, translator);		
		//Transform direction
		SimpleVector newDir = aff.transform(oldDir);
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(randmatrix, oldDir),0));
		//Transform Point
		PointND oldPoint = new PointND(oldDir);
		PointND newP = aff.transform(oldPoint);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.add(SimpleOperators.multiply(randmatrix, oldDir),translator),0));
		//Inverse Transform

		AffineTransform invaff = aff.inverse();
		//Invert new Direction
		Assert.assertTrue(SimpleOperators.equalElementWise(invaff.transform(newDir),oldDir,CONRAD.SMALL_VALUE));
		//Invert new Point
		Assert.assertTrue(SimpleOperators.equalElementWise(invaff.transform(newP).getAbstractVector(),oldPoint.getAbstractVector(),CONRAD.SMALL_VALUE));

	}

	@Test
	public void testMinMaxConstructor(){
		for (int i = 0; i < 1000; i++) {
			PointND beforePoint1 = new PointND(Math.random(), Math.random(), Math.random());
			PointND beforePoint2 = new PointND(Math.random(), Math.random(), Math.random());
			PointND afterPoint1 = new PointND(Math.random(), Math.random(), Math.random());
			PointND afterPoint2 = new PointND(Math.random(), Math.random(), Math.random());
			AffineTransform testTransform = new AffineTransform(beforePoint1, beforePoint2, afterPoint1, afterPoint2);
			PointND transformedPoint1 = testTransform.transform(beforePoint1);
			PointND transformedPoint2 = testTransform.transform(beforePoint2);

			Assert.assertTrue(transformedPoint1.euclideanDistance(afterPoint1) < CONRAD.SMALL_VALUE && transformedPoint2.euclideanDistance(afterPoint2) < CONRAD.SMALL_VALUE);
		}
	}
}
