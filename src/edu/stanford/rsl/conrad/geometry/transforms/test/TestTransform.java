/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.transforms.test;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class TestTransform {

	
	/**
	 * A few tests to ensure that the extraction of the rotational part works
	 */
	@Test
	public void testRotation(){
		SimpleMatrix rot =Rotations.createRotationMatrixAboutAxis(new SimpleVector(Math.random(), Math.random(), Math.random()), Math.random()*Math.PI);
		
		ScaleRotate scaleRotate = new ScaleRotate(rot);
		SimpleMatrix test1 = scaleRotate.getRotation(3);
		test1.subtract(rot);
		double error = test1.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
		Translation translations = new Translation(Math.random(), Math.random(), Math.random());
		
		ComboTransform comboTransform = new ComboTransform(scaleRotate, translations);
		test1 = comboTransform.getRotation(3);
		test1.subtract(rot);
		error = test1.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
		comboTransform = new ComboTransform(translations, scaleRotate, translations);
		test1 = comboTransform.getRotation(3);
		test1.subtract(rot);
		error = test1.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
		SimpleMatrix twiceMatrix = SimpleOperators.multiplyMatrixProd(rot, rot);
		
		comboTransform = new ComboTransform(translations, scaleRotate, translations, scaleRotate);
		test1 = comboTransform.getRotation(3);
		test1.subtract(twiceMatrix);
		error = test1.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
	}
	
	/**
	 * A few tests to ensure that the extraction of the translational part works
	 */
	@Test
	public void testTranslation(){
		SimpleMatrix rot =Rotations.createRotationMatrixAboutAxis(new SimpleVector(Math.random(), Math.random(), Math.random()), Math.random()*Math.PI);
		
		ScaleRotate scaleRotate = new ScaleRotate(rot);
		SimpleVector test1 = scaleRotate.getTranslation(3);
		test1.subtract(new SimpleVector(0,0,0));
		double error = test1.normL2();
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
		Translation translations = new Translation(Math.random(), Math.random(), Math.random());
		
		ComboTransform comboTransform = new ComboTransform(scaleRotate, translations);
		test1 = comboTransform.getTranslation(3);
		test1.subtract(translations.getData());
		error = test1.normL2();
		Assert.assertTrue(Math.abs(error) < CONRAD.DOUBLE_EPSILON);
		
		SimpleVector translation = null;
		
		comboTransform = new ComboTransform(translations, scaleRotate, translations);
		test1 = comboTransform.getTranslation(3);
		PointND point = new PointND(Math.random(),Math.random(),Math.random());
		translation = comboTransform.transform(point).getAbstractVector();
		translation.subtract(SimpleOperators.multiply(rot, point.getAbstractVector()));
		test1.subtract(translation);
		error = test1.normL2();
		Assert.assertTrue(Math.abs(error) < CONRAD.FLOAT_EPSILON);
		
		comboTransform = new ComboTransform(translations, scaleRotate, translations, scaleRotate);
		test1 = comboTransform.getTranslation(3);
		point = new PointND(Math.random(),Math.random(),Math.random());
		translation = comboTransform.transform(point).getAbstractVector();
		translation.subtract(SimpleOperators.multiply(comboTransform.getRotation(3), point.getAbstractVector()));
		test1.subtract(translation);
		error = test1.normL2();
		Assert.assertTrue(Math.abs(error) < CONRAD.FLOAT_EPSILON);
		
	}
	
}
