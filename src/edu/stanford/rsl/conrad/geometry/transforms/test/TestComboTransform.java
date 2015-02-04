package edu.stanford.rsl.conrad.geometry.transforms.test;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.TestingTools;


/**
 * Class to test the accuracy of the {@link ComboTransform} class
 * 
 * @see ComboTransform
 * @author Rotimi X Ojo
 */
public class TestComboTransform {

	@Test
	public void RunTest4() {
		System.out.println("\tRunning Test 4");
		SimpleVector oldDir = TestingTools.randVector(3);
		SimpleVector translator= TestingTools.randVector(3);
		ScaleRotate r1 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		SimpleMatrix afMat = TestingTools.randUpperTriangularMatrix(3, 3);
		AffineTransform r2 = new AffineTransform(afMat, translator);
		Translation r3 = new Translation(translator);
		
		ComboTransform scr = new ComboTransform(r3, r1, r2);
		//Transform direction
		SimpleVector newDir = scr.transform(oldDir);;
		
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(afMat,SimpleOperators.multiply(r1.getData(),oldDir)),0));
		//Transform Point
		PointND oldP = new PointND(oldDir);
		PointND newP = scr.transform(oldP);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.add(translator,SimpleOperators.multiply(afMat,SimpleOperators.multiply(r1.getData(),SimpleOperators.add(r3.getData(),oldDir)))),0));		
		//Inverse Transform
		ComboTransform inscr = (ComboTransform) scr.inverse();
		//Invert new Direction		
		assert(SimpleOperators.equalElementWise(inscr.transform(newDir),oldDir,CONRAD.SMALL_VALUE*3));
		//Invert new Point
		assert(SimpleOperators.equalElementWise(inscr.transform(newP).getAbstractVector(),oldP.getAbstractVector(),CONRAD.SMALL_VALUE*3));	
		
	}

	@Test
	public void Runtest3() {
		System.out.println("\tRunning Test 3");
		SimpleVector oldDir = TestingTools.randVector(3);
		SimpleVector translator= TestingTools.randVector(3);
		ScaleRotate r1 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		SimpleMatrix afMat = TestingTools.randUpperTriangularMatrix(3, 3);
		AffineTransform r2 = new AffineTransform(afMat, translator);
		Translation r3 = new Translation(translator);
		
		ComboTransform scr = new ComboTransform(r1, r3, r2);
		//Transform direction
		SimpleVector newDir = scr.transform(oldDir);;
		
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(afMat,SimpleOperators.multiply(r1.getData(),oldDir)),0));
		//Transform Point
		PointND oldP = new PointND(oldDir);
		PointND newP = scr.transform(oldP);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.add(translator,SimpleOperators.multiply(afMat,SimpleOperators.add(r3.getData(),SimpleOperators.multiply(r1.getData(),oldDir)))),0));		
		//Inverse Transform
		ComboTransform inscr = (ComboTransform) scr.inverse();
		//Invert new Direction		
		assert(SimpleOperators.equalElementWise(inscr.transform(newDir),oldDir,CONRAD.SMALL_VALUE*3));
		//Invert new Point
		assert(SimpleOperators.equalElementWise(inscr.transform(newP).getAbstractVector(),oldP.getAbstractVector(),CONRAD.SMALL_VALUE*3));		
	}

	@Test
	public void RunTest2() {
		System.out.println("\tRunning Test 2");
		SimpleVector oldDir = TestingTools.randVector(3);
		SimpleVector translator= TestingTools.randVector(3);
		ScaleRotate r1 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		SimpleMatrix afMat = TestingTools.randUpperTriangularMatrix(3, 3);
		AffineTransform r2 = new AffineTransform(afMat, translator);
		Translation r3 = new Translation(translator);
		
		ComboTransform scr = new ComboTransform(r1, r2, r3);
		//Transform direction
		SimpleVector newDir = scr.transform(oldDir);;
		
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(afMat,SimpleOperators.multiply(r1.getData(),oldDir)),0));
		//Transform Point
		PointND oldP = new PointND(oldDir);
		PointND newP = scr.transform(oldP);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.add(r3.getData(),SimpleOperators.add(translator,SimpleOperators.multiply(afMat,SimpleOperators.multiply(r1.getData(),oldDir)))),0));		
		//Inverse Transform
		ComboTransform inscr = (ComboTransform) scr.inverse();
		//Invert new Direction		
		assert(SimpleOperators.equalElementWise(inscr.transform(newDir),oldDir,CONRAD.SMALL_VALUE*3));
		//Invert new Point
		assert(SimpleOperators.equalElementWise(inscr.transform(newP).getAbstractVector(),oldP.getAbstractVector(),CONRAD.SMALL_VALUE*3));
		
	}

	@Test
	public void RunTest1() {
		System.out.println("\tRunning Test 1");
		SimpleVector oldDir = TestingTools.randVector(3);
		ScaleRotate r1 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		ScaleRotate r2 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		ScaleRotate r3 = new ScaleRotate(TestingTools.randUpperTriangularMatrix(3, 3));
		
		ComboTransform scr = new ComboTransform(r1, r2, r3);
		//Transform direction
		SimpleVector newDir = scr.transform(oldDir);;
		assert(SimpleOperators.equalElementWise(newDir,SimpleOperators.multiply(r3.getData(),SimpleOperators.multiply(r2.getData(),SimpleOperators.multiply(r1.getData(), oldDir))),0));
		//Transform Point
		PointND oldP = new PointND(oldDir);
		PointND newP = scr.transform(oldP);
		assert(SimpleOperators.equalElementWise(newP.getAbstractVector(),SimpleOperators.multiply(r3.getData(),SimpleOperators.multiply(r2.getData(),SimpleOperators.multiply(r1.getData(), oldDir))),0));
		//Inverse Transform
		ComboTransform inscr = (ComboTransform) scr.inverse();
		//Invert new Direction		
		assert(SimpleOperators.equalElementWise(inscr.transform(newDir),oldDir,CONRAD.SMALL_VALUE*3));
		//Invert new Point
		assert(SimpleOperators.equalElementWise(inscr.transform(newP).getAbstractVector(),oldP.getAbstractVector(),CONRAD.SMALL_VALUE*3));
		
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/