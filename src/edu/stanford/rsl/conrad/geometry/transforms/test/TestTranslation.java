package edu.stanford.rsl.conrad.geometry.transforms.test;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.TestingTools;

/**
 * Class to test the accuracy of the {@link Translation} class
 * 
 * @see Translation
 * @author Rotimi X Ojo
 */
public class TestTranslation {
	
	@Test
	public void main(){
		System.out.println("Testing Tanslation class");
		SimpleVector translator = TestingTools.randVector(3);
		SimpleVector oldDir = TestingTools.randVector(3);
		Translation trans = new Translation(translator);
		//Transform direction
		SimpleVector newDir = trans.transform(oldDir);
		assert(SimpleOperators.equalElementWise(oldDir,newDir,0));
		
		//Transform point
		PointND oldPoint = new PointND(oldDir);
		PointND newP = trans.transform(oldPoint);
		assert(SimpleOperators.equalElementWise(trans.transform(oldPoint).getAbstractVector(),SimpleOperators.add(oldPoint.getAbstractVector(),translator),0));
		//Inverse Transform
		Translation invTrans = trans.inverse();
		//Invert new Direction
		assert(SimpleOperators.equalElementWise(invTrans.transform(newDir),oldDir,0));
		//Invert new Point	
		assert(SimpleOperators.equalElementWise(invTrans.transform(newP).getAbstractVector(),oldPoint.getAbstractVector(),0));	
		System.out.println("Testing Complete");	
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/