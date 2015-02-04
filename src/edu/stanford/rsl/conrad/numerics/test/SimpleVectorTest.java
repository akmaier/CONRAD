package edu.stanford.rsl.conrad.numerics.test;


//import org.junit.After;
//import org.junit.AfterClass;
//import org.junit.Before;
//import org.junit.BeforeClass;

import org.junit.Test;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.TestingTools;



public class SimpleVectorTest {

//	@BeforeClass
//	public static void setUpBeforeClass() throws Exception {
//	}
//
//	@AfterClass
//	public static void tearDownAfterClass() throws Exception {
//	}
//
//	@Before
//	public void setUp() throws Exception {
//	}
//
//	@After
//	public void tearDown() throws Exception {
//	}
	
	@Test
	public void testToStringAndFromString() {
		int len = TestingTools.rand(1, 5);
		SimpleVector v_in = new SimpleVector(len);
		v_in.randomize(-1.0, 10.0);
		SimpleVector v_out = new SimpleVector(v_in.toString());
		TestingTools.assertEqualElementWise(v_in, v_out, TestingTools.DELTA);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/