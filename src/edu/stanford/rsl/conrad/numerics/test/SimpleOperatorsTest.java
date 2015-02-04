package edu.stanford.rsl.conrad.numerics.test;

import static edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;

import static org.junit.Assert.*;

//import org.junit.After;
//import org.junit.AfterClass;
//import org.junit.Before;
//import org.junit.BeforeClass;
import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.TestingTools;


public class SimpleOperatorsTest {

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
	public void testInverseQR() {
		// create a random matrix
		final int N = TestingTools.rand(1, 8);
		SimpleMatrix M = new SimpleMatrix(N, N);
		M.randomize(-5.0, 5.0);
		
		// test inversion
		SimpleMatrix Minv = M.inverse(InversionType.INVERT_QR);
		SimpleMatrix I1 = SimpleOperators.multiplyMatrixProd(M, Minv);
		SimpleMatrix I2 = SimpleOperators.multiplyMatrixProd(Minv, M);
		assertTrue(I1.isIdentity(TestingTools.DELTA));
		assertTrue(I2.isIdentity(TestingTools.DELTA));
	}

	@Test
	public void testInverseUpperTriangular() {
		// create a random upper triangular matrix
		final int N = TestingTools.rand(1, 8);
		SimpleMatrix M = new SimpleMatrix(N, N);
		M.randomize(-5.0, 5.0);
		for (int row = 1; row < N; ++row)
			for (int col = 0; col < row; ++col)
				M.setElementValue(row, col, 0.0);
		
		// test inversion
		SimpleMatrix Minv = M.inverse(InversionType.INVERT_UPPER_TRIANGULAR);
		SimpleMatrix I1 = SimpleOperators.multiplyMatrixProd(M, Minv);
		SimpleMatrix I2 = SimpleOperators.multiplyMatrixProd(Minv, M);
		assertTrue(I1.isIdentity(TestingTools.DELTA));
		assertTrue(I2.isIdentity(TestingTools.DELTA));
	}

	@Test
	public void testInverseRt2D() {
		SimpleMatrix R = TestingTools.randRotationMatrix2D();
		SimpleVector t = new SimpleVector(2);
		t.randomize(-10.0, 10.0);
		SimpleMatrix Rt = General.createHomAffineMotionMatrix(R, t);
		SimpleMatrix Rtinv = Rt.inverse(InversionType.INVERT_RT);
		SimpleMatrix I1 = SimpleOperators.multiplyMatrixProd(Rt, Rtinv);
		SimpleMatrix I2 = SimpleOperators.multiplyMatrixProd(Rtinv, Rt);
		assertTrue(I1.isIdentity(TestingTools.DELTA));
		assertTrue(I2.isIdentity(TestingTools.DELTA));
	}

	@Test
	public void testInverseRt3D() {
		SimpleMatrix R = TestingTools.randRotationMatrix3D();
		SimpleVector t = new SimpleVector(3);
		t.randomize(-10.0, 10.0);
		SimpleMatrix Rt = General.createHomAffineMotionMatrix(R, t);
		SimpleMatrix Rtinv = Rt.inverse(InversionType.INVERT_RT);
		SimpleMatrix I1 = SimpleOperators.multiplyMatrixProd(Rt, Rtinv);
		SimpleMatrix I2 = SimpleOperators.multiplyMatrixProd(Rtinv, Rt);
		assertTrue(I1.isIdentity(TestingTools.DELTA));
		assertTrue(I2.isIdentity(TestingTools.DELTA));
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/