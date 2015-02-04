package edu.stanford.rsl.conrad.numerics.test;

import org.junit.Test;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.utils.TestingTools;

//import org.junit.After;
//import org.junit.AfterClass;
//import org.junit.Before;
//import org.junit.BeforeClass;


public class SolversTest {

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
	public void testSolveUpperTriangular() {
		int n = TestingTools.rand(1, 15);
		SimpleMatrix U = TestingTools.randUpperTriangularMatrix(n, n);
		SimpleVector b = new SimpleVector(n);
		b.randomize(-1.0, 1.0);
		SimpleVector x = Solvers.solveUpperTriangular(U, b);
		SimpleVector Ax = SimpleOperators.multiply(U, x);
		TestingTools.assertEqualElementWise(Ax, b, TestingTools.DELTA);
	}

	@Test
	public void testSolveLowerTriangular() {
		int n = TestingTools.rand(1, 15);
		SimpleMatrix L = TestingTools.randLowerTriangularMatrix(n, n);
		SimpleVector b = new SimpleVector(n);
		b.randomize(-1.0, 1.0);
		SimpleVector x = Solvers.solveLowerTriangular(L, b);
		SimpleVector Ax = SimpleOperators.multiply(L, x);
		TestingTools.assertEqualElementWise(Ax, b, TestingTools.DELTA);
	}

	@Test
	public void testSolveLSE() {
		int n = TestingTools.rand(2, 10);
		SimpleMatrix A = TestingTools.randMatrixNonSingular(n);
		SimpleVector b = TestingTools.randVector(n);
		SimpleVector x = Solvers.solveLinearSysytemOfEquations(A, b);
		SimpleVector Ax = SimpleOperators.multiply(A, x);
		TestingTools.assertEqualElementWise(Ax, b, TestingTools.DELTA);
	}

	@Test
	public void testSolveLinearLeastSquares() {
		int m = TestingTools.rand(2, 15);
		int n = TestingTools.rand(2, m-1);
		SimpleMatrix A = TestingTools.randMatrix(m, n);
		SimpleVector b = TestingTools.randVector(m);
		SimpleVector x = Solvers.solveLinearLeastSquares(A, b);
		SimpleVector AtAx = SimpleOperators.multiply(A.transposed(), SimpleOperators.multiply(A, x));
		SimpleVector Atb = SimpleOperators.multiply(A.transposed(), b);
		TestingTools.assertEqualElementWise(AtAx, Atb, TestingTools.DELTA);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/