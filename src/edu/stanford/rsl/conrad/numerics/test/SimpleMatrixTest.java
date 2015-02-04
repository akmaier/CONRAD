package edu.stanford.rsl.conrad.numerics.test;

import static org.junit.Assert.*;

//import org.junit.After;
//import org.junit.AfterClass;
//import org.junit.Before;
//import org.junit.BeforeClass;

import org.junit.Test;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.TestingTools;



public class SimpleMatrixTest {

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
	public void testGetSubMatrix() {
		final int m = TestingTools.rand(2, 10);
		final int n = TestingTools.rand(2, 10);
		SimpleMatrix M = new SimpleMatrix(m, n);
		M.randomize(-1.0, 1.0);
		int selectRows[] = new int[m-1];
		int selectCols[] = new int[n-1];
		int deleteRow;
		int deleteCol;
		
		// test removing first row and first column
		for (int i = 0; i < m-1; ++i) selectRows[i] = i+1;
		for (int i = 0; i < n-1; ++i) selectCols[i] = i+1;
		deleteRow = 0;
		deleteCol = 0;
		SimpleMatrix Sff_rect = M.getSubMatrix(1, 1, m-1, n-1);
		SimpleMatrix Sff_sel = M.getSubMatrix(selectRows, selectCols);
		SimpleMatrix Sff_del = M.getSubMatrix(deleteRow, deleteCol);
		TestingTools.assertEqualElementWise(Sff_rect, Sff_sel, 0.0);
		TestingTools.assertEqualElementWise(Sff_sel, Sff_del, 0.0);
		TestingTools.assertEqualElementWise(Sff_del, Sff_rect, 0.0);

		// test removing last row and last column
		for (int i = 0; i < m-1; ++i) selectRows[i] = i;
		for (int i = 0; i < n-1; ++i) selectCols[i] = i;
		deleteRow = m-1;
		deleteCol = n-1;
		SimpleMatrix Sll_rect = M.getSubMatrix(0, 0, m-1, n-1);
		SimpleMatrix Sll_sel = M.getSubMatrix(selectRows, selectCols);
		SimpleMatrix Sll_del = M.getSubMatrix(deleteRow, deleteCol);
		TestingTools.assertEqualElementWise(Sll_rect, Sll_sel, 0.0);
		TestingTools.assertEqualElementWise(Sll_sel, Sll_del, 0.0);
		TestingTools.assertEqualElementWise(Sll_del, Sll_rect, 0.0);
}

	@Test
	public void testDeterminant() {
		// test 1x1 matrix
		SimpleMatrix M1 = new SimpleMatrix(1, 1);
		M1.randomize(-1.0, 1.0);
		assertEquals(M1.getElement(0, 0), M1.determinant(), TestingTools.DELTA);
		
		// test 2x2 matrix
		SimpleMatrix M2 = new SimpleMatrix(2, 2);
		M2.randomize(-1.0, 1.0);
		assertEquals(M2.getElement(0, 0)*M2.getElement(1, 1) - M2.getElement(0, 1)*M2.getElement(1, 0), M2.determinant(), TestingTools.DELTA);
		
		// test 3x3 matrix
		SimpleMatrix M3 = new SimpleMatrix(3, 3);
		M3.randomize(-1.0, 1.0);
		double det = 0.0;
		det += M3.getElement(0, 0)*M3.getSubMatrix(0, 0).determinant();
		det -= M3.getElement(0, 1)*M3.getSubMatrix(0, 1).determinant();
		det += M3.getElement(0, 2)*M3.getSubMatrix(0, 2).determinant();
		assertEquals(det, M3.determinant(), TestingTools.DELTA);
		
		// test NxN matrix
		final int N = TestingTools.rand(2, 10);
		SimpleMatrix MN = new SimpleMatrix(N, N);
		MN.randomize(-1.0, 1.0);
		det = 0.0;
		double factor = 1.0;
		for (int i = 0; i < MN.getRows(); ++i) {
			det += factor * MN.getElement(i, 0) * MN.getSubMatrix(i, 0).determinant();
			factor *= -1.0;
		}
		assertEquals(det, MN.determinant(), TestingTools.DELTA);
	}

	@Test
	public void testToStringAndFromString() {
		int rows = TestingTools.rand(1, 5);
		int cols = TestingTools.rand(1, 5);
		SimpleMatrix M_in = new SimpleMatrix(rows, cols);
		M_in.randomize(-1.0, 10.0);
		SimpleMatrix M_out = new SimpleMatrix(M_in.toString());
		TestingTools.assertEqualElementWise(M_in, M_out, TestingTools.DELTA);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/