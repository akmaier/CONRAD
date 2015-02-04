package edu.stanford.rsl.conrad.numerics.test;

import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.numerics.DecompositionRQ;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class RQTest {

	@Test
	public void testRQ(){
		SimpleMatrix A = new SimpleMatrix("[[1, 2, 3]; [4, 5, 6]; [7, 8, 9]]");
		DecompositionRQ rq = new DecompositionRQ(A);
		SimpleMatrix test = SimpleOperators.multiplyMatrixProd(rq.getR(), rq.getQ());
		test.add(A.negated());
		Assert.assertEquals(true, test.norm(MatrixNormType.MAT_NORM_FROBENIUS) < CONRAD.FLOAT_EPSILON);
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/