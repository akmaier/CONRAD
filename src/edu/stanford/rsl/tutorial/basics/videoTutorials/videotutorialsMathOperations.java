package edu.stanford.rsl.tutorial.basics.videoTutorials;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;

public class videotutorialsMathOperations {

	public static void main(String[] args) {
		
		//create a zero vector
		SimpleVector v1 = new SimpleVector();
		
		//initialise the zero vector (length 3, entry values zero) + print v1
		v1.init(3);
		System.out.println("v1 = " + v1);
		
		//create a vector from a list of doubles + print v2
		SimpleVector v2 = new SimpleVector(1.1, 2.2, 3.3);
		System.out.println("v2 = " + v2);
		
		//set all entries of v1 to 1 + print v1
		v1.ones();
		System.out.println("v1 = " + v1);
		
		//set the second entry of v1 to 3 + print this entry
		v1.setElementValue(1, 3);
		double val1 = v1.getElement(1);
		System.out.println("Second entry of v1: " + val1);
		
		
		//create a 2x3 matrix + print M
		SimpleMatrix M = new SimpleMatrix(2,3);
		System.out.println("M = " + M);
		
		//set all entries of M to 7 + print M
		M.fill(7);
		System.out.println("M = " + M);
		
		//set the value of M at indices (1,0) to 2
		M.setElementValue(1, 0, 2);
		
		//get and print the the value of M at indices (1,0)
		double val2 = M.getElement(1, 0);
		System.out.println("Value of M at indices (0,1): " + val2);
		
		//transpose M + print the transposed matrix
		SimpleMatrix Mtransposed = M.transposed();
		System.out.println("Mtransposed = " + Mtransposed);
		
		
		//add v1 and v2 + print the result
		SimpleVector vecsum = SimpleOperators.add(v1, v2);
		System.out.println("v1 + v2 =" + vecsum);
		
		//compute the inner product of v1 and v2 + print the result
		double inprod = SimpleOperators.multiplyInnerProd(v1, v2);
		System.out.println("v1 * v2 = " + inprod);
		
		//multiply M and v2 + print the result
		SimpleVector matvecprod = SimpleOperators.multiply(M, v2);
		System.out.println("M * v2 = " + matvecprod);
		
	}

}
