/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.ecc;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.*;


public class EpipolarConsistency {
	
	// center matrix //
	private SimpleMatrix CENTER;
	
	// projection matrix of a view //
	public SimpleMatrix P;
	
	// source position of a view //
	public SimpleVector C;
	
	// radon transformed image of view //
	public Grid2D radon;
	
	// further dimensions //
	public int projectionWidth;
	public int projectionHeight;
	public int radonHeight;
	public int radonWidth;
	public double projectionDiag;
	public double lineIncrement;
	public double angleIncrement;
	
	
	/**
	 * constructor to compute the metric for epipolar consistency for one view.
	 * To compute the line integrals between two views, it is neccessary to create two instances of this class
	 * and call the other methods on it
	 * @param projection: Grid2D representing the projection
	 * @param radon: Grid2D representing the radon transformed and derived image
	 *  derivation is done in t direction (distance on detector)
	 * @param projIndex: Projection that stands for the index number
	 *  of projection matrix stored in the xml file
	 */
	public EpipolarConsistency(Grid2D projection, Grid2D radon, Projection projIndex) {
		
		// Initialize center matrix //
		CENTER = new SimpleMatrix(3,4);
		CENTER.setDiagValue(new SimpleVector(1.0, 1.0, 1.0));
		
		// get data out of projection //
		this.projectionWidth = projection.getWidth();
		this.projectionHeight = projection.getHeight();
		
		// get data out of radon transformed image //
		this.radonWidth = radon.getWidth();
		this.radonHeight = radon.getHeight();
		this.projectionDiag = Math.sqrt(projectionWidth*projectionWidth + projectionHeight*projectionHeight);
		this.lineIncrement = radonWidth / projectionDiag;
		this.angleIncrement = radonHeight / Math.PI;

		// store radon transformed image //
		this.radon = radon;
		
		// get projection matrix P (3x4) //
		this.P = SimpleOperators.multiplyMatrixProd(projIndex.getK(), CENTER);
		this.P = SimpleOperators.multiplyMatrixProd(this.P, projIndex.getRt());
		
		// get source position C (nullspace of the projection) //
		DecompositionSVD decoP = new DecompositionSVD(this.P);
		this.C = decoP.getV().getCol(3);
		
		// normalize source vectors by last component //
		// it is important that the last component is positive to have a positive center
		// as it is defined in oriented projective geometry
		this.C = this.C.dividedBy(this.C.getElement(3));

	}
	
	
	/**
	 * method to calculate a mapping K from two source positions C0, C1 to a plane
	 * @param C0: first source position (first view) as SimpleVector (4 entries)
	 * @param C1: second source position (second view) as SimpleVector (4 entries)
	 * @return: SimpleMatrix representing the mapping K (size 4x3)
	 */
	public static SimpleMatrix createMappingToEpipolarPlane(SimpleVector C0, SimpleVector C1) {
		
		// compute Pluecker coordinates //
		double L01 = C0.getElement(0)*C1.getElement(1) - C0.getElement(1)*C1.getElement(0);
		double L02 = C0.getElement(0)*C1.getElement(2) - C0.getElement(2)*C1.getElement(0);
		double L03 = C0.getElement(0)*C1.getElement(3) - C0.getElement(3)*C1.getElement(0);
		double L12 = C0.getElement(1)*C1.getElement(2) - C0.getElement(2)*C1.getElement(1);
		double L13 = C0.getElement(1)*C1.getElement(3) - C0.getElement(3)*C1.getElement(1);
		double L23 = C0.getElement(2)*C1.getElement(3) - C0.getElement(3)*C1.getElement(2);
		
		// construct B (6x1) //
		SimpleVector B = new SimpleVector(L01, L02, L03, L12, L13, L23);
		
		// compute infinity point in direction of B //
		SimpleVector N = new SimpleVector(-L03, -L13, -L23, 0);
		
		// compute plane E0 containing B and X0=(0,0,0,1) //
		SimpleVector E0 = SimpleOperators.getPlueckerJoin(B, new SimpleVector(0, 0, 0, 1));
		
		// find othonormal basis from plane normals //
		// (vectors are of 3x1)		
		SimpleVector a2 = new SimpleVector(E0.getElement(0), E0.getElement(1), E0.getElement(2));
		SimpleVector a3 = new SimpleVector(N.getElement(0), N.getElement(1), N.getElement(2));
		// set vectors to unit length
		a2.normalizeL2();
		a3.normalizeL2();
				
		// calculate cross product to get the last basis vector //
		SimpleVector a1 = General.crossProduct(a2, a3).negated();
		// (a1 is already of unit length -> no normalization needed)
		
		// set up assembly matrix A (4x3) //
		SimpleMatrix A = new SimpleMatrix(4, 3);
		A.setSubColValue(0, 0, a1);
		A.setSubColValue(0, 1, a2);
		A.setSubColValue(0, 2, C0);
		
		// return mapping matrix K (4x3 = 4x4 * 4x3) //
		return SimpleOperators.multiplyMatrixProd(SimpleOperators.getPlueckerMatrixDual(B), A);
		
	}
	
	
	/**
	 * method to calculate alpha and t as indices from the radon transformed image
	 * inputs are the inverse projection image (as SimpleMatrix) and the
	 * epipolar plane E
	 * @param P_Inverse: inverse of projection image as SimpleMatrix (size 4x3)
	 * @param E: epipolar plane E as SimpleVector (4 entries)
	 * @return
	 */
	private double getValueByAlphaAndT(SimpleMatrix P_Inverse, SimpleVector E) {
		
		// compute corresponding epipolar lines //
		// (epipolar lines are of 3x1 = 3x4 * 4x1)
		SimpleVector l_kappa = SimpleOperators.multiply(P_Inverse.transposed(), E);
		
		// init the coordinate shift //
		int t_u = this.projectionWidth/2;
		int t_v = this.projectionHeight/2;
		
		// compute angle alpha and distance to origin t //
		double l0 = l_kappa.getElement(0);
		double l1 = l_kappa.getElement(1);
		double l2 = l_kappa.getElement(2);
		
		double alpha_kappa_RAD = Math.atan2(-l0, l1) + Math.PI/2;
		
		
		double t_kappa = -l2 / Math.sqrt(l0*l0+l1*l1);
		// correct the coordinate shift //
		t_kappa -= t_u * Math.cos(alpha_kappa_RAD) + t_v * Math.sin(alpha_kappa_RAD);

		// correct some alpha falling out of the radon window //
		if (alpha_kappa_RAD < 0) {
			alpha_kappa_RAD *= -1.0;
		} else if (alpha_kappa_RAD > Math.PI) {
			alpha_kappa_RAD = 2.0*Math.PI - alpha_kappa_RAD;
		}
		
		// write back to l_kappa //
		l_kappa.setElementValue(0, Math.cos(alpha_kappa_RAD));
		l_kappa.setElementValue(1, Math.sin(alpha_kappa_RAD));
		l_kappa.setElementValue(2, -t_kappa);
		
		// calculate x and y coordinates for derived radon transformed image //
		double x = t_kappa * this.lineIncrement + 0.5 * this.radonWidth;
		double y = alpha_kappa_RAD * this.angleIncrement;


		// get intensity value out of radon transformed image //
		// (interpolation needed)
		return InterpolationOperators.interpolateLinear(this.radon, x, y);
		
	}
	
	
	/**
	 * method to compute the epipolar line integrals that state the epipolar consistency conditions
	 * by comparison of two views
	 * @param kappa_RAD: angle of epipolar plane
	 * @param epi1: class instance representing the first view
	 * @param epi2: class instance representing the second view
	 * @param K: mapping matrix K obtained from method createMappingToEpipolarPlane()
	 * @param P1_Inverse: inverse of projection image of first view as SimpleMatrix (size 4x3) 
	 * @param P2_Inverse: inverse of projection image of second view as SimpleMatrix (size 4x3) 
	 * @return
	 */
	public static double[] computeEpipolarLineIntegrals(double kappa_RAD, EpipolarConsistency epi1, EpipolarConsistency epi2, SimpleMatrix K, SimpleMatrix P1_Inverse, SimpleMatrix P2_Inverse) {
		
		// compute points on unit-circle (3x1) //
		SimpleVector x_kappa = new SimpleVector(Math.cos(kappa_RAD), Math.sin(kappa_RAD), 1);
		
		// compute epipolar plane E_kappa (4x1 = 4x3 * 3x1) //
		SimpleVector E_kappa = SimpleOperators.multiply(K, x_kappa);
		
		// compute line integral out of derived radon transform //
		double value1 = epi1.getValueByAlphaAndT(P1_Inverse, E_kappa);
		double value2 = epi2.getValueByAlphaAndT(P2_Inverse, E_kappa);

		// both values are returned //
		return new double[]{value1, value2};
		
	}

}
/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
