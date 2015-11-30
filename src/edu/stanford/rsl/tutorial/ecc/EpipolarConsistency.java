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
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.tutorial.motion.estimation.SobelKernel1D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;

/**
 * class to compare two view with each other by
 * applying the Epipolar Consistency Conditions
 * @author Martin Berzl
 */
public class EpipolarConsistency {
	
	
	/**
	 * inner class View representing a view with all its properties
	 * by combining two views in a correct manner, it is possible to compute the
	 * epipolar consistency conditions out of it
	 * @author Martin Berzl
	 */
	public class View {
		
		// center matrix //
		private SimpleMatrix CENTER;
		
		// projection matrix of a view //
		public SimpleMatrix P;
		
		// inverse projection matrix of a view //
		public SimpleMatrix P_Inverse;
		
		// source positions of a view //
		public SimpleVector C;
		
		// radon transformed image of a view //
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
		 * constructor to create a view
		 * @param projection: projection image as Grid2D
		 * @param radon: radon transformed and derived image as Grid2D
		 * @param projMatrix: projection matrix as Projection
		 */
		public View(Grid2D projection, Grid2D radon, Projection projMatrix) {
			
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
			this.P = SimpleOperators.multiplyMatrixProd(projMatrix.getK(), CENTER);
			this.P = SimpleOperators.multiplyMatrixProd(this.P, projMatrix.getRt());
			
			// get source position C (nullspace of the projection) //
			DecompositionSVD decoP = new DecompositionSVD(this.P);
			this.C = decoP.getV().getCol(3);
			
			// normalize source vectors by last component //
			// it is important that the last component is positive to have a positive center
			// as it is defined in oriented projective geometry
			this.C = this.C.dividedBy(this.C.getElement(3));
		}
		
		
		/**
		 * Method to calculate alpha and t as indices from the radon transformed image of a view.
		 * Neccessary are the inverse projection image (as SimpleMatrix) and the
		 * epipolar plane E
		 * @param E: epipolar plane E as SimpleVector (4 entries)
		 * @return: line integral value as double
		 */
		private double getValueByAlphaAndT(SimpleVector E) {
			
			// compute corresponding epipolar lines //
			// (epipolar lines are of 3x1 = 3x4 * 4x1)
			SimpleVector l_kappa = SimpleOperators.multiply(this.P_Inverse.transposed(), E);
			
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
		
	}
	/**
	 * end of inner class View
	 */
	
	
	// attributes reserved for the epipolar consistency class //
	// first view //
	public View view1;
	
	// second view //
	public View view2;
	
	// mapping matrix by combining first and second view //
	private SimpleMatrix K;
	
	
	/**
	 * constructor to compute the metric for epipolar consistency for one view.
	 * To compute the line integrals between two views, it is neccessary to create two instances of this class
	 * and call the other methods on it
	 * @param projection1: Grid2D representing the first projection
	 * @param projection2 Grid2D representing the second projection
	 * @param radon1: Grid2D representing the radon transformed and derived image of the first view
	 *  derivation is done in t direction (distance on detector)
	 * @param radon2: Grid2D representing the radon transformed and derived image of the second view
	 *  derivation is done in t direction (distance on detector)
	 * @param projMatrix1: Projection that stands for the index number
	 *  of projection matrix of the first view stored in the xml file
	 * @param projMatrix2: Projection that stands for the index number
	 *  of projection matrix of the second view stored in the xml file
	 */
	public EpipolarConsistency(Grid2D projection1, Grid2D projection2, Grid2D radon1, Grid2D radon2, Projection projMatrix1, Projection projMatrix2) {
		
		// create two views //
		view1 = new View(projection1, radon1, projMatrix1);
		view2 = new View(projection2, radon2, projMatrix2);

	}
	
	/**
	 * getter-method to have access to the mapping matrix K
	 * @return Simple Matrix representing the mapping matrix
	 */
	public SimpleMatrix getMappingMatrix() {
		return K;
	}
	
	/**
	 * setter-method to store the mapping matrix K
	 * @param K: SimpleMatrix containing the mapping matrix
	 */
	public void setMappingMatrix(SimpleMatrix K) {
		this.K = K;
	}
	
	
	/**
	 * method to compute the squared radon transformed and derived image for one view
	 * the derivation is done in t-direction (distance to origin)
	 * @param data: Grid2D which represents the projection
	 * @param radonSize: value to determine the size of the squared radon transformed
	 *  and derived image
	 * @return: radon transformed image as Grid2D (its size is radonSize x radonSize)
	 */
	public static Grid2D computeRadonTrafoAndDerive(Grid2D data, int radonSize) {
		
		Grid2D radon = null;
		
		// optional: preprocessing can be done here
		// for example:
		/*
		float value;
		for (int i = 0; i < data.getWidth(); i++) {
			for (int j = 0; j < data.getHeight(); j++) {
				if (j < 10 || j > data.getHeight() - 11) {
					// set border to zero
					value = 0.0f;
				} else {
					value = (float)(-Math.log(data.getAtIndex(i, j) / 0.842f));
				}
				data.setAtIndex(i, j, value);
			}
		}

		data.show("preprocessed image");
		*/

		//* get some dimensions out of data (projection) *//
		int projSizeX = data.getWidth();
		int projSizeY = data.getHeight();
		double projectionDiag = Math.sqrt(projSizeX*projSizeX + projSizeY*projSizeY);
		
		double deltaS = projectionDiag / radonSize;
		double deltaTheta = Math.PI / radonSize;
		
		//* create a parallel projector in order to compute the radon transformation *//
		ParallelProjector2D projector = new ParallelProjector2D(
				Math.PI, deltaTheta, projectionDiag, deltaS);

		//* create radon transformation *//
		radon = projector.projectRayDriven(data);
		// for a faster GPU implementation use: (possibly not working):
		//radon = projector.projectRayDrivenCL(data);
				

		//* derivative by Sobel operator in t-direction *//
		final int size = 1024;
		SobelKernel1D kernel= new SobelKernel1D(size, 9);
		kernel.applyToGrid(radon);
		
		//* optional: save file in tiff-format *//
		/*
		FileSaver saveRadon = new FileSaver(ImageUtil.wrapGrid(radon, ""));
		saveRadon.saveAsTiff();
		*/
		
		return radon;
		
	}
	
	
	/**
	 * method to calculate a mapping K from two source positions C0, C1 to a plane
	 * C0 (C1) is the source position from the first (second) view
	 */
	public void createMappingToEpipolarPlane() {
		
		// set up source matrices //
		SimpleVector C0 = this.view1.C;
		SimpleVector C1 = this.view2.C;
		
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
		
		// store mapping matrix K (4x3 = 4x4 * 4x3) //
		this.K = SimpleOperators.multiplyMatrixProd(SimpleOperators.getPlueckerMatrixDual(B), A);
		
	}
	
	
	/**
	 * method to compute the epipolar line integrals that state the epipolar consistency conditions
	 * by comparison of two views
	 * @param kappa_RAD: angle of epipolar plane
	 * @return double[] array containing two values (one for each view's line integral)
	 */
	public double[] computeEpipolarLineIntegrals(double kappa_RAD) {
		
		// compute points on unit-circle (3x1) //
		SimpleVector x_kappa = new SimpleVector(Math.cos(kappa_RAD), Math.sin(kappa_RAD), 1);
		
		// compute epipolar plane E_kappa (4x1 = 4x3 * 3x1) //
		SimpleVector E_kappa = SimpleOperators.multiply(this.K, x_kappa);
		
		// compute line integral out of derived radon transform //
		double value1 = view1.getValueByAlphaAndT(E_kappa);
		double value2 = view2.getValueByAlphaAndT(E_kappa);

		// both values are returned //
		return new double[]{value1, value2};
		
	}
	
	/**
	 * method to compute the metric for the epipolar consistency conditions between two views
	 * with varying angle kappa from lowerBorderAngle to upperBorderAngle with an increment of angleIncrement
	 * @param lowerBorderAngle
	 * @param upperBorderAngle
	 * @param angleIncrement
	 * @return: 2D-array containing the line integral values of two different views of all angles
	 * 	running in the range defined by the input parameters
	 *  the stored format is [angle, valueView1, valueView2] in the first dimension,
	 *  increasing/decreasing angle in the second
	 */
	public double[][] evaluateConsistency(double lowerBorderAngle, double upperBorderAngle, double angleIncrement) {
		
		// compute the mapping matrix to the epipolar plane //
		createMappingToEpipolarPlane();
		// (K is a 4x3 matrix)
		
		// calculate inverses of projection matrices and save to view //
		this.view1.P_Inverse = this.view1.P.inverse(InversionType.INVERT_SVD);
		this.view2.P_Inverse = this.view2.P.inverse(InversionType.INVERT_SVD);
		
		
		// get number of decimal places of the angleIncrement //
		String[] split = Double.toString(angleIncrement).split("\\.");
		int decimalPlaces = split[1].length();
		
		// obtain size parameter for results array //
		int height = (int) ((upperBorderAngle - lowerBorderAngle) / angleIncrement + 1);
	
		// results are saved in an array in the format [angle,valueView1,valueView2]
		double[][] results = new double[height][3];
		int count = 0;
				
		// go through angles //
		for (double kappa = lowerBorderAngle; kappa <= upperBorderAngle; kappa += angleIncrement) {
			
			double kappa_RAD = kappa / 180.0 * Math.PI;
			
			// get values for line integrals that fulfill the epipolar consistency conditions //
			double[] values = computeEpipolarLineIntegrals(kappa_RAD);
			
			// store values in results array //
			results[count][0] = Math.round(kappa*Math.pow(10, decimalPlaces)) / (Math.pow(10, decimalPlaces) + 0.0);
			results[count][1] = values[0];
			results[count][2] = values[1];
			count++;
		}

		// show results //
		for (int i = 0; i < results.length; i++) {
			System.out.println("at angle kappa: " + results[i][0] + " P1: " + results[i][1] + " P2: " + results[i][2]);
		}
		
		return results;
		
	}	

}
/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
