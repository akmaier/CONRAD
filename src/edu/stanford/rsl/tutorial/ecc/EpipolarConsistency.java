/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.ecc;

import ij.IJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.Nrrd_Reader;
import edu.stanford.rsl.conrad.numerics.*;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.motion.estimation.SobelKernel1D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;


public class EpipolarConsistency {
	
	//* center matrix *//
	private SimpleMatrix CENTER;
	
	//* projection matrix of a view *//
	public SimpleMatrix P;
	
	//* source position of a view *//
	public SimpleVector C;
	
	//* radon transformed image of view *//
	public Grid2D radon;
	
	//* further dimensions *//
	public int projectionWidth;
	public int projectionHeight;
	public int radonHeight;
	public int radonWidth;
	public double projectionDiag;
	public double lineIncrement;
	public double angleIncrement;
	
	/**
	 * constructor to compute the epipolar consistency for one view.
	 * To compute the line integrals between two views, it is neccessary to create two instances of this class
	 * and call the other methods on it
	 * @param directory: for image input files
	 * @param fileNameP: filename of a projection image
	 * @param fileNameRadon: filename of it's radon transformed and in t-direction derived image
	 * @param fileFormat: nrrd or tiff
	 * @param index: stands for the index number of projection matrix stored in the xml file
	 */
	public EpipolarConsistency(String directory, String fileNameP, String fileNameRadon, String fileFormat, int index) {
		
		//* Initialize center matrix *//
		CENTER = new SimpleMatrix(3,4);
		CENTER.setDiagValue(new SimpleVector(1.0, 1.0, 1.0));
				
		
		String xmlFile = "";
		
		
		//* load projection and radon transformed image *//
		Grid2D projection;
		Grid2D radon;
		
		if (fileFormat == "nrrd" || fileFormat == ".nrrd") {
			
			Nrrd_Reader nrrd = new Nrrd_Reader();
			ImagePlus imgP = nrrd.load(directory, fileNameP);
			// Convert from ImageJ to Grid3D and afterwards to Grid2D.
			// Note that no data is copied here.
			projection = ImageUtil.wrapImagePlus(imgP).getSubGrid(0);
			
			// load RadonTrafo
			ImagePlus imgRadon = nrrd.load(directory, fileNameRadon);
			radon = ImageUtil.wrapImagePlus(imgRadon).getSubGrid(0);
			
			xmlFile = "ConradNRRD.xml";
			
		} else {
			
			Grid3D data = null;
			try {
			    ImagePlus imp = IJ.openImage(directory + "\\" + fileNameP);
			    data = ImageUtil.wrapImagePlus(imp);

			} catch (Exception e) {
			    e.printStackTrace();
			}
			projection = data.getSubGrid(index);
			
			// compute RadonTrafo
			radon = computeRadonTrafoAndDerive(projection);
			
			xmlFile = "ConradTIFF.xml";
			
		}

		
		//* get data out of projection *//
		this.projectionWidth = projection.getWidth();
		this.projectionHeight = projection.getHeight();
			
		//* show projection image *//
		projection.show("Projection");
		
		
		//* get data out of radon transformed image *//
		this.radonWidth = radon.getWidth();
		this.radonHeight = radon.getHeight();
		this.projectionDiag = Math.sqrt(projectionWidth*projectionWidth + projectionHeight*projectionHeight);
		this.lineIncrement = radonWidth / projectionDiag;
		this.angleIncrement = radonHeight / Math.PI;

		//* show radon transformed image *//
		radon.show("Radon-Trafo");
		this.radon = radon;
		
		
		//* Configuration of CONRAD *//	
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(directory + "\\" + xmlFile));

		//* get projection matrix P (3x4) *//
		Trajectory geo = Configuration.getGlobalConfiguration().getGeometry();
		Projection[] matrices = geo.getProjectionMatrices();
		
		Projection proIndex = matrices[index];
		this.P = SimpleOperators.multiplyMatrixProd(proIndex.getK(), CENTER);
		this.P = SimpleOperators.multiplyMatrixProd(P, proIndex.getRt());
				
		
		//* get source position C (nullspace of the projection) *//
		DecompositionSVD decoP = new DecompositionSVD(P);
		this.C = decoP.getV().getCol(3);
		
		//* normalize source vectors by last component *//
		// it is important that the last component is positive to have a positive center
		// as it is defined in oriented projective geometry
		this.C = this.C.dividedBy(this.C.getElement(3));

	}
	
	
	/**
	 * method to compute the radon transformed and derived image for one view (projection)
	 * the derivation is done in t-direction (distance to origin)
	 * @param data: Grid2D which represents the projection
	 * @return: radon transformed image as Grid2D
	 */
	private Grid2D computeRadonTrafoAndDerive(Grid2D data) {
		
		Grid2D radon = null;
		
		// possible preprocessing can be done here
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

		//* Project forward in parallel *//
		ParallelProjector2D projector = new ParallelProjector2D(Math.PI, 1.0 / this.angleIncrement, this.projectionDiag, 1.0 / this.lineIncrement);
		radon = projector.projectRayDrivenCL(data);
		// for an CPU implementation use (if GPU version doesn't work etc.):
		//radon = projector.projectRayDriven(data);
				
		
		// derivative by Sobel operator in t-direction
		final int size = 1024;
		SobelKernel1D kernel= new SobelKernel1D(size,9);
		kernel.applyToGrid(radon);
		
		//* optional: save file in tiff-format *//
		/*
		FileSaver saveRadon = new FileSaver(ImageUtil.wrapGrid(radon, ""));
		saveRadon.saveAsTiff();
		*/
		
		return radon;
		
	}
	
	/**
	 * method to compute the Pluecker dual coordinates of a vector L
	 * @param L: SimpleVector having 6 elements
	 * @return: SimpleMatrix of size 4x4
	 */
	private static SimpleMatrix PlueckerMatrixDual(SimpleVector L) {
		
		SimpleMatrix L_out = new SimpleMatrix(4, 4);
		// first row
		L_out.setElementValue(0, 1, +L.getElement(5));
		L_out.setElementValue(0, 2, -L.getElement(4));
		L_out.setElementValue(0, 3, +L.getElement(3));
		
		// second row
		L_out.setElementValue(1, 0, -L.getElement(5));
		L_out.setElementValue(1, 2, +L.getElement(2));
		L_out.setElementValue(1, 3, -L.getElement(1));
		
		// third row
		L_out.setElementValue(2, 0, +L.getElement(4));
		L_out.setElementValue(2, 1, -L.getElement(2));
		L_out.setElementValue(2, 3, +L.getElement(0));
		
		// last row
		L_out.setElementValue(3, 0, -L.getElement(3));
		L_out.setElementValue(3, 1, +L.getElement(1));
		L_out.setElementValue(3, 2, -L.getElement(0));
		
		return L_out;
		
	}

	/**
	 * method to compute the Pluecker join of a line L and a point X
	 * @param L: line L as SimpleVector (6 entries)
	 * @param X: point X as SimpleVector (4 entries)
	 * @return: SimpleVector of size 4x1 representing a plane
	 */
	private static SimpleVector PlueckerJoin(SimpleVector L, SimpleVector X) {
		
		//* calculate plane E (4x1) from [~L]x * X *//
		double v1 = + X.getElement(1)*L.getElement(5) - X.getElement(2)*L.getElement(4) + X.getElement(3)*L.getElement(3);
		double v2 = - X.getElement(0)*L.getElement(5) + X.getElement(2)*L.getElement(2) - X.getElement(3)*L.getElement(1);
		double v3 = + X.getElement(0)*L.getElement(4) - X.getElement(1)*L.getElement(2) + X.getElement(3)*L.getElement(0);
		double v4 = - X.getElement(0)*L.getElement(3) + X.getElement(1)*L.getElement(1) - X.getElement(2)*L.getElement(0);
		
		return new SimpleVector(v1, v2, v3, v4);
		
	}

	/**
	 * method to calculate a mapping K from two source positions C0, C1 to a plane
	 * @param C0: first source position (first view) as SimpleVector (4 entries)
	 * @param C1: second source position (second view) as SimpleVector (4 entries)
	 * @return: SimpleMatrix representing the mapping K (size 4x3)
	 */
	public static SimpleMatrix createMappingToEpipolarPlane(SimpleVector C0, SimpleVector C1) {
		
		//* compute Pluecker coordinates *//
		double L01 = C0.getElement(0)*C1.getElement(1) - C0.getElement(1)*C1.getElement(0);
		double L02 = C0.getElement(0)*C1.getElement(2) - C0.getElement(2)*C1.getElement(0);
		double L03 = C0.getElement(0)*C1.getElement(3) - C0.getElement(3)*C1.getElement(0);
		double L12 = C0.getElement(1)*C1.getElement(2) - C0.getElement(2)*C1.getElement(1);
		double L13 = C0.getElement(1)*C1.getElement(3) - C0.getElement(3)*C1.getElement(1);
		double L23 = C0.getElement(2)*C1.getElement(3) - C0.getElement(3)*C1.getElement(2);
		
		//* construct B (6x1) *//
		SimpleVector B = new SimpleVector(L01, L02, L03, L12, L13, L23);
		
		//* compute infinity point in direction of B *//
		SimpleVector N = new SimpleVector(-L03, -L13, -L23, 0);
		
		//* compute plane E0 containing B and X0=(0,0,0,1) *//
		SimpleVector E0 = PlueckerJoin(B, new SimpleVector(0, 0, 0, 1));
		
		//* find othonormal basis from plane normals *//
		// (vectors are of 3x1)		
		SimpleVector a2 = new SimpleVector(E0.getElement(0), E0.getElement(1), E0.getElement(2));
		SimpleVector a3 = new SimpleVector(N.getElement(0), N.getElement(1), N.getElement(2));
		// set vectors to unit length
		a2.normalizeL2();
		a3.normalizeL2();
				
		//* calculate cross product to get the last basis vector *//
		SimpleVector a1 = General.crossProduct(a2, a3).negated();
		// (a1 is already of unit length -> no normalization needed)
		
		//* set up assembly matrix A (4x3) *//
		SimpleMatrix A = new SimpleMatrix(4, 3);
		A.setSubColValue(0, 0, a1);
		A.setSubColValue(0, 1, a2);
		A.setSubColValue(0, 2, C0);
		
		//* return mapping matrix K (4x3 = 4x4 * 4x3) *//
		return SimpleOperators.multiplyMatrixProd(PlueckerMatrixDual(B), A);
		
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
		
		//* compute corresponding epipolar lines *//
		// (epipolar lines are of 3x1 = 3x4 * 4x1)
		SimpleVector l_kappa = SimpleOperators.multiply(P_Inverse.transposed(), E);
		
		//* init the coordinate shift *//
		int t_u = this.projectionWidth/2;
		int t_v = this.projectionHeight/2;
		
		//* compute angle alpha and distance to origin t *//
		double l0 = l_kappa.getElement(0);
		double l1 = l_kappa.getElement(1);
		double l2 = l_kappa.getElement(2);
		
		double alpha_kappa_RAD = Math.atan2(-l0, l1) + Math.PI/2;
		
		double t_kappa = -l2 / Math.sqrt(l0*l0+l1*l1);
		// correct the coordinate shift
		t_kappa -= t_u * Math.cos(alpha_kappa_RAD) + t_v * Math.sin(alpha_kappa_RAD);

		
		//* write back to l_kappa *//
		l_kappa.setElementValue(0, Math.cos(alpha_kappa_RAD));
		l_kappa.setElementValue(1, Math.sin(alpha_kappa_RAD));
		l_kappa.setElementValue(2, -t_kappa);
		
		//* calculate x and y coordinates for derived radon transformed image *//
		double x = t_kappa * this.lineIncrement + 0.5 * this.radonWidth;
		double y = alpha_kappa_RAD * this.angleIncrement;


		//* get intensity value out of radon transformed image *//
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
		
		//* compute points on unit-circle (3x1) *//
		SimpleVector x_kappa = new SimpleVector(Math.cos(kappa_RAD), Math.sin(kappa_RAD), 1);
		
		//* compute epipolar plane E_kappa (4x1 = 4x3 * 3x1) *//
		SimpleVector E_kappa = SimpleOperators.multiply(K, x_kappa);
		
		//* compute line integral out of derived radon transform *//
		double value1 = epi1.getValueByAlphaAndT(P1_Inverse, E_kappa);
		double value2 = epi2.getValueByAlphaAndT(P2_Inverse, E_kappa);

		//* both values are returned *//
		return new double[]{value1, value2};
		
	}


}
