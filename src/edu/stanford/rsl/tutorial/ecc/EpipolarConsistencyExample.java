/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.ecc;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLForwardProjector;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * this shows an implementation how to use the Epipolar Consistency class
 * to compute line integrals that fulfill the Epipolar Consistency Conditions
 * for this purpose a 3D phantom is simulated and all projections are computed out of it
 * the used xml file is the standard Conrad.xml file created automatically
 * when starting ReconstructionPipelineFrame
 */
public class EpipolarConsistencyExample {

	public static void main(String[] args) {
		
		new ImageJ();
		
		// load configuration from xml file //
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();

		// get the dimensions //
		Trajectory geo = conf.getGeometry();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		
		// simulate a 3D phantom as object //
		Grid3D phantom = new NumericalSheppLogan3D(
				imgSizeX, imgSizeY, imgSizeZ).getNumericalSheppLoganPhantom();
		
		phantom.show("simulated object");
		
		
		// calculate all projections contained in the xml file //
		// (in PMatrixSerialization)
		Grid3D totalProj = null;
		
		// a forward projector is needed
		OpenCLForwardProjector forwProj =  new OpenCLForwardProjector();
		forwProj.setTex3D(ImageUtil.wrapGrid3D(phantom, ""));
		
		// start calculating the projections
		try {
			forwProj.configure();
			totalProj = ImageUtil.wrapImagePlus(forwProj.project());
		} catch (Exception e) {
			System.err.println(e);
			return;
		}
		
		// display all projections //
		totalProj.show("projections of object");
		
		// now we set up two views arbitrary out of all projections
		// by creating the class instances
		// these views are going to be compared in the following
		// the indices are needed for both the projection images and the projection matrices
		// they state the position of the projection matrix in xml file
		// ( PMatrixSerialization; 0 is the first projection matrix ) //
		int index1 = 50;
	    int index2 = 100;
		
	    // get the projection images and show //
		Grid2D projection1 = totalProj.getSubGrid(index1);
		projection1.show("Projection1");
		
		Grid2D projection2 = totalProj.getSubGrid(index2);
		projection2.show("Projection2");
		
		
		// get projection matrices as a Projection class //
		Projection[] matrices = geo.getProjectionMatrices();
		
		Projection proIndex1 = matrices[index1];
		Projection proIndex2 = matrices[index2];
		
		// precompute radon transformed and derived images and show //
		final int radonSize = 1024;
		Grid2D radon1 = SimpleOperators.computeRadonTrafoAndDerive(projection1, radonSize);
		radon1.show("Radon1");
		
		Grid2D radon2 = SimpleOperators.computeRadonTrafoAndDerive(projection2, radonSize);
		radon2.show("Radon2");
		
		
		
		
		// create class instances //
		EpipolarConsistency epi1 = new EpipolarConsistency(projection1, radon1, proIndex1);
		EpipolarConsistency epi2 = new EpipolarConsistency(projection2, radon2, proIndex2);

		
		// get the mapping matrix to the epipolar plane //
		SimpleMatrix K = EpipolarConsistency.createMappingToEpipolarPlane(epi1.C, epi2.C);
		// (K is a 4x3 matrix)
		
		// calculate inverses of projection matrices //
		SimpleMatrix Pa_Inverse = epi1.P.inverse(InversionType.INVERT_SVD);
		SimpleMatrix Pb_Inverse = epi2.P.inverse(InversionType.INVERT_SVD);
				

		// go through angles //
		// we go through a range of [-8°, +8°] in a stepsize of 0.05°
		double angleBorder = 8.0;
		double angleIncrement = 0.05;
		// get number of decimal places of angleIncrement
		String[] split = Double.toString(angleIncrement).split("\\.");
		int decimalPlaces = split[1].length();
		
		int height = (int) (angleBorder * 2 / angleIncrement + 1);
	
		// results are saved in an array in the format [angle,valueView1,valueView2]
		double[][] results = new double[height][3];
		int count = 0;
				
		for (double kappa = -angleBorder; kappa <= angleBorder; kappa += angleIncrement) {
			
			double kappa_RAD = kappa / 180.0 * Math.PI;
			
			// get values for line integrals that fulfill the epipolar consistency conditions //
			double[] values = EpipolarConsistency.computeEpipolarLineIntegrals(kappa_RAD, epi1, epi2, K, Pa_Inverse, Pb_Inverse);
			results[count][0] = Math.round(kappa*Math.pow(10, decimalPlaces)) / (Math.pow(10, decimalPlaces) + 0.0);
			results[count][1] = values[0];
			results[count][2] = values[1];
			count++;
		}

		// show results //
		for (int i = 0; i < results.length; i++) {
			System.out.println("at angle kappa: " + results[i][0] + " P1: " + results[i][1] + " P2: " + results[i][2]);
		}

	}

}
/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */