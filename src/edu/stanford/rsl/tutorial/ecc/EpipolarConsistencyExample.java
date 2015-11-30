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
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLForwardProjector;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * this shows an implementation how to use the Epipolar Consistency class
 * to compute line integrals that fulfill the Epipolar Consistency Conditions
 * for this purpose a 3D phantom is simulated and 200 projections are computed out of it
 * the used xml file is the standard Conrad.xml file created automatically
 * when starting ReconstructionPipelineFrame
 * @author Martin Berzl
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
		
		System.out.println("Simulation of data started ..");
		
		// simulate a 3D phantom as object //
		Grid3D phantom = new NumericalSheppLogan3D(
				imgSizeX, imgSizeY, imgSizeZ).getNumericalSheppLoganPhantom();
		
		phantom.show("simulated object");
		
		
		// calculate all projections contained in the xml file //
		// (in PMatrixSerialization)
		Grid3D totalProj = null;
		
		System.out.println("Creation of projections ..");
		
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
		Grid2D projection2 = totalProj.getSubGrid(index2);

		
		// get projection matrices as a Projection class //
		Projection[] matrices = geo.getProjectionMatrices();
		
		Projection projMatrix1 = matrices[index1];
		Projection projMatrix2 = matrices[index2];
		
		System.out.println("Precomputing of radon transformations ..");
		
		// precompute radon transformed and derived images and show //
		final int radonSize = 1024;
		Grid2D radon1 = EpipolarConsistency.computeRadonTrafoAndDerive(projection1, radonSize);
		projection1.show("Projection1");
		radon1.show("Radon1");
		
		Grid2D radon2 = EpipolarConsistency.computeRadonTrafoAndDerive(projection2, radonSize);
		projection2.show("Projection2");
		radon2.show("Radon2");
		
		
		// simulation and precomputing of data done //
		System.out.println("\nSimulation and precomputing of data done!\n"
				+ "  creation of metric .. ");
		
		// create class instance //
		EpipolarConsistency metric = new EpipolarConsistency(
				projection1, projection2, radon1, radon2, projMatrix1, projMatrix2);
		
		// go through angles //
		// we go through a range of [-8°, +8°] in a stepsize of 0.05°
		// get number of decimal places of angleIncrement
		double angleBorder = 8.0;
		double angleIncrement = 0.05;
		
		metric.evaluateConsistency(-angleBorder, angleBorder, angleIncrement);	
		
		System.out.println("Done.");

	}

}
/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */