package edu.stanford.rsl.conrad.geometry.trajectories;

import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.io.DennerleinProjectionSource;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

/**
 * Class to read a projection file in Dennerlein Format.
 * 
 *
 */
public class DennerleinProjectionTableFileTrajectory extends ConfigFileBasedTrajectory{

	/**
	 * 
	 */
	private static final long serialVersionUID = -1392472185220533956L;
	private boolean success = false;

	public DennerleinProjectionTableFileTrajectory(String filename, Trajectory model) throws IOException{
		super(model);
		readConfigFile(filename);
	}

	public DennerleinProjectionTableFileTrajectory(Trajectory model) {
		super(model);
	}

	public void readConfigFile(String filename) throws IOException{
		DennerleinProjectionSource source = new DennerleinProjectionSource();
		source.initStream(filename);
		ImagePlusDataSink sink = new ImagePlusDataSink();
		Grid2D imp = source.getNextProjection();
		try {
			int i =0;
			while (imp != null){
				sink.process(imp, i);
				i++;
				imp = source.getNextProjection();
			}
			sink.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Grid3D matrices = sink.getResult();
		numProjectionMatrices = matrices.getSize()[2];
		this.projectionStackSize = numProjectionMatrices;
		projectionMatrices = new Projection[numProjectionMatrices];
		secondaryAngles = new double[numProjectionMatrices];
		primaryAngles = new double[numProjectionMatrices];
		for (int i = 0; i<numProjectionMatrices;i++){
			SimpleMatrix mat = new SimpleMatrix(3,4);
			for (int j =0; j <3;j++){
				for(int k = 0; k < 4;k++){
					mat.setElementValue(j, k, matrices.getSubGrid(i).getPixelValue(k, j));
				}
			}
			projectionMatrices[i] = new Projection(mat);
		}
		updatePrimaryAngles();
	}


	@Override
	public boolean getSuccess() {
		return success;
	}

}
/*
 * Copyright (C) 2010-2014 Chris Schwemmer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/