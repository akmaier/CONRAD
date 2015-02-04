package edu.stanford.rsl.conrad.utils;

import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;

public abstract class TessellationUtil {

	static int lowerLimit =5;
	static int upperLimit =50;
	
	/**
	 * Method to estimate a good sampling factor given the voxel size and the size of the volume that is used for discretization.
	 * Given the current volume configuration (from the gloabal configuration) and the shape to tessellate, the method determines
	 * a good estimate for samplingU. <br>
	 * A good value for the subSamplingFactor is 4. The higher the value, the less points will be tessellated.
	 * A value of 1 tessellates at least one point per voxel.   
	 * @param shape the shape to tessellate
	 * @param subSamplingFactor the subsampling factor
	 * @return the sampling factor in u direction.
	 */
	public static double getSamplingU(AbstractSurface shape){
		
		PointND min = shape.getMin();
		PointND max = shape.getMax();
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		double voxelSizeX = traj.getVoxelSpacingX();
		double voxelSizeY = traj.getVoxelSpacingY();
		double voxelSizeZ = traj.getVoxelSpacingZ();
		
		int dimx = traj.getReconDimensionX();
		double samplingU = dimx / getSubSamplingFactor();
		
		int width = (int) ((max.get(0) - min.get(0)) / voxelSizeX);
		double samplingFactorU = samplingU / width;
		
		double rangeX = (max.get(0) - min.get(0)) / voxelSizeX;
		double rangeY = (max.get(1) - min.get(1)) / voxelSizeY;
		double rangeZ = (max.get(2) - min.get(2)) / voxelSizeZ;
		int maxRange = (int) Math.ceil(Math.max(Math.max(rangeX, rangeY), rangeZ));
		samplingU = ((int)(maxRange * samplingFactorU));
		if (samplingU < lowerLimit) samplingU = lowerLimit;
		if (samplingU > upperLimit) samplingU = upperLimit;
		return samplingU;
	}
	
	public static double getSubSamplingFactor(){
		return Double.parseDouble(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.SPLINE_SUBSAMPLING_FACTOR));
	}
	
	/**
	 * Method to estimate a good sampling factor given the voxel size and the size of the volume that is used for discretization.
	 * Given the current volume configuration (from the gloabal configuration) and the shape to tessellate, the method determines
	 * a good estimate for samplingV. <br>
	 * A good value for the subSamplingFactor is 4. The higher the value, the less points will be tessellated.
	 * A value of 1 tessellates at least one point per voxel.   
	 * @param shape the shape to tessellate
	 * @param subSamplingFactor the subsampling factor
	 * @return the sampling factor in v direction.
	 */
	public static double getSamplingV(AbstractSurface shape){
		
		PointND min = shape.getMin();
		PointND max = shape.getMax();
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		double voxelSizeX = traj.getVoxelSpacingX();
		double voxelSizeY = traj.getVoxelSpacingY();
		double voxelSizeZ = traj.getVoxelSpacingZ();
		
		int dimy = traj.getReconDimensionY();
		double samplingV = dimy / getSubSamplingFactor();
		
		int height = (int) ((max.get(1) - min.get(1)) / voxelSizeY);
		double samplingFactorV = samplingV / height;
		
		double rangeX = (max.get(0) - min.get(0)) / voxelSizeX;
		double rangeY = (max.get(1) - min.get(1)) / voxelSizeY;
		double rangeZ = (max.get(2) - min.get(2)) / voxelSizeZ;
		int maxRange = (int) Math.ceil(Math.max(Math.max(rangeX, rangeY), rangeZ));
		samplingV = ((int)(maxRange * samplingFactorV));
		if (samplingV < lowerLimit) samplingV = lowerLimit;
		if (samplingV > upperLimit) samplingV = upperLimit;
		return samplingV;
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/