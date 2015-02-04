package edu.stanford.rsl.conrad.rendering;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Creates wrapper for elements required to define a ray detector.
 * @author Rotimi X Ojo
 */
public class RayDetector {
	
    private Trajectory geometry = Configuration.getGlobalConfiguration().getGeometry();
	private  SimpleVector principalPointPix;
	private  SimpleVector principalPointWorldCood;
	private  SimpleVector camCenterWorldCood;
	private Projection projection;

	public RayDetector(int sliceNumber) {
		init(geometry.getProjectionMatrices()[sliceNumber]);
	}

	public RayDetector(Projection projection) {
		init(projection);
	}
	
	public RayDetector(Projection projection,Trajectory geometry) {
		this.geometry = geometry;
		init(projection);
	}

	private void init(Projection projection) {
		this.projection = projection;
		principalPointPix = projection.getPrincipalPoint();
		camCenterWorldCood = projection.computeCameraCenter();			
		SimpleVector CamToDetectorCamCood = projection.computePrincipalAxis()	.multipliedBy(geometry.getSourceToDetectorDistance());
		principalPointWorldCood = SimpleOperators.add(camCenterWorldCood,CamToDetectorCamCood);

	}
	
	public int getHeightInPixels(){
		return geometry.getDetectorHeight();
	}
	
	public int getWidthInPixels(){
		return geometry.getDetectorWidth();
	}
	
	public double getPixelWidth(){
		return geometry.getPixelDimensionX();
	}
	
	public double getPixelHeight(){
		return geometry.getPixelDimensionY();
	}
	
	public SimpleVector getPrincipalPointInPixels(){
		return principalPointPix;
	}
	
	public SimpleVector getPrincipalPointInMM(){
		return principalPointWorldCood;
	}

	/**
	 * Ordered Collection of rays incident on ray detector
	 * @return the edges from each detector pixel to the source
	 */
	public StraightLine[][] getIncidentRays(){
		StraightLine[][] rays = new StraightLine[getHeightInPixels()][getWidthInPixels()];		

		for(int y = 0; y < getHeightInPixels(); y++){
			for(int x = 0; x < getWidthInPixels(); x++){						
				SimpleVector globPixelCtr = convertPixelIndexToVector(new SimpleVector(y,x));
				rays[y][x] = new StraightLine(new PointND(camCenterWorldCood), globPixelCtr.normalizedL2());			
			}
		}
		return rays;
	}
	
	private SimpleVector convertPixelIndexToVector(
			SimpleVector pixelVec) {
		return projection.computeRayDirection(pixelVec);
	}

	
	/**
	 * For anti-aliasing
	 * @return super sampled data
	 */
	public StraightLine[][][] getSuperSampledIncidentRays(){
		return null;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/