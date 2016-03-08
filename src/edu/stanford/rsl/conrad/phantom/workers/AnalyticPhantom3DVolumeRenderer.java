package edu.stanford.rsl.conrad.phantom.workers;

import ij.process.FloatProcessor;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.AbstractRayTracer;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.Priority1DRayTracer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Renders arbitrarily defined phantoms
 * Works now with the correct origin computations.
 * 
 * @author Rotimi X Ojo
 * 
 */
public class AnalyticPhantom3DVolumeRenderer extends SliceWorker {
	protected boolean renderAttenuation = false;
	protected AnalyticPhantom phantom = null;
	protected PrioritizableScene currentScene = null;
	private static double DEFAULT_ATTENUATION = -1024; // air(HU)
	protected double xrayEnergy = 80;
	protected AttenuationType attType = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;

	@Override
	public String getProcessName() {
		return "Generic Phantom Renderer";
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void workOnSlice(int sliceNumber) {
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		double originIndexX = geom.getOriginInPixelsX();
		double originIndexY = geom.getOriginInPixelsY();
		double originIndexZ = geom.getOriginInPixelsZ();
		double voxelSizeX = geom.getVoxelSpacingX();
		double voxelSizeY = geom.getVoxelSpacingY();
		double voxelSizeZ = geom.getVoxelSpacingZ();
		FloatProcessor slice = new FloatProcessor(geom.getReconDimensionX(), geom.getReconDimensionY());

		PrioritizableScene phantomScene = phantom;
		AbstractRayTracer tracer = new Priority1DRayTracer();
		if (phantom instanceof AnalyticPhantom4D){
			if (currentScene == null) {
				AnalyticPhantom4D scene = (AnalyticPhantom4D) phantom;
				phantomScene = scene.getScene(0);
				String key = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.RENDER_PHANTOM_VOLUME_AUTO_CENTER);
				if (key != null){
					if (key.equals("true")) {
						Translation translate = phantom.computeCenterTranslation();
						for (PhysicalObject o : phantomScene){
							o.applyTransform(translate);
						}
					}
				}
				currentScene = phantomScene;
			}
			phantomScene = currentScene;
		}
		tracer.setScene(phantomScene);
		slice.add(DEFAULT_ATTENUATION);
		double z  = (sliceNumber - originIndexZ) * voxelSizeZ;
		double xFirst = (-originIndexX) * voxelSizeX;
		double xLast = (slice.getWidth()-originIndexX) * voxelSizeX;
		
		if (renderAttenuation && xrayEnergy > 0)
			slice.setValue(phantomScene.getBackgroundMaterial().getAttenuation(xrayEnergy,attType));
		else
			slice.setValue(phantomScene.getBackgroundMaterial().getDensity());
			
			
		slice.fill();
		for (int i = 0; i < slice.getHeight(); i ++){
			double y  = (i - originIndexY) * voxelSizeY;
			PointND pointLeft = new PointND(xFirst, y, z);
			PointND pointRight = new PointND(xLast, y, z);
			long basetime = System.currentTimeMillis();
			StraightLine line = new StraightLine(pointLeft, pointRight);
			line.normalize();
			ArrayList<PhysicalObject> segments = tracer.castRay(line);
			long castTime = System.currentTimeMillis() - basetime;
			basetime = System.currentTimeMillis();
			if (segments != null) {
				for (PhysicalObject o: segments){
					Edge lineSegment = (Edge) o.getShape(); 
					PointND p1 = lineSegment.getPoint();
					PointND p2 = lineSegment.getEnd();
					int ix1 = (int) Math.round((p1.get(0) / voxelSizeX) + originIndexX);
					int ix2 = (int) Math.round((p2.get(0) / voxelSizeX) + originIndexX);
					if (ix2 <= ix1){
						continue;
					}
					int iy = (int) Math.round((y / voxelSizeY) + originIndexY);
					
					if (renderAttenuation && xrayEnergy > 0)
						slice.setValue(o.getMaterial().getAttenuation(xrayEnergy, attType));
					else
						slice.setValue(o.getMaterial().getDensity());
						
						
					slice.drawLine(ix1, iy, ix2-1, iy);
				}
				long renderTime = System.currentTimeMillis() - basetime;
				if (sliceNumber == 123) System.out.println("Cast time " + castTime + " render time " + renderTime + " " + segments.size());
			}
		}
		Grid2D grid = new Grid2D((float[])slice.getPixels(), slice.getWidth(), slice.getHeight());
		this.imageBuffer.add(grid, sliceNumber);
	}

	public SliceWorker clone() {
		AnalyticPhantom3DVolumeRenderer newRend = new AnalyticPhantom3DVolumeRenderer();
		newRend.phantom = this.phantom;
		newRend.attType=attType;
		newRend.xrayEnergy = xrayEnergy;
		newRend.renderAttenuation = renderAttenuation;
		newRend.showStatus = showStatus;
		return newRend;
	}

	@Override
	public void configure() throws Exception {
		super.configure();
		phantom = UserUtil.queryPhantom("Select Phantom", "Select Phantom");
		phantom.configure();
		renderAttenuation = UserUtil.queryBoolean("Render energy dependent attenuation?");
		if(renderAttenuation)
			xrayEnergy = UserUtil.queryDouble("Monochromatic Xray energy [keV]", xrayEnergy);
	}






}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/