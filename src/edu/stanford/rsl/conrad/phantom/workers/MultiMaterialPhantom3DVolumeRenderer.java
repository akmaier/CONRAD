/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.phantom.workers;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.rendering.AbstractRayTracer;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.Priority1DRayTracer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Renders arbitrarily defined phantoms into material images
 * 
 * 
 * @author Andreas Maier
 * 
 */
public class MultiMaterialPhantom3DVolumeRenderer extends SliceWorker {
	protected AnalyticPhantom phantom = null;
	protected PrioritizableScene currentScene = null;

	@Override
	public String getProcessName() {
		return "Generic Multi Material Phantom Renderer";
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
		//FloatProcessor slice = new FloatProcessor(geom.getReconDimensionX(), geom.getReconDimensionY());

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
		// create list of all available materials.
		ArrayList<Material> materials = new ArrayList<Material>();
		for (int j=0; j < phantomScene.size(); j++){
			Material current = phantomScene.getObject(j).getMaterial();
			if (!materials.contains(current)){
				materials.add(current);
			}
		}
		// Create channel labels
		String []channelNames = new String[materials.size()];
		for (int i=0; i < materials.size(); i ++){
			channelNames[i]=materials.get(i).getName(); 
		}

		MultiChannelGrid2D multiChannelGrid2D = new MultiChannelGrid2D(geom.getReconDimensionX(), geom.getReconDimensionY(), materials.size());
		multiChannelGrid2D.setChannelNames(channelNames);

		for (int c= 0; c<multiChannelGrid2D.getNumberOfChannels();c++){
			Grid2D slice = new Grid2D(multiChannelGrid2D.getWidth(), multiChannelGrid2D.getHeight());
			multiChannelGrid2D.setChannel(c, slice);
			double z  = (sliceNumber - originIndexZ) * voxelSizeZ;
			double xFirst = (-originIndexX) * voxelSizeX;
			double xLast = (slice.getWidth()-originIndexX) * voxelSizeX;
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
						if(o.getMaterial().equals(materials.get(c))){
							Edge lineSegment = (Edge) o.getShape(); 
							PointND p1 = lineSegment.getPoint();
							PointND p2 = lineSegment.getEnd();
							int ix1 = (int) Math.round((p1.get(0) / voxelSizeX) + originIndexX);
							int ix2 = (int) Math.round((p2.get(0) / voxelSizeX) + originIndexX);
							if (ix2 <= ix1){
								continue;
							}
							int iy = (int) Math.round((y / voxelSizeY) + originIndexY);
							for (int l = ix1; l < ix2; l++){
								slice.setAtIndex(l, iy, 1);
							}
						}
					}
					long renderTime = System.currentTimeMillis() - basetime;
					if (sliceNumber == 123) System.out.println("Cast time " + castTime + " render time " + renderTime + " " + segments.size());
				}
			}
		}
		this.imageBuffer.add(multiChannelGrid2D, sliceNumber);
	}

	public SliceWorker clone() {
		MultiMaterialPhantom3DVolumeRenderer newRend = new MultiMaterialPhantom3DVolumeRenderer();
		newRend.phantom = this.phantom;
		return newRend;
	}

	@Override
	public void configure() throws Exception {
		super.configure();
		phantom = UserUtil.queryPhantom("Select Phantom", "Select Phantom");
		phantom.configure();
	}






}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */