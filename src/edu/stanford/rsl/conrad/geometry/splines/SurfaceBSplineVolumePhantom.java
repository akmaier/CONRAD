package edu.stanford.rsl.conrad.geometry.splines;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.process.FloatProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.VectorPoint3D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.workers.SliceWorker;
import edu.stanford.rsl.conrad.phantom.xcat.XCatScene;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.absorption.AbsorptionModel;
import edu.stanford.rsl.conrad.physics.absorption.SelectableEnergyMonochromaticAbsorptionModel;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.AbstractRayTracer;
import edu.stanford.rsl.conrad.rendering.AbstractScene;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.Priority1DRayTracer;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class SurfaceBSplineVolumePhantom extends SliceWorker {

	private ArrayList<SurfaceBSpline> list;
	private AbstractScene scene;
	private PointND [] hullPoints;
	private ArrayList<Iterator<VectorPoint3D>> voxelList;
	private SimpleVector bounds;
	private boolean preserveAspects = true;
	private ArrayList<SimpleVector> localBounds;
	private boolean showRaster = false;
	private boolean showVertices = false;
	private boolean renderSolid = true;

	protected double xrayEnergy = 80;
	protected AttenuationType attType = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;
	boolean renderAttenuation = false;


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
		if (renderSolid){
			if (renderAttenuation && xrayEnergy > 0)
				slice.setValue(scene.getBackgroundMaterial().getAttenuation(xrayEnergy,attType));
			else
				slice.setValue(scene.getBackgroundMaterial().getDensity());
			slice.fill();
		}

		AbstractRayTracer tracer = new Priority1DRayTracer();
		tracer.setScene(scene);

		if (showVertices) {
			Iterator<VectorPoint3D> iterator = voxelList.get(sliceNumber);
			while(iterator.hasNext()){
				VectorPoint3D voxel = iterator.next();
				slice.putPixelValue((int)voxel.getX(), (int)voxel.getY(), voxel.getVector().getElement(0));
			}
		}

		double z  = (sliceNumber - originIndexZ) * voxelSizeZ;
		double xFirst = (-originIndexX) * voxelSizeX;
		double xLast = (slice.getWidth()-originIndexX) * voxelSizeX;
		if (renderSolid) {
			//long basetimeSlice = System.currentTimeMillis();
			for (int i = 0; i < slice.getHeight(); i ++){
				double y  = (i - originIndexY) * voxelSizeY;
				PointND pointLeft = new PointND(xFirst, y, z);
				PointND pointRight = new PointND(xLast, y, z);
				//long basetime = System.currentTimeMillis();
				StraightLine line = new StraightLine(pointLeft, pointRight);
				line.normalize();
				//System.out.println(line.getDirection());
				ArrayList<PhysicalObject> segments = tracer.castRay(line);
				//long castTime = System.currentTimeMillis() - basetime;
				//basetime = System.currentTimeMillis();
				if (segments != null) {
					for (PhysicalObject o: segments){
						Edge lineSegment = (Edge) o.getShape(); 
						PointND p1 = lineSegment.getPoint();
						PointND p2 = lineSegment.getEnd();
						int ix1 = (int) Math.round((p1.get(0) / voxelSizeX) + originIndexX);
						int ix2 = (int) Math.round((p2.get(0) / voxelSizeX) + originIndexX);
						int iy = (int) Math.round((y / voxelSizeY) + originIndexY);
						if (renderAttenuation && xrayEnergy > 0)
							slice.setValue(o.getMaterial().getAttenuation(xrayEnergy, attType)*o.getMaterial().getDensity());
						else
							slice.setValue(o.getMaterial().getDensity());
						slice.drawLine(ix1, iy, ix2, iy);
					}
					//long renderTime = System.currentTimeMillis() - basetime;
					//if (sliceNumber == 50) System.out.println("Cast time " + castTime + " render time " + renderTime + " " + segments.size());
				}

			}
			//long slicetime = System.currentTimeMillis() - basetimeSlice;
			//if (sliceNumber == 123) System.out.println("Slice time: " + slicetime);

		}
		if (showRaster) {
			for (double splineNum = 0; splineNum < list.size(); splineNum++) {
				for  (PointND p: hullPoints){
					if (Math.round(((p.get(2) / voxelSizeZ) + originIndexZ)) == sliceNumber){
						slice.putPixelValue((int)((p.get(0) / voxelSizeX) + originIndexX), (int)((p.get(1) / voxelSizeY) + originIndexY), splineNum+1);
					}
				}
			}
		}
		Grid2D grid = new Grid2D((float[]) slice.getPixels(), slice.getWidth(), slice.getHeight());
		imageBuffer.add(grid, sliceNumber);
	}

	@Override
	public String getProcessName() {
		return "Surface BSpline Phantom";
	}

	@Override
	public String getBibtexCitation() {
		return "see medline";
	}

	@Override
	public String getMedlineCitation() {
		return "L. Piegl and W. Tiller. The NURBS Book. Second Edition. Springer, Berlin, Heidelberg, New York. 1997.";
	}

	private void initBounds(){
		localBounds = new ArrayList<SimpleVector>();
		for (int i =0 ; i < list.size(); i++){
			SurfaceBSpline spline = list.get(i);
			SimpleVector bounds = SimpleOperators.concatenateVertically(spline.getMin().getAbstractVector(), spline.getMax().getAbstractVector());
			//System.out.println(bounds);
			localBounds.add(bounds);
		}
		SimpleVector max = SimpleOperators.max((SimpleVector []) localBounds.toArray(new SimpleVector [localBounds.size()])).getSubVec(3, 3);
		SimpleVector min = SimpleOperators.min((SimpleVector []) localBounds.toArray(new SimpleVector [localBounds.size()])).getSubVec(0, 3);

		bounds = SimpleOperators.concatenateVertically(min, max);		
	}

	public void readSplineListFromFile(String filename) throws IOException{
		list = SurfaceBSpline.readSplinesFromFile(filename);
		initBounds();
	}

	public void setSplineList(ArrayList<SurfaceBSpline> list){
		this.list = list;
		initBounds();
	}

	public void resizeVolumeToMatchBounds(PointND min, PointND max){
		bounds = SimpleOperators.concatenateVertically(min.getAbstractVector(), max.getAbstractVector());
		double rangeX = (max.getAbstractVector().getElement(0) - min.getAbstractVector().getElement(0))*1.05;
		double rangeY = (max.getAbstractVector().getElement(1) - min.getAbstractVector().getElement(1))*1.05;
		double rangeZ = (max.getAbstractVector().getElement(2) - min.getAbstractVector().getElement(2))*1.05;
		//System.out.println("Range Z: " + rangeZ);
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		double voxelSizeX = rangeX / ((double) traj.getReconDimensionX());
		double voxelSizeY = rangeY / ((double) traj.getReconDimensionY());
		double voxelSizeZ = rangeZ / ((double) traj.getReconDimensionZ());
		double shiftX = 0;
		double shiftY = 0;
		double shiftZ = 0;
		if (preserveAspects) {
			double maxRatio = voxelSizeX;
			if (maxRatio < voxelSizeY) maxRatio = voxelSizeY;
			if (maxRatio < voxelSizeZ) maxRatio = voxelSizeZ;
			// size of object in voxel space
			double sizeX = rangeX / maxRatio;
			double sizeY = rangeY / maxRatio;
			double sizeZ = rangeZ / maxRatio;
			voxelSizeX = maxRatio;
			voxelSizeY = maxRatio;
			voxelSizeZ = maxRatio;
			shiftX = (traj.getReconDimensionX() - sizeX) / 2;
			shiftY = (traj.getReconDimensionY() - sizeY) / 2;
			shiftZ = (traj.getReconDimensionZ() - sizeZ) / 2;
		}
		traj.setOriginInPixelsX(shiftX-(min.getAbstractVector().getElement(0) / voxelSizeX));
		traj.setOriginInPixelsY(shiftY-(min.getAbstractVector().getElement(1) / voxelSizeY));
		traj.setOriginInPixelsZ(shiftZ-(min.getAbstractVector().getElement(2) / voxelSizeZ));
		traj.setVoxelSpacingX(voxelSizeZ);
		traj.setVoxelSpacingY(voxelSizeY);
		traj.setVoxelSpacingZ(voxelSizeX);
	}

	public void setBoundsFromGeometry(AnalyticPhantom4D phantom4d){
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		SimpleVector minVec = new SimpleVector(traj.getOriginX(), traj.getOriginY(), traj.getOriginZ());
		String key = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.RENDER_PHANTOM_VOLUME_AUTO_CENTER);
		boolean set= false;
		if (key != null){
			if (key.equals("true")) {
				traj.setOriginToPhantomCenter(phantom4d);
				set=true;	
			}
		} 
		if (!set) {
			SimpleVector max = new SimpleVector(traj.getVoxelSpacingX()*traj.getReconDimensionX(),traj.getVoxelSpacingY()*traj.getReconDimensionY(), traj.getVoxelSpacingZ()*traj.getReconDimensionZ());
			max.add(minVec);
			bounds = SimpleOperators.concatenateVertically(minVec, max);
		}
	}

	public void resizeVolumeToMatchSplineSpace(){
		double rangeX = (bounds.getElement(3) - bounds.getElement(0))*1.05;
		double rangeY = (bounds.getElement(4) - bounds.getElement(1))*1.05;
		double rangeZ = (bounds.getElement(5) - bounds.getElement(2))*1.05;
		//System.out.println("Range Z: " + rangeZ);
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		double voxelSizeX = rangeX / ((double) traj.getReconDimensionX());
		double voxelSizeY = rangeY / ((double) traj.getReconDimensionY());
		double voxelSizeZ = rangeZ / ((double) traj.getReconDimensionZ());
		double shiftX = 0;
		double shiftY = 0;
		double shiftZ = 0;
		if (preserveAspects) {
			double maxRatio = voxelSizeX;
			if (maxRatio < voxelSizeY) maxRatio = voxelSizeY;
			if (maxRatio < voxelSizeZ) maxRatio = voxelSizeZ;
			// size of object in voxel space
			double sizeX = rangeX / maxRatio;
			double sizeY = rangeY / maxRatio;
			double sizeZ = rangeZ / maxRatio;
			voxelSizeX = maxRatio;
			voxelSizeY = maxRatio;
			voxelSizeZ = maxRatio;
			shiftX = (traj.getReconDimensionX() - sizeX) / 2;
			shiftY = (traj.getReconDimensionY() - sizeY) / 2;
			shiftZ = (traj.getReconDimensionZ() - sizeZ) / 2;
		}
		traj.setOriginInPixelsX(shiftX - (bounds.getElement(0) / voxelSizeX));
		traj.setOriginInPixelsY(shiftY - (bounds.getElement(1) / voxelSizeY));
		traj.setOriginInPixelsZ(shiftZ - (bounds.getElement(2) / voxelSizeZ));
		traj.setVoxelSpacingX(voxelSizeZ);
		traj.setVoxelSpacingY(voxelSizeY);
		traj.setVoxelSpacingZ(voxelSizeX);
	}

	public ArrayList<AbstractShape> tesselateSplines(double samplingU, double samplingV){
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		ArrayList<AbstractShape> meshList = new ArrayList<AbstractShape>();
		int numSlices = traj.getReconDimensionZ();
		voxelList = new ArrayList<Iterator<VectorPoint3D>>();
		ArrayList<ArrayList<VectorPoint3D>> pointList = new ArrayList<ArrayList<VectorPoint3D>>();
		for (int i = 0; i < numSlices; i++){
			pointList.add(new ArrayList<VectorPoint3D>());
		}	
		double voxelSizeX = traj.getVoxelSpacingX();
		double voxelSizeY = traj.getVoxelSpacingY();
		double voxelSizeZ = traj.getVoxelSpacingZ();
		double originIndexX = traj.getOriginInPixelsX();
		double originIndexY = traj.getOriginInPixelsY();
		double originIndexZ = traj.getOriginInPixelsZ();
		SimpleVector scaling = new SimpleVector(1.0 / voxelSizeX, 1.0 / voxelSizeY, 1.0 / voxelSizeZ);
		SimpleVector shift = new SimpleVector(originIndexX, originIndexY, originIndexZ);
		// Voxelize surface
		double samplingFactorU = samplingU / traj.getReconDimensionX();
		double samplingFactorV = samplingV / traj.getReconDimensionY();
		for (double splineNum = 0; splineNum < list.size(); splineNum++) {
			SimpleVector bound = localBounds.get((int) splineNum);
			double rangeX = (bound.getElement(3) - bound.getElement(0)) / voxelSizeX;
			double rangeY = (bound.getElement(4) - bound.getElement(1)) / voxelSizeY;
			double rangeZ = (bound.getElement(5) - bound.getElement(2)) / voxelSizeZ;
			int maxRange = (int) Math.ceil(Math.max(Math.max(rangeX, rangeY), rangeZ));
			samplingU = ((int)(maxRange * samplingFactorU));
			samplingV = ((int)(maxRange * samplingFactorV));
			// Old rendering (Sampline of the surface at a fixed grid):

			if (showStatus) IJ.showStatus("Sampling Shape " + list.get((int) splineNum).getTitle() + " (dim "+ (int)rangeX + "x"+ (int)rangeY + "x" + (int)rangeZ +") with " + samplingU + "x" +samplingV + " grid");
			//System.out.println("Sampling Shape " + list.get((int) splineNum).getTitle());
			double u = splineNum/list.size();
			if (showStatus) IJ.showProgress(u);

			if (showVertices) {
				for(double i =0 ; i < samplingU; i++){
					for (double j = 0; j < samplingV; j++){
						PointND p = list.get((int)splineNum).evaluate(i/samplingU, j/ samplingV);
						p.getAbstractVector().multiplyElementWiseBy(scaling);
						p.getAbstractVector().add(shift);
						int zPosition = (int) p.getAbstractVector().getElement(2);
						if (zPosition >= 0 && zPosition < numSlices){
							ArrayList<VectorPoint3D> list = pointList.get(zPosition);
							list.add(new VectorPoint3D(p, splineNum+1));
						}
					}
				}
			}
			AbstractShape mesh = list.get((int)splineNum).tessellateMesh(samplingU, samplingV);
			//System.out.println("Building done");
			meshList.add(mesh);
		}
		if (showStatus) IJ.showProgress(1.0);
		// prepare iterators for rendering.
		for (int i = 0; i < numSlices; i++){
			voxelList.add(Collections.synchronizedList(pointList.get(i)).iterator());
		}
		return meshList;
	}

	public void setScene(AbstractScene scene){
		this.scene = scene;
	}


	public void generateDefaultScene(double samplingU, double samplingV) throws IOException{
		scene = new PrioritizableScene();
		ArrayList<AbstractShape> meshList = tesselateSplines(samplingU, samplingV);
		for (int i = 0; i<meshList.size(); i++){
			AbstractShape mesh = meshList.get(i);
			PhysicalObject obj = new PhysicalObject();
			obj.setNameString(list.get((int)i).getTitle());
			obj.setMaterial(new Material(i+1));
			obj.setShape(mesh);
			((PrioritizableScene)scene).add(obj, PrioritizableScene.ADD_LOWEST_PRIORITY);
			if (showRaster) {
				hullPoints = mesh.getRasterPoints(200000);
			}
		}
	}

	public void configure() throws Exception{
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		renderAttenuation = UserUtil.queryBoolean("Render energy dependent attenuation?");
		if(renderAttenuation)
			xrayEnergy = UserUtil.queryDouble("Monochromatic Xray energy [keV]", xrayEnergy);
		GenericDialog gd = new GenericDialog("Configure Surface BSpline Volume");
		int width = traj.getReconDimensionX();
		int height = traj.getReconDimensionY();
		gd.addSlider("Sampling in u direction", 10, width, width/2);
		gd.addSlider("Sampling in v direction", 10, height, height/2);
		gd.showDialog();
		if (gd.wasCanceled()){
			throw new RuntimeException("User cancelled selection.");
		}
		double samplingU = gd.getNextNumber();
		double samplingV = gd.getNextNumber();
		String filename = FileUtil.myFileChoose(".nrb", false);
		readSplineListFromFile(filename);
		generateDefaultScene(samplingU, samplingV);
		super.configure();
	}

	@Override
	public SliceWorker clone() {
		SurfaceBSplineVolumePhantom clone = new SurfaceBSplineVolumePhantom();
		clone.list = list;
		clone.voxelList = voxelList;
		clone.scene = scene;
		clone.hullPoints = hullPoints;
		clone.showStatus = showStatus;
		clone.renderAttenuation = renderAttenuation;
		clone.xrayEnergy = xrayEnergy;
		return clone;
	}

	/**
	 * Returns an SimpleVector that specifies the bounding box of the BSpline Phantom with six entries:<BR>
	 * <li>Minimum X Coordinate</li>
	 * <li>Minimum Y Coordinate</li>
	 * <li>Minimum Z Coordinate</li>
	 * <li>Maximum X Coordinate</li>
	 * <li>Maximum Y Coordinate</li>
	 * <li>Maximum Z Coordinate</li>
	 * @return the bounding box vector
	 */
	public SimpleVector getBounds(){
		return bounds;
	}

	
	public void setXrayEnergy(double xrayEnergy) {
		this.xrayEnergy = xrayEnergy;
	}
	
	public void setRenderAttenuation(boolean renderAttenuation) {
		this.renderAttenuation = renderAttenuation;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */