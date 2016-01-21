/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.MotionUtil;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSplineVolumePhantom;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.renderer.MetricPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.xcat.SquatScene;
import edu.stanford.rsl.conrad.phantom.xcat.XCatScene;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class XCatMetricPhantomCreator {

	private int steps = 2;

	/**
	 * Method to render 3-D versions of the phantoms.
	 * @param args the arguments (none)
	 */
	public static void main(String [] args){

		CONRAD.setup();
		XCatMetricPhantomCreator creator = new XCatMetricPhantomCreator();
		AnalyticPhantom4D scene = creator.instantiateScene();
		ImagePlus hyper = creator.renderMetricVolumePhantom(scene);
		hyper.show();
	}

	public void renderMetricVolumePhantom(){
		renderMetricVolumePhantom(null);
	}

	public AnalyticPhantom4D instantiateScene(){

		try {
			steps = UserUtil.queryInt("Enter number of steps for sampling:", steps);
		} catch (Exception e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		
		AnalyticPhantom4D scene = null;
		boolean filenameSet = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION) != null;
		//select the scene
		if (filenameSet){
			scene = (AnalyticPhantom4D) MotionUtil.get4DSpline();
		}


		if(scene == null){
			try {
				scene = (AnalyticPhantom4D) UserUtil.queryPhantom("Scene Selection", "Please select a 4D phantom:");
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			try {
				scene.configure();
				boolean write = true;
				if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION) == null) write = false;
				if (write) {
					FileOutputStream fos = new FileOutputStream(Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION));
					ObjectOutputStream oos = new ObjectOutputStream(fos);
					oos.writeObject(scene);
					oos.flush();
					oos.close();
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return scene;
	}

	public ImagePlus renderMetricVolumePhantom(AnalyticPhantom4D scene){
		try {
			// read global configuration
			Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
			int dimz = geom.getReconDimensionZ();
			int dimx = geom.getReconDimensionX();
			int dimy = geom.getReconDimensionY();
			// create hyperstack (4D volume), 3D volume that has time line
			ImageStack hyperStack = new ImageStack(dimx, dimy);


			//scene.setTimeWarper(new HarmonicTimeWarper(1));
			ArrayList<SurfaceBSpline> list = null;//scene.getSplines();
			if (scene instanceof XCatScene) {
				list = ((XCatScene)scene).getSplines();
			}
			Calibration calibration = null;

			PointND saveWorldOrigin = new PointND(geom.getOriginX(),geom.getOriginY(),geom.getOriginZ());

			// render the scene
			// the number of steps gives the number of volumes rendered. (k < 40)

			double[] timeList = new double[getSteps()];
			for (int i = 0; i < timeList.length; i++) {
				double steps = (getSteps()==1)?1.0:((double)(getSteps()-1));
				timeList[i]=((double)i)/steps;
			}


			for (int k = 0; k < getSteps(); k++){
				ImageGridBuffer buffer = new ImageGridBuffer();
				IJ.showStatus("Rendering State " + k);
				IJ.showProgress(((double)k)/getSteps());

				// configure worker
				// worker: renders the scene, it basically cast a ray thru the scene for every pixel for every projection
				SurfaceBSplineVolumePhantom phantomWorker = new SurfaceBSplineVolumePhantom();
				phantomWorker.setImageProcessorBuffer(buffer);
				phantomWorker.setShowStatus(true);
				boolean resize = true;
				if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_AUTO_RESIZE) != null){
					if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_AUTO_RESIZE).equals("false")) {
						resize = false;
					} else {
						// set scene boundaries to recon volume.
						// Thus we have to create a new origin location that centers the recon volume around the scene volume without changing the voxel spacing.
						PointND origin = new PointND(scene.getMax());
						origin.getAbstractVector().subtract(scene.getMin().getAbstractVector());
						origin.getAbstractVector().divideBy(2.0);
						origin.getAbstractVector().add(scene.getMin().getAbstractVector());
						PointND maximumCoordinate = new PointND(Configuration.getGlobalConfiguration().getGeometry().getReconVoxelSizes()[0]*Configuration.getGlobalConfiguration().getGeometry().getReconDimensionX(),
								Configuration.getGlobalConfiguration().getGeometry().getReconVoxelSizes()[1]*Configuration.getGlobalConfiguration().getGeometry().getReconDimensionY(),
								Configuration.getGlobalConfiguration().getGeometry().getReconVoxelSizes()[2]*Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ());
						SimpleVector reg = maximumCoordinate.getAbstractVector().dividedBy(2.0);
						origin.getAbstractVector().subtract(reg);
						maximumCoordinate.getAbstractVector().add(origin.getAbstractVector());
						Configuration.getGlobalConfiguration().getGeometry().setOriginInWorld(origin);
						scene.setMin(origin);
						scene.setMax(maximumCoordinate);
					}
				}
				if (list != null && list.size() > 0) {
					phantomWorker.setSplineList(list);
				}

				/**
				 * Time calculation
				 * 
				 * * .4 => Inspriration only
				 */
				double time = 0;
				if (timeList == null){
					time= (k/(double)getSteps());
				}
				else{
					time = timeList[k];
				}
				System.out.println("Current timepoint: "+ time);


				if (scene instanceof SquatScene){
					// Update Origin according to the knee center point
					/*
					PointND centeredOrigin = new PointND(269.3, 287.2, -638.3);
					PointND origin = new PointND(0,0,0);
					for (int i = 0; i < geom.getReconDimensions().length; i++) {
						double size = geom.getReconDimensions()[i]*geom.getReconVoxelSizes()[i];
						origin.set(i, centeredOrigin.get(i)-(size/2.0-0.5));
					}



					PointND leftTibiaTop = new PointND(7.6, 287.2,-638.3);
					PointND rightTibiaTop = new PointND(269.3, 287.2, -638.3);
					SimpleVector centerOffset = SimpleOperators.subtract(rightTibiaTop.getAbstractVector(), leftTibiaTop.getAbstractVector()).dividedBy(2.0);
					PointND initialOrigin = new PointND(SimpleOperators.add(leftTibiaTop.getAbstractVector(), centerOffset));
					 */
					double yoffset = 50;
					PointND rightTibiaTop = new PointND(269.3, 287.2, -638.3);
					PointND initialOrigin = rightTibiaTop;

					PointND currentOrigin = scene.getPosition(initialOrigin, 0, time);
					// update the origin, i.e. the volume position to the current knee position
					System.out.println("XCat centered origin: " + currentOrigin);

					PointND currentOriginCorner = currentOrigin.clone();
					for (int i = 0; i < geom.getReconDimensions().length; i++) {
						double size = geom.getReconDimensions()[i]*geom.getReconVoxelSizes()[i];
						currentOriginCorner.set(i, currentOrigin.get(i)-(size/2.0-0.5));
					}
					currentOriginCorner.set(1, currentOriginCorner.get(1)+yoffset);
					geom.setOriginInWorld(currentOriginCorner);



					System.out.println("Voxel Coordinate System centered origin: " +  new SimpleVector(General.worldToVoxel(currentOrigin.getCoordinates(), geom.getReconVoxelSizes(), currentOriginCorner.getCoordinates())));
					System.out.println("Voxel Coordinate System moving origin: " +  new SimpleVector(General.worldToVoxel(rightTibiaTop.getCoordinates(), geom.getReconVoxelSizes(), currentOriginCorner.getCoordinates())));
					System.out.println(" ");
					System.out.println(" ");
					//SimpleOperators.subtract(currentOrigin.getAbstractVector(),initialOrigin.getAbstractVector()));


					/*
					PointND leftFemurTop = new PointND(16.6, 273.6,-207.42);
					/*PointND rightFemurTop = new PointND(269.3, 273.6,-207.42);
					SimpleVector rotCenterOffset = SimpleOperators.subtract(rightFemurTop.getAbstractVector(), leftFemurTop.getAbstractVector()).dividedBy(2.0);
					PointND rotCenter = new PointND(SimpleOperators.add(leftFemurTop.getAbstractVector(), rotCenterOffset));

					double lowerZoffset = 0;
					PointND leftTibiaTop = new PointND(7.6, 287.2,-638.3+lowerZoffset);
					PointND rightTibiaTop = new PointND(269.3, 287.2, -638.3+lowerZoffset);
					SimpleVector centerOffset = SimpleOperators.subtract(rightTibiaTop.getAbstractVector(), leftTibiaTop.getAbstractVector()).dividedBy(2.0);
					PointND initialOrigin = new PointND(SimpleOperators.add(leftTibiaTop.getAbstractVector(), centerOffset));



					MotionField rotMot = new RotationMotionField(leftFemurTop , new SimpleVector(1,0,0), General.toRadians(0));
					PointND movingOrigin = rotMot.getPosition(initialOrigin, 0, time);
					PointND currentOrigin = new PointND(new double[movingOrigin.getDimension()]);

					// determine the current coordinate system origin by the moving center point
					for (int i = 0; i < geom.getReconDimensions().length; i++) {
						double size = geom.getReconDimensions()[i]*geom.getReconVoxelSizes()[i];
						currentOrigin.set(i, movingOrigin.get(i)-(size/2.0-0.5));
					}
					System.out.println(currentOrigin);
					// update the origin, i.e. the volume position to the current knee position
					geom.setOriginInWorld(currentOrigin);*/

				}



				if (resize){
					if (scene.getMin() == null){
						phantomWorker.resizeVolumeToMatchSplineSpace();
					} else {
						phantomWorker.resizeVolumeToMatchBounds(scene.getMin(), scene.getMax());
					}
				} else {
					phantomWorker.setBoundsFromGeometry(scene);
				}


				// tessellate scene -> convert to triangles -> render the triangles
				PrioritizableScene current = scene.getScene(time);

				phantomWorker.setScene(current);

				// create renderer
				MetricPhantomRenderer phantom = new MetricPhantomRenderer();
				ArrayList<Integer> processors = new ArrayList<Integer>();
				for (int i = 0; i < dimz; i++){
					processors.add(new Integer(i));
				}
				phantomWorker.setSliceList(Collections.synchronizedList(processors).iterator());

				// render
				phantom.setModelWorker(phantomWorker);
				phantom.createPhantom();
				ImagePlus renderedBSpline = buffer.toImagePlus("State "+k);
				calibration = renderedBSpline.getCalibration();

				// the volume to the hyperstack
				for (int i=1; i<= dimz; i++){
					hyperStack.addSlice("Slice z = " +(i-1) + " t = " + k, renderedBSpline.getStack().getProcessor(i));
				}

			}



			calibration.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
			calibration.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
			calibration.zOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
			calibration.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
			calibration.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
			calibration.pixelDepth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
			// finalize the hyperstack
			ImagePlus hyper = new ImagePlus();
			hyper.setCalibration(calibration);
			hyper.setStack(scene.getName(), hyperStack);
			hyper.setDimensions(1, dimz, getSteps());
			hyper.setOpenAsHyperStack(true);
			IJ.showProgress(1.0);

			//geom.setOriginInWorld(saveWorldOrigin);

			// display the result.
			return hyper;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * @return the steps
	 */
	public int getSteps() {
		return steps;
	}

	/**
	 * @param steps the steps to set
	 */
	public void setSteps(int steps) {
		this.steps = steps;
	}
}
