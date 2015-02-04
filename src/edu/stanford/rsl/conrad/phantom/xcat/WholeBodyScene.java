package edu.stanford.rsl.conrad.phantom.xcat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.PointBasedMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Simple scene for XCat to display the whole body. This scene does not define any motion.
 * 
 * @author akmaier
 *
 */
public class WholeBodyScene extends XCatScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7514545759333848517L;
	protected boolean autoResize = false;
	public WholeBodyScene (){
		warper = new IdentityTimeWarper();
	}

	@Override
	public void configure() throws Exception{
		setName("Whole Body Scene");
		init();
		createPhysicalObjects();
	}



	public PrioritizableScene tessellateScene(double time){
		// remove previous state scene objects.
		PrioritizableScene scene = new PrioritizableScene();



		ArrayList<AbstractSurface> list = new ArrayList<AbstractSurface>();
		list.addAll(splines);
		list.addAll(variants);
		
		int numberOfThreads = Math.min(CONRAD.getNumberOfThreads(),list.size());
		TessellationThread[] threads = new TessellationThread[numberOfThreads];

		Iterator<?> iter = Collections.synchronizedList(list).iterator();
		for (int splineNum = 0; splineNum < numberOfThreads; splineNum++) {
			threads[splineNum] = new TessellationThread(iter, warper.warpTime(time));
		}
		ParallelThreadExecutor executor = new ParallelThreadExecutor(threads);
		try {
			executor.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		for (int splineNum = 0; splineNum < numberOfThreads; splineNum++) {
			for (AbstractShape s: threads[splineNum].getObjects()) {
				add(scene, s, s.getName());
			}
		}
		return scene;
	}

	/**
	 * Determines the bounds of the scene.
	 */
	protected void updateSceneLimits(){
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		int splineNum = splines.size();
		for (int i=splineNum-1; i >= 0; i--){
			if ((getSplineNameMaterialNameLUT().get(splines.get(i).getTitle()) != null) &&
					(getSplinePriorityLUT().get(splines.get(i).getTitle()) != null)) {
				max.updateIfHigher(splines.get(i).getMax());
				min.updateIfLower(splines.get(i).getMin());
			} else {
				splines.remove(i);
			}
		}
	}
	
	/**
	 * Determines the bounds of the scene.
	 */
	protected void init(){
		init(true);
	}

	/**
	 * Determines the bounds of the scene.
	 */
	protected void init(boolean keepIntersectingSplines){

		splines = readSplines();
		if (max == null){ 
			updateSceneLimits();
		}

		// remove splines outside the volume of interest using z axis
		int splineNum = splines.size();
		if (keepIntersectingSplines) {
			for (int i=splineNum-1; i >= 0; i--){
				if (splines.get(i).getMax().get(2) 
						< min.get(2)){// below VOI
					splines.remove(i);
				} else {
					if (splines.get(i).getMin().get(2) > max.get(2)) { // above VOI
						splines.remove(i);
					}
				}
			}
		}
		else{
			// remove splines outside the volume of interest using z axis
			for (int i=splineNum-1; i >= 0; i--){
				if (splines.get(i).getMax().get(2) 
						> max.get(2)){// below VOI
					splines.remove(i);
				} else {
					if (splines.get(i).getMin().get(2) < min.get(2)) { // above VOI
						splines.remove(i);
					}
				}
			}
		}
	}

	/**
	 * Reads XCat from file.
	 * @return a list of splines
	 */
	public ArrayList<SurfaceBSpline> readSplines(){
		try {
			String filename = "mtorso";
			if (!maleGender) {
				filename = "ftorso";
			}
			return SurfaceBSpline.readSplinesFromFile(XCatDirectory + "/"+ filename + ".nrb");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Creates a motion field from the scene motion plus a sequence of points in time.
	 * @param numberOfBSplineTimePoints
	 * @param additionalMotion
	 * @param context
	 * @return the motion field
	 */
	public MotionField getSceneMotion(int numberOfBSplineTimePoints, ArrayList<ArrayList<PointND>> additionalMotion, int context){
		for (TimeVariantSurfaceBSpline tSpline:variants){
			for (int t=0; t<numberOfBSplineTimePoints; t++){
				additionalMotion.get(t).addAll(tSpline.getControlPoints(t));
			}
		}
		PointND[][] array = new PointND[additionalMotion.size()][additionalMotion.get(0).size()];
		for (int t = 0; t < additionalMotion.size(); t++){
			array[t] = additionalMotion.get(t).toArray(array[t]);
		}
		MotionField motion = new PointBasedMotionField(array, context);
		return motion;
	}

	/**
	 * Creates a motion field from a sequence of points in time.
	 * @param numberOfBSplineTimePoints
	 * @param additionalMotion
	 * @param context 
	 * @return the motion field
	 */
	public MotionField getCompoundMotion(int numberOfBSplineTimePoints, ArrayList<ArrayList<PointND>> additionalMotion, int context){
		PointND[][] array = new PointND[additionalMotion.size()][additionalMotion.get(0).size()];
		for (int t = 0; t < additionalMotion.size(); t++){
			array[t] = additionalMotion.get(t).toArray(array[t]);
		}
		MotionField motion = new PointBasedMotionField(array, context);
		return motion;
	}


	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		return initialPosition;
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		ArrayList<PointND> newList = new ArrayList<PointND>();
		for (int i = 0; i<times.length; i++){
			newList.add(initialPosition);
		}
		return newList;
	}

	@Override
	public String getName() {
		return "XCat Whole Body";
	}

	@Override
	public String getBibtexCitation() {
		return "@article{Segars08-RCS,\n" +
				"  author={Segars WP, Mahesh M, Beck TJ, Frey EC, Tsui BMW},\n" +
				"  title={Realistic CT simulation using the 4D XCAT phantom},\n" +
				"  journal={Med. Phys.},\n" +
				"  volume={35},\n" +
				"  pages={3800-3808},\n" +
				"  year={2008}\n" +
				"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Segars WP, Mahesh M, Beck TJ, Frey EC, Tsui BMW. Realistic CT simulation using the 4D XCAT phantom. Med. Phys. 35:3800-3808 (2008);";
	}

	@Override
	public float[] getBinaryRepresentation() {
		// TODO Auto-generated method stub
		return null;
	}
}

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
