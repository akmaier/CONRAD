/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.phantom.workers;

import java.util.ArrayList;
import java.util.Iterator;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.AbstractRayTracer;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * <p>Projects arbitrarily defined phantoms to a detector using ray casting.<br/>
 * The pixel value on the detector is determined by the absorption model.<BR><BR>
 * <b>If you change anything in this class, notify the conrad-dev mailing list.</b>
 * 
 * @author Rotimi X Ojo
 * @author Andreas Maier
 * 
 */
public class AnalyticPhantomProjectorWorker extends SliceWorker {

	protected AnalyticPhantom phantom;
	protected Trajectory trajectory = Configuration.getGlobalConfiguration().getGeometry();
	protected XRayDetector detector;
	private PriorityRayTracer rayTracer = new PriorityRayTracer();
	private static boolean accurate = false;


	// First rule of optimization is: Don't optimize!
	//Start for speed up
	//private StraightLine castLine = new StraightLine(new PointND(0,0,0),new SimpleVector(0,0,0));
	//private SimpleVector pixel = new SimpleVector(0, 0);
	//private ArrayList<PhysicalObject> segmentsBuff = new ArrayList<PhysicalObject> (1);
	protected PhysicalObject environment = new PhysicalObject();
	//private PointND startPoint = new PointND(0,0);
	//private PointND endPoint = new PointND(0,0);
	//private Edge environmentEdge = new Edge(startPoint, endPoint);
	//private PointND raySource = new PointND(0,0,0);
	//End for speed up
	
	/**
	 * Allow deriving classes to change the ray tracer
	 * @return instance of a ray tracer
	 */
	protected AbstractRayTracer getRayTracer() {
		return rayTracer;
	}
	

	@Override
	public void workOnSlice(int sliceNumber) {
		PrioritizableScene phantomScene = phantom;
		if (phantom instanceof AnalyticPhantom4D) {
			AnalyticPhantom4D scene4D = (AnalyticPhantom4D) phantom;
			phantomScene = scene4D.getScene(((double) sliceNumber) / trajectory.getProjectionStackSize());

			
			String disableAutoCenterBoolean = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.DISABLE_CENTERING_4DPHANTOM_PROJECTION_RENDERING);
			boolean disableAutoCenter = false;
			if (disableAutoCenterBoolean != null){
				disableAutoCenter = Boolean.parseBoolean(disableAutoCenterBoolean);
			}
			
			Translation centerTranslation = new Translation(new SimpleVector(0,0,0));
			if (!disableAutoCenter){
				SimpleVector center = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);
				centerTranslation = new Translation(center.negated());
			}
			
			for (PhysicalObject o:phantomScene){
				o.getShape().applyTransform(centerTranslation);
				//System.out.println(o.getShape().getMax() + " " + o.getShape().getMin());

				//Translate a part of XCAT to the center of source & detector for 2D projection (e.g. knee at the center of the 2d projection) 
				String translationString = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING);
				if (translationString != null){
					// Center b/w RKJC & LKJC: -292.6426  211.7856  440.7783 (subj 5, static60),-401.1700  165.9885  478.5600 (subj 2, static60)
					// XCAT Center by min & max: -177.73999504606988, 179.8512744259873, 312.19713254613583
					// translationVector = (XCAT Center by min & max) - (Center b/w RKJC & LKJC)=>
					// 114.9026, -31.9343, -128.5811 (subj5),  120, 3, -110(subj2) Try 114.0568    2.4778 -106.2550
					String [] values = translationString.split(", ");
					SimpleVector translationVector = new SimpleVector(Double.parseDouble(values[0]), Double.parseDouble(values[1]), Double.parseDouble(values[2]));
					Translation translationToRotationCenter = new Translation(translationVector);
					o.getShape().applyTransform(translationToRotationCenter);
				}				
			}
			//System.out.println(phantomScene.getMax() + " " + phantomScene.getMin());
		}
		Grid2D slice = raytraceScene(phantomScene, trajectory.getProjectionMatrix(sliceNumber));
		this.imageBuffer.add(slice, sliceNumber);
	}		

	public Grid2D raytraceScene(PrioritizableScene phantomScene, Projection projection){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		//Grid2D slice = new Grid2D(geom.getDetectorWidth(), geom.getDetectorHeight());
		Grid2D slice = detector.createDetectorGrid(geom.getDetectorWidth(), geom.getDetectorHeight(), projection);
		getRayTracer().setScene(phantomScene);
		// Second rule of optimization is: Optimize later.
		PointND raySource = new PointND(0,0,0);
		raySource.setCoordinates(projection.computeCameraCenter());
		StraightLine castLine = new StraightLine(raySource,new SimpleVector(0,0,0));

		SimpleVector centerPixDir = null;
		if (accurate){
			centerPixDir = projection.computePrincipalAxis();
		}
		//SimpleVector prinpoint = trajectory.getProjectionMatrix(sliceNumber).getPrincipalPoint();	

		double xcorr = 0;//trajectory.getDetectorWidth()/2 - prinpoint.getElement(0);
		double ycorr = 0;//trajectory.getDetectorHeight()/2 - prinpoint.getElement(1);

		double length = trajectory.getSourceToDetectorDistance();
		Edge environmentEdge = new Edge(new PointND(0), new PointND(length));


		ArrayList<PhysicalObject> fallBackBackground = new ArrayList<PhysicalObject> (1);
		SimpleVector pixel = new SimpleVector(0, 0);
		boolean negate = false;
		for(int y = 0; y < trajectory.getDetectorHeight(); y++){
			for(int x = 0; x < trajectory.getDetectorWidth();x++){
				pixel.setElementValue(0, x-xcorr);
				pixel.setElementValue(1, y-ycorr);
				SimpleVector dir = projection.computeRayDirection(pixel);
				if ((y==0) && (x==0)) {
					//Check that ray direction is towards origin.
					double max = 0;
					int index = 0;
					for (int i=0; i < 3; i++){
						if (Math.abs(dir.getElement(i)) > max) {
							max = Math.abs(dir.getElement(i));
							index = i;
						}
					}
					double t = - raySource.get(index) / dir.getElement(index);
					if (t < 0) negate = true; 
				}
				if (negate) dir.negate();
				castLine.setDirection(dir);
				
				ArrayList<PhysicalObject> segments = getRayTracer().castRay(castLine);
				if (accurate){
					double dirCosine = SimpleOperators.multiplyInnerProd(centerPixDir,dir);
					length = trajectory.getSourceToDetectorDistance()/dirCosine;					
				}

				if(segments == null){
					fallBackBackground.clear();
					segments = fallBackBackground;
				} else {				
					if (accurate) {
						environmentEdge = new Edge(new PointND(0), new PointND(length - getTotalSegmentsLength(segments)));
					}
				}
				environment.setShape(environmentEdge);
				segments.add(environment);
				// old code:
				// double integral = absorptionModel.evaluateLineIntegral(segments);
				// slice.putPixelValue(x, y, integral);
				
				detector.writeToDetector(slice, x, y, segments);
				
			}
		}
		return slice;
	}


	private double getTotalSegmentsLength(ArrayList<PhysicalObject> segments) {
		double sum = 0;
		Iterator<PhysicalObject> it = segments.iterator();
		while(it.hasNext()){
			//sum+=((Edge) it.next().getShape()).getLength();			
		}
		return sum;
	}




	public SliceWorker clone() {
		AnalyticPhantomProjectorWorker newRend = new AnalyticPhantomProjectorWorker();
		newRend.phantom = phantom;
		newRend.detector  = detector;
		newRend.environment.setMaterial(phantom.getBackgroundMaterial());
		newRend.rayTracer.setScene((PrioritizableScene)phantom);
		return newRend;
	}


	@Override
	public void configure() throws Exception {
		// We will now read this from the Configuration. The detector needs to be preconfigured!
		//detector = (XRayDetector) UserUtil.queryObject("Select Detector:", "Detector Selection", XRayDetector.class);
		//detector.configure();
		detector = Configuration.getGlobalConfiguration().getDetector();
		detector.init();
		phantom = UserUtil.queryPhantom("Select Phantom", "Select Phantom");
		Material mat = null;

		do{
			String materialstr = UserUtil.queryString("Enter Background Medium:", "vacuum");
			mat = MaterialsDB.getMaterialWithName(materialstr);
		} while(mat == null);



		phantom.setBackground(mat);
		phantom.configure();

		getRayTracer().setScene((PrioritizableScene)phantom);
		environment.setMaterial(phantom.getBackgroundMaterial());
		super.configure();
	}

	public void configure(AnalyticPhantom phan, XRayDetector detector) throws Exception {
		this.detector  = detector;
		this.detector.init();
		phantom = phan;
		if (phantom.getBackgroundMaterial()==null){
			Material mat = null;
			String materialstr = "vacuum";
			mat = MaterialsDB.getMaterialWithName(materialstr);
			phantom.setBackground(mat);
		}
		getRayTracer().setScene((PrioritizableScene)phantom);
		environment.setMaterial(phantom.getBackgroundMaterial());
		super.configure();
	}

	@Override
	public String getProcessName() {
		return "Generic Phantom Projector";
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	/**
	 * @return the absorptionModel
	 */
	public XRayDetector getDetector() {
		return detector;
	}

	/**
	 * @param absorptionModel the absorptionModel to set
	 */
	public void setDetector(XRayDetector detector) {
		this.detector = detector;
	}

}
