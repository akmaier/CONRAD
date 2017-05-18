package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.geometry.motion.*;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.NearestNeighborTimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * Class to simulate very simple knee joint motion in XCat.
 * Scene contains only the lower body as only the legs and hip are of interest for a squat.
 * 
 * @author jhchoi21 / berger / Oleksiy Rybakov
 *
 */
public class DynamicSquatScene extends WholeBodyScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5961188801136828415L;
	private ViconMarkerBuilder viconBuilder;

	boolean writeGroundTruthToFile = true;
	
	boolean buildStaticWithRef = false;
	
	boolean selectDefaultVICONMotion = true;
	
	int refProjection = 0;
	
	static final int subjectNumber = 2;
	
	public DynamicSquatScene (){	
	}	
	
	static final HashMap<Integer, SimpleVector> staticPhantomCenteringTranslations = initPhantomCenteringTranslations();

	private final static HashMap<Integer, SimpleVector> initPhantomCenteringTranslations(){
		// Center b/w RKJC & LKJC: -292.6426  211.7856  440.7783 (subj 5, static60),-401.1700  165.9885  478.5600 (subj 2, static60)
		// XCAT Center by min & max: -177.73999504606988, 179.8512744259873, 312.19713254613583
		// translationVector = (XCAT Center by min & max) - (Center b/w RKJC & LKJC)=>
		// 114.9026, -31.9343, -128.5811 (subj5),  120, 3, -110 (subj2) Try 114.0568, 2.4778, -106.2550
		HashMap<Integer, SimpleVector> map = new HashMap<Integer, SimpleVector>();
		map.put(2,new SimpleVector(120, 3, -110));
		map.put(5, new SimpleVector(114.9026, -31.9343, -128.5811));
		return map;
	}
	
	@Override
	public void configure(){
		
		// parameters for (possible) artificial motion field
		double lengthLowerLeft = 433.976;
		double lengthLowerRight = 434.196;
		double lengthUpperLeft = 443.947;
		double lengthUpperRight = 443.174;
		double numberOfSquats = 4;
					
		// Here we use the medical knee flexion angle (in degrees).
		double angleMin = 60;
		double angleMax = 90;
		
		try {
			buildStaticWithRef = UserUtil.queryBoolean("Build static version with reference frame?");
			if(buildStaticWithRef)
				refProjection = UserUtil.queryInt("Reference Projection Nr?", refProjection);
			writeGroundTruthToFile=UserUtil.queryBoolean("Write Ground Truth Motion to XML?");
			// parameter for choice of motion field (VICON marker motion or artificial motion field)
			selectDefaultVICONMotion = UserUtil.queryBoolean("Select pre-defined VICON MOTION");
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		warper = new IdentityTimeWarper();
		if(selectDefaultVICONMotion){
			viconBuilder = new ViconMarkerBuilder();
		} else {
			// parameters for artificial motion field
			lengthLowerLeft = 433.976;
			lengthLowerRight = 434.196;
			lengthUpperLeft = 443.947;
			lengthUpperRight = 443.174;
			numberOfSquats = 4;
			
			// Here we use the medical knee flexion angle (in degrees).
			angleMin = 60;
			angleMax = 90;
			
			try {
				numberOfSquats = UserUtil.queryDouble("Please select the number of squats during acquisition time.", numberOfSquats);
				angleMin = UserUtil.queryDouble("Please select the smallest knee flexion angle in degrees.", angleMin);
				angleMax = UserUtil.queryDouble("Please select the largest knee flexion angle in degrees.", angleMax);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		int numberOfBSplineTimePoints = Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices();

		// mm res of XCAT, center is somewhere in the hip
		min = new PointND(-300, 50, -1200);
		max = new PointND(370, 400, 500);

		setName("Dynamic Squat Scene");
		init();

		// create lists for the complete motion field.		
		ArrayList<ArrayList<PointND>> lowerBodyMotion = new ArrayList<ArrayList<PointND>>();
		for (int t=0; t<numberOfBSplineTimePoints; t++){
			lowerBodyMotion.add(new ArrayList<PointND>());
		}				

		String [] partsLeftLower = {
				"l_tibia", 
				"lefttibia",
				"l_foot",
				"l_fibula",
				"leftfibula"
		};

		String [] partsRightLower = {
				"r_tibia", 
				"righttibia",
				"r_foot",
				"r_fibula",
				"rightfibula"
		};

		String [] partsLeftUpper = {
				"l_femur", 
				"leftfemur",				
				"l_patella", 
				"leftpatella"
		};

		String [] partsRightUpper = {	
				"r_femur", 
				"rightfemur",
				"r_patella", 
				"rightpatella"
		};

		String [] partsLeftDual = {	
				"leftleg"
		};

		String [] partsRightDual = {	
				"rightleg"
		};

		VICONMarkerMotionField VICONMotion = null;
		ArtificialMotionField artificialMotion = null;
		String transformPart = null;
		// for each spline in the body
		for (int i = splines.size()-1; i >= 0; i--){

			// test whether the spline is constant, constant spline dose not move.
			boolean displayTest = false;			
			for (String s: partsLeftLower){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					displayTest = true;
					transformPart = "LeftLower";
				}				
			}		
			for (String s: partsRightLower){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){					
					displayTest = true;
					transformPart = "RightLower";
				}				
			}
			for (String s: partsLeftUpper){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){					
					displayTest = true;
					transformPart = "LeftUpper";
				}				
			}			
			for (String s: partsRightUpper){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					displayTest = true;
					transformPart = "RightUpper";
				}				
			}			
			for (String s: partsLeftDual){ // apply two different transform based on 3d coordinate (e.g. left leg)
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					displayTest = true;
					transformPart = "LeftDual";
				}				
			}
			for (String s: partsRightDual){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					displayTest = true;
					transformPart = "RightDual";
				}				
			}
			if (displayTest){
				// define motion around lower body bones.
				if(selectDefaultVICONMotion){
					VICONMotion = new VICONMarkerMotionField(transformPart, splines.get(i).getTitle(), viconBuilder, this.buildStaticWithRef, this.refProjection); 
					VICONMotion.setTimeWarper(warper);											
					// ?? time =0, data not inserted, thus numberOfBSplineTimePoints+1
					variants.add(new NearestNeighborTimeVariantSurfaceBSpline(splines.get(i), VICONMotion, numberOfBSplineTimePoints+1, false));
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());				
				} else {
					// creation of artificial motion field
					artificialMotion = new ArtificialMotionField(transformPart,
							splines.get(i).getTitle(), lengthLowerLeft, lengthLowerRight,
							lengthUpperLeft, lengthUpperRight, numberOfSquats, angleMin, angleMax); 
					artificialMotion.setTimeWarper(warper);
					
					// passing artificial motion field to time variant splines
					variants.add(new NearestNeighborTimeVariantSurfaceBSpline(splines.get(i),
							artificialMotion, numberOfBSplineTimePoints + 1, false));
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
				}
			} else {	
			}
			splines.remove(i);

		}	


		if (splines.size()<1){
			// intentionally inserted in case of no spline left, but variant
			ArrayList<PointND> tmpCtrlPnt = new ArrayList<PointND>(); 
			tmpCtrlPnt.add(new PointND(0, 0, 0));		
			splines.add(new SurfaceBSpline(tmpCtrlPnt, new SimpleVector(0, 1), new SimpleVector(0, 2)));
		}

		min = new PointND(Double.MAX_VALUE, Double.MIN_VALUE, Double.MAX_VALUE);
		max = new PointND(-Double.MAX_VALUE, -Double.MIN_VALUE, -Double.MAX_VALUE);
		
		createPhysicalObjects();

		min = new PointND(Double.MAX_VALUE, Double.MIN_VALUE, Double.MAX_VALUE);
		max = new PointND(-Double.MAX_VALUE, -Double.MIN_VALUE, -Double.MAX_VALUE);
		for (TimeVariantSurfaceBSpline spline: variants){
				min.updateIfLower(spline.getMin());
				max.updateIfHigher(spline.getMax());
		}
		
		// For rendering we only consider the reference time and extract the scene bound of that specific time frame
		PointND minSceneBound = new PointND(Double.MAX_VALUE, Double.MIN_VALUE, Double.MAX_VALUE);
		PointND maxSceneBound = new PointND(-Double.MAX_VALUE, -Double.MIN_VALUE, -Double.MAX_VALUE);
		for (TimeVariantSurfaceBSpline spline: variants){
			for(PointND pts : spline.getControlPoints(0)){
				minSceneBound.updateIfLower(pts);
				maxSceneBound.updateIfHigher(pts);
			}
		}
		
		SimpleVector dynamicCenterTranslation = SimpleOperators.add(minSceneBound.getAbstractVector(), maxSceneBound.getAbstractVector()).dividedBy(2).negated();
		Translation dct = new Translation(dynamicCenterTranslation);
		Translation staticTranslation = new Translation(staticPhantomCenteringTranslations.get(subjectNumber));
		
		Iterator<PhysicalObject> it = this.iterator();
		while (it.hasNext()) {
			PhysicalObject spline = it.next();
			spline.applyTransform(dct);
			spline.applyTransform(staticTranslation);
		}
		
		if (writeGroundTruthToFile){
			// dynamic transforms
			SimpleMatrix dynamicCenterTransform = SimpleMatrix.I_4.clone();
			dynamicCenterTransform.setSubColValue(0, 3, dynamicCenterTranslation);

			// global transforms
			SimpleMatrix staticXMLBasedCenterTransform = SimpleMatrix.I_4.clone();
			staticXMLBasedCenterTransform.setSubColValue(0, 3, staticPhantomCenteringTranslations.get(subjectNumber));

			try {
				// Order is [scalingValues, groundTruthVICONMatrices, dynamicCenterTransformMatrix, staticXMLCenterTransformMatrix]
				if(selectDefaultVICONMotion){
					XmlUtils.exportToXML(new Object[]{VICONMotion.getBoneScalingMemorizer(), VICONMotion.getBoneGlobalTransformMemorizer(), dynamicCenterTransform, staticXMLBasedCenterTransform});
				} else {
					XmlUtils.exportToXML(new Object[]{artificialMotion.getBoneScalingMemorizer(), artificialMotion.getBoneGlobalTransformMemorizer(), dynamicCenterTransform, staticXMLBasedCenterTransform});
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	@Override
	public String getName() {
		return "XCat Dynamic Squat";
	}	

	public ArrayList<TimeVariantSurfaceBSpline> getVariants() {
		return variants;
	}	

	public static TreeMap<Integer, SimpleMatrix> getCurrentTransformMap(int BoneKey, TreeMap<String,TreeMap<Integer, SimpleMatrix>> map){
		// scaling goes here if applicable				
		TreeMap<Integer, SimpleMatrix> currMap = null;
		if (map!=null){
			switch (BoneKey) {
			case 1: //lFemur
				currMap =  map.get("LeftUpper");
				break;
			case 2: //lTibia
				currMap =  map.get("LeftLower");
				break;
			case 3: //lFibula
				currMap =  map.get("LeftLower");
				break;
			case 4: //lPatella
				currMap =  map.get("LeftUpper");
				break;
			case 5: //rFemur
				currMap =  map.get("RightUpper");
				break;
			case 6: //rTibia
				currMap =  map.get("RightLower");
				break;
			case 7: //rFibula
				currMap =  map.get("RightLower");
				break;
			case 8: //rPatella
				currMap =  map.get("RightUpper");
				break;
			default:
				break;
			}
		}	
		return currMap;
	}

	public static SimpleMatrix[] getGroundTruthTransform(int boneKey, String gtXMLpath){
		TreeMap<String, TreeMap<Integer, SimpleMatrix>> motionMap = null;
		SimpleMatrix dynamicTranslation = null;
		SimpleMatrix staticTranslation = null;
		try {
			Object[] gtObjects = (Object[])XmlUtils.importFromXML(gtXMLpath);
			motionMap = (TreeMap<String, TreeMap<Integer, SimpleMatrix>>) gtObjects[1];
			dynamicTranslation = (SimpleMatrix) gtObjects[2];
			staticTranslation = (SimpleMatrix) gtObjects[3];
		} catch (Exception e) {
			e.printStackTrace();
		}

		Iterator<Entry<Integer,SimpleMatrix>> it = getCurrentTransformMap(boneKey,motionMap).entrySet().iterator();
		SimpleMatrix[] out = new SimpleMatrix[getCurrentTransformMap(boneKey,motionMap).size()];
		while (it.hasNext()) {
			Entry<Integer, SimpleMatrix> entry = it.next();
			out[entry.getKey()] = SimpleOperators.multiplyMatrixProd(
					staticTranslation,
					SimpleOperators.multiplyMatrixProd(dynamicTranslation,entry.getValue())
					);
		}
		return out;
	}
}
/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */