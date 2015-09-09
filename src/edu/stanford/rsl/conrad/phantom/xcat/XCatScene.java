package edu.stanford.rsl.conrad.phantom.xcat;


import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

/**
 * Class to model scenes involving Paul Seagar's XCAT.<BR><BR>
 * <img src="http://dmip1.rad.jhmi.edu/xcat/ncat_ct.jpg" alt = "XCat example">
 * 
 * @author akmaier
 *
 */
public abstract class XCatScene extends AnalyticPhantom4D {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7268743275585203174L;

	/**
	 * Select the gender of the phantom.
	 */
	protected static boolean maleGender = true;

	/**
	 * If supine is selected, the patient is laying on this back. If it is set to false, the patient is defined as laying on this front side.
	 */
	protected static boolean supine = true;

	/**
	 * This reads in the position of the XCat files from the registry.
	 */
	protected String XCatDirectory = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.XCAT_PATH);

	/**
	 * defines whether the bone marrow is rendered.
	 */
	protected static boolean renderMarrow = true;

	/**
	 * Rendering of the arm marrow. Use this option with care. The arm bone marrow of XCat may belong to arms pointing upwards.
	 */
	protected static boolean renderArmMarrow = false;

	/**
	 * Time warping is used for animation. A time warper is a function that get an input time and return an output time.
	 */
	protected TimeWarper warper;

	protected ArrayList<SurfaceBSpline> splines = new ArrayList<SurfaceBSpline>();

	protected ArrayList<TimeVariantSurfaceBSpline> variants = new ArrayList<TimeVariantSurfaceBSpline>();

	
	/**
	 * returns the geometric definition of XCat. Only returns the splines that are rendered in the respective scene.
	 * @return the splines
	 */
	public ArrayList<SurfaceBSpline> getSplines() {
		return splines;
	}
	
	public ArrayList<TimeVariantSurfaceBSpline> getVariants() {
		return variants;
	}

	
	public void createPhysicalObjects(){
		clear();
		for (TimeVariantSurfaceBSpline spline: variants){
			add(this, spline, spline.getTitle());
		}
		for (SurfaceBSpline spline: splines){
			add(this, spline, spline.getTitle());
		}
	}
	
	/**
	 * This definition of XCat does tessellate the scene, i.e. render the complete scene in triangles.
	 * Function is based on several parameters to determine the number of points required to get a
	 * sufficient tessellation of the respective object.
	 * 
	 * @param voxelSizeX the resolution of the object in X direction
	 * @param voxelSizeY the resolution of the object in Y direction
	 * @param voxelSizeZ the resolution of the object in Z direction
	 * @param samplingU sampling factor in the spline internal u direction
	 * @param samplingV sampling factor in the spline internal v direction
	 * @param time the time between 0 and 1 to draw the scene. Note that the time is being warped according to the time warper.
	 * @return a scene at a given time consisting only of triangles.
	 */
	public abstract PrioritizableScene tessellateScene(double time); 

	public PrioritizableScene getScene(double time){
		return tessellateScene(time);
	}


	/**
	 * Adds a new shape to the scene. The shape is compared against the look-up table of known shapes to determine its material.
	 * @param phantom the phantom to add the shape
	 * @param shape the shape
	 * @param name the name of the shape used for the look up.
	 */
	protected void add(PrioritizableScene phantom, AbstractShape shape, String name){
		if (shape != null){
			if (getSplineNameMaterialNameLUT().keySet().contains(name)){
				PhysicalObject obj = new PhysicalObject();
				obj.setNameString(name);
				obj.setMaterial(generateFromSplineName(name));
				obj.setShape(shape);
				try{
					//System.out.println(obj.getNameString() + " " + getSplinePriorityLUT().get(name));
					phantom.add(obj, getSplinePriorityLUT().get(name));
				} catch (Exception e){
					System.out.println("ignored '" + name+"'");
				}
			} else {
				System.out.println("ignored '" + name +"'");
			}
		}
	}
	


	/**
	 * Returns the scene as an collection of Bsplines:
	 * <pre>
	 * type
	 * total size in floats
	 * # number of splines
	 * Bsplines
	 * priorities
	 * materials
	 * </pre>
	 * @return the binary representation
	 */
	public abstract float [] getBinaryRepresentation();


	/**
	 * Look up table for the priorites of the different shapes.
	 * Shapes with a higher priority are drawn over shapes with lower priority.
	 * @return the prioirity lut
	 */
	public static HashMap <String, Integer> getSplinePriorityLUT(){
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		map.put("*****BODY******", 0);
		map.put("**RightArm", 0);
		map.put("**LeftArm", 0);
		map.put("**RightLeg", 0);
		map.put("**LeftLeg", 0);
		map.put("*****RLUNG******", 8);
		map.put("*****LLUNG******", 8);
		map.put("*****LIVER******", 20);
		map.put("****Gall-Bladder****", 10);
		map.put("****Left-diaphragm****", 60);
		map.put("*****KIDNEY1******", 10);
		map.put("*****KIDNEY2******", 10);
		map.put("*****STOMACH******", 10);
		map.put("*****SPLEEN******", 10);
		map.put("*****STERNUM******", 10);
		map.put("*****thor_cartilage1****", 20);
		map.put("*****thor_cartilage2****", 20);
		map.put("*****thor_cartilage3****", 20);
		map.put("*****thor_cartilage4****", 20);
		map.put("*****thor_cartilage5****", 20);
		map.put("*****thor_cartilage6****", 20);
		map.put("*****thor_cartilage7****", 20);
		map.put("*****thor_cartilage8****", 20);
		map.put("*****thor_cartilage9****", 20);
		map.put("*****thor_cartilage10****", 20);
		map.put("*****thor_cartilage11****", 20);
		map.put("*****thor_cartilage12****", 20);
		map.put("*****lumbar_cartilage1****", 20);
		map.put("*****lumbar_cartilage2****", 20);
		map.put("*****lumbar_cartilage3****", 20);
		map.put("*****lumbar_cartilage4****", 20);
		map.put("*****lumbar_cartilage5****", 20);
		map.put("*****lumbar_cartilage6****", 20);
		map.put("****cerv_vert1a****", 10);
		map.put("****cerv_vert1b****", 10);
		map.put("****cerv_vert1c****", 10);
		map.put("****cerv_vert1d****", 10);
		map.put("****cerv_vert1e****", 10);
		map.put("****cerv_vert1f****", 10);
		map.put("*****cerv_vert2a****", 10);
		map.put("*****cerv_vert2b****", 10);
		map.put("*****cerv_vert2c****", 10);
		map.put("*****cerv_vert3a****", 10);
		map.put("*****cerv_vert3b****", 10);
		map.put("*****cerv_vert3c****", 10);
		map.put("*****cerv_vert4a****", 10);
		map.put("*****cerv_vert4b****", 10);
		map.put("*****cerv_vert4c****", 10);
		map.put("*****cerv_vert5a****", 10);
		map.put("*****cerv_vert5b****", 10);
		map.put("*****cerv_vert5c****", 10);
		map.put("*****cerv_vert6a****", 10);
		map.put("*****cerv_vert6b****", 10);
		map.put("*****cerv_vert6c****", 10);
		map.put("*****cerv_vert7a****", 10);
		map.put("*****cerv_vert7b****", 10);
		map.put("*****cerv_vert7c****", 10);
		map.put("*****thor_vert1a****", 10);
		map.put("*****thor_vert1b****", 10);
		map.put("*****thor_vert2a****", 10);
		map.put("*****thor_vert2b****", 10);
		map.put("*****thor_vert3a****", 10);
		map.put("*****thor_vert3b****", 10);
		map.put("*****thor_vert4a****", 10);
		map.put("*****thor_vert4b****", 10);
		map.put("*****thor_vert5a****", 10);
		map.put("*****thor_vert5b****", 10);
		map.put("*****thor_vert6a****", 10);
		map.put("*****thor_vert6b****", 10);
		map.put("*****thor_vert7a****", 10);
		map.put("*****thor_vert7b****", 10);
		map.put("*****thor_vert8a****", 10);
		map.put("*****thor_vert8b****", 10);
		map.put("*****thor_vert9a****", 10);
		map.put("*****thor_vert9b****", 10);
		map.put("*****thor_vert10a****", 10);
		map.put("*****thor_vert10b****", 10);
		map.put("*****thor_vert11a****", 10);
		map.put("*****thor_vert11b****", 10);
		map.put("*****thor_vert12a****", 10);
		map.put("*****thor_vert12b****", 10);
		map.put("*****lum_vert1a****", 10);
		map.put("*****lum_vert1b****", 10);
		map.put("*****lum_vert2a****", 10);
		map.put("*****lum_vert2b****", 10);
		map.put("*****lum_vert3a****", 10);
		map.put("*****lum_vert3b****", 10);
		map.put("*****lum_vert4a****", 10);
		map.put("*****lum_vert4b****", 10);
		map.put("*****lum_vert5a****", 10);
		map.put("*****lum_vert5b****", 10);
		map.put("*****sacrum****", 20);
		map.put("*****sacrum_cart****", 30);
		map.put("***RRIB1-Cartilage", 30);
		map.put("***RRIB2-Cartilage", 30);
		map.put("***RRIB3-Cartilage", 30);
		map.put("***RRIB4-Cartilage", 30);
		map.put("***RRIB5-Cartilage", 30);
		map.put("***RRIB6-Cartilage", 30);
		map.put("***RRIB7-Cartilage", 30);
		map.put("***RRIB8-Cartilage", 30);
		map.put("***RRIB9-Cartilage", 30);
		map.put("***Right-Sternum-Cartilage", 30);
		map.put("***RRIB1", 10);
		map.put("***RRIB2", 10);
		map.put("***RRIB3", 10);
		map.put("***RRIB4", 10);
		map.put("***RRIB5", 10);
		map.put("***RRIB6", 10);
		map.put("***RRIB7", 10);
		map.put("***RRIB8", 10);
		map.put("***RRIB9", 10);
		map.put("***RRIB10", 10);
		map.put("***RRIB11", 10);
		map.put("***RRIB12", 10);
		map.put("***LRIB1-Cartilage", 30);
		map.put("***LRIB2-Cartilage", 30);
		map.put("***LRIB3-Cartilage", 30);
		map.put("***LRIB4-Cartilage", 30);
		map.put("***LRIB5-Cartilage", 30);
		map.put("***LRIB6-Cartilage", 30);
		map.put("***LRIB7-Cartilage", 30);
		map.put("***LRIB8-Cartilage", 30);
		map.put("***LRIB9-Cartilage", 30);
		map.put("***Left-Sternum-Cartilage", 30);
		map.put("***LRIB1", 10);
		map.put("***LRIB2", 10);
		map.put("***LRIB3", 10);
		map.put("***LRIB4", 10);
		map.put("***LRIB5", 10);
		map.put("***LRIB6", 10);
		map.put("***LRIB7", 10);
		map.put("***LRIB8", 10);
		map.put("***LRIB9", 10);
		map.put("***LRIB10", 10);
		map.put("***LRIB11", 10);
		map.put("***LRIB12", 10);
		map.put("**RightHumerus", 10);
		map.put("**RightRadius", 10);
		map.put("**RightUlna", 10);
		map.put("**RightCapitate", 10);
		map.put("**RightHamate", 10);
		map.put("**RightLunate", 10);
		map.put("**RightMetacarpal", 10);
		map.put("**RightMetacarpal", 10);
		map.put("**RightMetacarpal", 10);
		map.put("**RightMetacarpal", 10);
		map.put("**RightMetacarpal", 10);
		map.put("**RightPisiform", 10);
		map.put("**RightScaphoid", 10);
		map.put("**RightTrapezium", 10);
		map.put("**RightTriquetrum", 10);
		map.put("**RightTrapezoid", 10);
		map.put("**RightFinger1", 10);
		map.put("**RightFinger2", 10);
		map.put("**RightFinger3", 10);
		map.put("**RightFinger4", 10);
		map.put("**RightFinger5", 10);
		map.put("**RightFinger6", 10);
		map.put("**RightFinger7", 10);
		map.put("**RightFinger8", 10);
		map.put("**RightFinger9", 10);
		map.put("**RightFinger10", 10);
		map.put("**RightFinger11", 10);
		map.put("**RightFinger12", 10);
		map.put("**RightFinger13", 10);
		map.put("**RightFinger14", 10);
		map.put("**LeftHumerus", 10);
		map.put("**LeftRadius", 10);
		map.put("**LeftUlna", 10);
		map.put("**LeftCapitate", 10);
		map.put("**LeftHamate", 10);
		map.put("**LeftLunate", 10);
		map.put("**LeftMetacarpal", 10);
		map.put("**LeftMetacarpal", 10);
		map.put("**LeftMetacarpal", 10);
		map.put("**LeftMetacarpal", 10);
		map.put("**LeftMetacarpal", 10);
		map.put("**LeftPisiform", 10);
		map.put("**LeftScaphoid", 10);
		map.put("**LeftTrapezium", 10);
		map.put("**LeftTriquetrum", 10);
		map.put("**LeftTrapezoid", 10);
		map.put("**LeftFinger1", 10);
		map.put("**LeftFinger2", 10);
		map.put("**LeftFinger3", 10);
		map.put("**LeftFinger4", 10);
		map.put("**LeftFinger5", 10);
		map.put("**LeftFinger6", 10);
		map.put("**LeftFinger7", 10);
		map.put("**LeftFinger8", 10);
		map.put("**LeftFinger9", 10);
		map.put("**LeftFinger10", 10);
		map.put("**LeftFinger11", 10);
		map.put("**LeftFinger12", 10);
		map.put("**LeftFinger13", 10);
		map.put("**LeftFinger14", 10);
		map.put("**RightFemur", 10);
		map.put("**RightTibia", 10);
		map.put("**RightFibula", 10);
		map.put("**RightPatella", 10);
		map.put("**RightTalus", 10);
		map.put("**RightCalcaneus", 10);
		map.put("**RightCuboid", 10);
		map.put("**Right_inter_cuneiform", 10);
		map.put("**Right_lat_cuneiform", 10);
		map.put("**Right_med_cuneiform", 10);
		map.put("**RightTarsal", 10);
		map.put("**RightTarsal", 10);
		map.put("**RightTarsal", 10);
		map.put("**RightTarsal", 10);
		map.put("**RightTarsal", 10);
		map.put("**RightNavicular", 10);
		map.put("RightToe1", 10);
		map.put("RightToe2", 10);
		map.put("RightToe3", 10);
		map.put("RightToe4", 10);
		map.put("RightToe5", 10);
		map.put("RightToe6", 10);
		map.put("RightToe7", 10);
		map.put("RightToe8", 10);
		map.put("RightToe9", 10);
		map.put("RightToe10", 10);
		map.put("RightToe11", 10);
		map.put("RightToe12", 10);
		map.put("RightToe13", 10);
		map.put("RightToe14", 10);
		map.put("*LeftFemur", 10);
		map.put("**LeftTibia", 10);
		map.put("**LeftFibula", 10);
		map.put("**LeftPatella", 10);
		map.put("**LeftTalus", 10);
		map.put("**LeftCalcaneus", 10);
		map.put("**LeftCuboid", 10);
		map.put("**Left_inter_cuneiform", 10);
		map.put("**Left_lat_cuneiform", 10);
		map.put("**Left_med_cuneiform", 10);
		map.put("**LeftTarsal", 10);
		map.put("**LeftTarsal", 10);
		map.put("**LeftTarsal", 10);
		map.put("**LeftTarsal", 10);
		map.put("**LeftTarsal", 10);
		map.put("**LeftNavicular", 10);
		map.put("LeftToe1", 10);
		map.put("LeftToe2", 10);
		map.put("LeftToe3", 10);
		map.put("LeftToe4", 10);
		map.put("LeftToe5", 10);
		map.put("LeftToe6", 10);
		map.put("LeftToe7", 10);
		map.put("LeftToe8", 10);
		map.put("LeftToe9", 10);
		map.put("LeftToe10", 10);
		map.put("LeftToe11", 10);
		map.put("LeftToe12", 10);
		map.put("LeftToe13", 10);
		map.put("LeftToe14", 10);
		map.put("***RIGHT_COLLARBONE***", 10);
		map.put("***RIGHT_SCAPULA***", 10);
		map.put("****LEFT_COLLARBONE****", 10);
		map.put("****LEFT_SCAPULA****", 10);
		map.put("****skull_face1****", 10);
		map.put("****skull_face2****", 10);
		map.put("****skull1****", 10);
		map.put("****skull2****", 10);
		map.put("****skull3****", 10);
		map.put("****skull4****", 10);
		map.put("****skull5****", 10);
		map.put("****jaw****", 10);
		map.put("****nasal_passage1****", 20);
		map.put("****nasal_passage2****", 20);
		map.put("****brain****", 50);
		map.put("****cerebellum****", 50);
		map.put("****brain_stem****", 50);
		if (!maleGender){
			if (!supine) {
				map.put("****RBREAST-Prone*****", 0);
				map.put("****LBREAST-Prone*****", 0); 
			} else {
				map.put("****RBREAST-Supine*****", 0);
				map.put("****LBREAST-Supine*****", 0);
			}
		}
		map.put("*****RHIP*****", 10);
		map.put("*****RPELVIS1*****", 10);
		map.put("*****RPELVIS2*****", 10);
		map.put("*****RHIPC*****", 10);
		map.put("*****RPELVIS1C*****", 10);
		map.put("*****RPELVIS2C*****", 10);
		map.put("*****LHIP*****", 10);
		map.put("*****LPELVIS1*****", 10);
		map.put("*****LPELVIS2*****", 10);
		map.put("*****LHIPC*****", 10);
		map.put("*****LPELVIS1C*****", 10);
		map.put("*****LPELVIS2C*****", 10);
		map.put("*****R_URETER*****", 10);
		map.put("*****L_URETER*****", 10);
		map.put("*****BLADDER*****", 10);
		map.put("*****R_VAS_DEF*****", 10);
		map.put("*****L_VAS_DEF*****", 10);
		if (maleGender) {
			map.put("*****SEMINAL_VES*****", 20);
			map.put("*****PROSTATE*****", 20);
			map.put("*****URETHRA*****", 20);
			map.put("*****R_TEST*****", 20);
			map.put("*****L_TEST*****", 20);
			map.put("*****PENIS*****", 0);
		} else {
			map.put("****UTERUS*****", 20);
			map.put("****RIGHT_OVARY*****", 20);
			map.put("****RIGHT_OVARY2*****", 20);
			map.put("****R_FL_TUBE*****", 20);
			map.put("****LEFT_OVARY*****", 20);
			map.put("****LEFT_OVARY2*****", 20);
			map.put("****L_FL_TUBE*****", 20);
			map.put("****VAGINA1*****", 20);
			map.put("****VAGINA2*****", 20);
		}
		map.put("*****ASC_LARGE_INTEST*****", 20);
		map.put("*****TRANS_LARGE_INTEST*****", 20);
		map.put("*****DESC_LARGE_INTEST*****", 20);
		map.put("*****RECTUM*****", 20);
		map.put("*****ASC_LARGE_INTEST_AIR*****", 50);
		map.put("*****TRANS_LARGE_INTEST_AIR*****", 50);
		map.put("*****DESC_LARGE_INTEST_AIR*****", 50);
		map.put("*****RECTUM_AIR*****", 50);
		map.put("****small_intest_0****", 20);
		map.put("****small_intest_1****", 20);
		map.put("****small_intest_2****", 20);
		map.put("****small_intest_3****", 20);
		map.put("****small_intest_4****", 20);
		map.put("****small_intest_5****", 20);
		map.put("****small_intest_6****", 20);
		map.put("****small_intest_7****", 20);
		map.put("****small_intest_8****", 20);
		map.put("****small_intest_9****", 20);
		map.put("****small_intest_10****", 20);
		map.put("****small_intest_11****", 20);
		map.put("****small_intest_12****", 20);
		map.put("****small_intest_13****", 20);
		map.put("****small_intest_14****", 20);
		map.put("****small_intest_15****", 20);
		map.put("****small_intest_16****", 20);
		map.put("****small_intest_17****", 20);
		map.put("****small_intest_18****", 20);
		map.put("****small_intest_19****", 20);
		map.put("****small_intest_20****", 20);
		map.put("****small_intest_21****", 20);
		map.put("****small_intest_22****", 20);
		map.put("****small_intest_23****", 20);
		map.put("****small_intest_24****", 20);
		map.put("****small_intest_25****", 20);
		map.put("****small_intest_26****", 20);
		map.put("****small_intest_27****", 20);
		map.put("****small_intest_28****", 20);
		map.put("****small_intest_29****", 20);
		map.put("****small_intest_30****", 20);
		map.put("****small_intest_31****", 20);
		map.put("****small_intest_32****", 20);
		map.put("****small_intest_33****", 20);
		map.put("****small_intest_34****", 20);
		map.put("****small_intest_35****", 20);
		map.put("****small_intest_36****", 20);
		map.put("****small_intest_37****", 20);
		map.put("****small_intest_38****", 20);
		map.put("****small_intest_39****", 20);
		map.put("****small_intest_40****", 20);
		map.put("****small_intest_41****", 20);
		map.put("****small_intest_42****", 20);
		map.put("****small_intest_air_0****", 50);
		map.put("****small_intest_air_1****", 50);
		map.put("****small_intest_air_2****", 50);
		map.put("****small_intest_air_3****", 50);
		map.put("****small_intest_air_4****", 50);
		map.put("****small_intest_air_5****", 50);
		map.put("****small_intest_air_6****", 50);
		map.put("****small_intest_air_7****", 50);
		map.put("****small_intest_air_8****", 50);
		map.put("****small_intest_air_9****", 50);
		map.put("****small_intest_air_10****", 50);
		map.put("****small_intest_air_11****", 50);
		map.put("****small_intest_air_12****", 50);
		map.put("****small_intest_air_13****", 50);
		map.put("****small_intest_air_14****", 50);
		map.put("****small_intest_air_15****", 50);
		map.put("****small_intest_air_16****", 50);
		map.put("****small_intest_air_17****", 50);
		map.put("****small_intest_air_18****", 50);
		map.put("****small_intest_air_19****", 50);
		map.put("****small_intest_air_20****", 50);
		map.put("****small_intest_air_21****", 50);
		map.put("****small_intest_air_22****", 50);
		map.put("****small_intest_air_23****", 50);
		map.put("****small_intest_air_24****", 50);
		map.put("****small_intest_air_25****", 50);
		map.put("****small_intest_air_26****", 50);
		map.put("****small_intest_air_27****", 50);
		map.put("****small_intest_air_28****", 50);
		map.put("****small_intest_air_29****", 50);
		map.put("****small_intest_air_30****", 50);
		map.put("****small_intest_air_31****", 50);
		map.put("****small_intest_air_32****", 50);
		map.put("****small_intest_air_33****", 50);
		map.put("****small_intest_air_34****", 50);
		map.put("****small_intest_air_35****", 50);
		map.put("****small_intest_air_36****", 50);
		map.put("****small_intest_air_37****", 50);
		map.put("****small_intest_air_38****", 50);
		map.put("****small_intest_air_39****", 50);
		map.put("****small_intest_air_40****", 50);
		map.put("****small_intest_air_41****", 50);
		map.put("****small_intest_air_42****", 50);
		map.put("*****AORTA*****", 1050);
		map.put("*****Artery1*****", 1150);
		map.put("*****Artery2*****", 1150);
		map.put("*****Artery3*****", 1150);
		map.put("*****Artery4*****", 1150);
		map.put("*****Artery5*****", 1150);
		map.put("*****Artery6*****", 1150);
		map.put("*****Artery7*****", 1150);
		map.put("*****Artery8*****", 1150);
		map.put("*****Artery9*****", 1150);
		map.put("*****Artery10*****", 1150);
		map.put("*****Artery11*****", 1150);
		map.put("*****Artery12*****", 1150);
		map.put("*****Artery13*****", 1150);
		map.put("*****L_KIDNEY_ART1*****", 150);
		map.put("*****L_KIDNEY_ART2*****", 150);
		map.put("*****R_KIDNEY_ART1*****", 150);
		map.put("*****R_KIDNEY_ART2*****", 150);
		map.put("*****INFERIOR_VENA_CAVA*****", 1040);
		map.put("*****VEIN*****", 1150);
		map.put("*****VEIN*****", 1150);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****VEIN*****", 1100);
		map.put("*****LIVER_VEINS*****", 100);
		map.put("*****LIVER_VEINS*****", 100);
		map.put("*****LIVER_VEINS*****", 100);
		map.put("*****LIVER_VEINS*****", 100);
		map.put("*****L_KIDNEY_VEIN*****",100);
		map.put("*****R_KIDNEY_VEIN*****", 100);
		map.put("*****PREAORTIC_NODE*****", 1100);
		map.put("*****COMMON_ILLIAC_NODE1*****", 100);
		map.put("*****COMMON_ILLIAC_NODE2*****", 100);
		map.put("*****COMMON_ILLIAC_NODE3*****", 100);
		map.put("*****COMMON_ILLIAC_NODE4*****", 100);
		map.put("*****RIGHT_EXT_ILLIAC_NODE1*****", 100);
		map.put("*****RIGHT_EXT_ILLIAC_NODE2*****", 100);
		map.put("*****RIGHT_EXT_ILLIAC_NODE3*****", 100);
		map.put("*****RIGHT_EXT_ILLIAC_NODE4*****", 100);
		map.put("*****RIGHT_EXT_ILLIAC_NODE5*****", 100);
		map.put("*****RIGHT_HYPOGASTRIC_NODE*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE2*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE3*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE4*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE5*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE6*****", 100);
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE7*****", 100);
		map.put("*****LEFT_EXT_ILLIAC_NODE1*****", 100);
		map.put("*****LEFT_EXT_ILLIAC_NODE2*****", 100);
		map.put("*****LEFT_EXT_ILLIAC_NODE3*****", 100);
		map.put("*****LEFT_EXT_ILLIAC_NODE4*****", 100);
		map.put("*****LEFT_EXT_ILLIAC_NODE5*****", 100);
		map.put("*****LEFT_HYPOGASTRIC_NODE*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", 100);
		map.put("****Airway-1*********", 70);
		map.put("*****PULMONARY_ART1*****", 1040);
		map.put("*****PULMONARY_VEIN1*****", 1040);
		if (renderMarrow) {
			map.put("**RRIB1_MARROW**", 80);
			map.put("**RRIB2_MARROW**", 80);
			map.put("**RRIB3_MARROW**", 80);
			map.put("**RRIB4_MARROW**", 80);
			map.put("**RRIB5_MARROW**", 80);
			map.put("**RRIB6_MARROW**", 80);
			map.put("**RRIB7_MARROW**", 80);
			map.put("**RRIB8_MARROW**", 80);
			map.put("**RRIB9_MARROW**", 80);
			map.put("**RRIB10_MARROW**", 80);
			map.put("**RRIB11_MARROW**", 80);
			map.put("**RRIB12_MARROW**", 80);
			map.put("**LRIB1_MARROW**", 80);
			map.put("**LRIB2_MARROW**", 80);
			map.put("**LRIB3_MARROW**", 80);
			map.put("**LRIB4_MARROW**", 80);
			map.put("**LRIB5_MARROW**", 80);
			map.put("**LRIB6_MARROW**", 80);
			map.put("**LRIB7_MARROW**", 80);
			map.put("**LRIB8_MARROW**", 80);
			map.put("**LRIB9_MARROW**", 80);
			map.put("**LRIB10_MARROW**", 80);
			map.put("**LRIB11_MARROW**", 80);
			map.put("**LRIB12_MARROW**", 80);
			map.put("**BACKBONE1_MARROW**", 80);
			map.put("**BACKBONE2_MARROW**", 80);
			map.put("**BACKBONE3_MARROW**", 80);
			map.put("**BACKBONE4_MARROW**", 80);
			map.put("**BACKBONE5_MARROW**", 80);
			map.put("**BACKBONE6_MARROW**", 80);
			map.put("**BACKBONE7_MARROW**", 80);
			map.put("**BACKBONE8_MARROW**", 80);
			map.put("**BACKBONE9_MARROW**", 80);
			map.put("**BACKBONE10_MARROW**", 80);
			map.put("**BACKBONE11_MARROW**", 80);
			map.put("**BACKBONE12_MARROW**", 80);
			map.put("**BACKBONE13_MARROW**", 80);
			map.put("**BACKBONE14_MARROW**", 80);
			map.put("**BACKBONE15_MARROW**", 80);
			map.put("**BACKBONE16_MARROW**", 80);
			map.put("**BACKBONE17_MARROW**", 80);
			map.put("**BACKBONE18_MARROW**", 80);
			map.put("**BACKBONE19_MARROW**", 80);
			map.put("**BACKBONE20_MARROW**", 80);
			map.put("**BACKBONE21_MARROW**", 80);
			map.put("**BACKBONE22_MARROW**", 80);
			map.put("**BACKBONE23_MARROW**", 80);
			map.put("**BACKBONE24_MARROW**", 80);
			map.put("**BACKBONE25_MARROW**", 80);
			map.put("**BACKBONE26_MARROW**", 80);
			map.put("**BACKBONE27_MARROW**", 80);
			map.put("**BACKBONE28_MARROW**", 80);
			map.put("**BACKBONE29_MARROW**", 80);
			map.put("**BACKBONE30_MARROW**", 80);
			map.put("**BACKBONE31_MARROW**", 80);
			map.put("**BACKBONE32_MARROW**", 80);
			map.put("**BACKBONE33_MARROW**", 80);
			map.put("**BACKBONE34_MARROW**", 80);
			map.put("**BACKBONE35_MARROW**", 80);
			map.put("**BACKBONE36_MARROW**", 80);
			map.put("**BACKBONE37_MARROW**", 80);
			map.put("**BACKBONE38_MARROW**", 80);
			map.put("**BACKBONE39_MARROW**", 80);
			map.put("**BACKBONE40_MARROW**", 80);
			map.put("**BACKBONE41_MARROW**", 80);
			map.put("**BACKBONE42_MARROW**", 80);
			map.put("**BACKBONE43_MARROW**", 80);
			map.put("**BACKBONE44_MARROW**", 80);
			map.put("**BACKBONE45_MARROW**", 80);
			map.put("**BACKBONE46_MARROW**", 80);
			map.put("**BACKBONE47_MARROW**", 80);
			map.put("**BACKBONE48_MARROW**", 80);
			map.put("**BACKBONE49_MARROW**", 80);
			map.put("**BACKBONE50_MARROW**", 80);
			map.put("**BACKBONE51_MARROW**", 80);
			map.put("**BACKBONE52_MARROW**", 80);
			map.put("**BACKBONE53_MARROW**", 80);
			map.put("**BACKBONE54_MARROW**", 80);
			map.put("**BACKBONE55_MARROW**", 80);
			map.put("**BACKBONE56_MARROW**", 80);
			map.put("**BACKBONE57_MARROW**", 80);
			map.put("**BACKBONE58_MARROW**", 80);
			map.put("**STERN_MARROW**", 80);
			map.put("**RSCAP_MARROW**", 80);
			map.put("**RCOLLAR_MARROW**", 80);
			map.put("**LSCAP_MARROW**", 80);
			map.put("**LCOLLAR_MARROW**", 80);
			if (renderArmMarrow){
				map.put("**RHUMERUS_MARROW**", 80);
				map.put("**R_RADIUS_MARROW**", 80);
				map.put("**R_ULNA_MARROW**", 80);
				map.put("**LHUMERUS_MARROW**", 80);
				map.put("**L_RADIUS_MARROW**", 80);
				map.put("**L_ULNA_MARROW**", 80);
				map.put("**R_HAND_MARROW0", 80);
				map.put("**R_HAND_MARROW1", 80);
				map.put("**R_HAND_MARROW2", 80);
				map.put("**R_HAND_MARROW3", 80);
				map.put("**R_HAND_MARROW4", 80);
				map.put("**R_HAND_MARROW5", 80);
				map.put("**R_HAND_MARROW6", 80);
				map.put("**R_HAND_MARROW7", 80);
				map.put("**R_HAND_MARROW8", 80);
				map.put("**R_HAND_MARROW9", 80);
				map.put("**R_HAND_MARROW10", 80);
				map.put("**R_HAND_MARROW11", 80);
				map.put("**R_HAND_MARROW12", 80);
				map.put("**R_HAND_MARROW13", 80);
				map.put("**R_HAND_MARROW14", 80);
				map.put("**R_HAND_MARROW15", 80);
				map.put("**R_HAND_MARROW16", 80);
				map.put("**R_HAND_MARROW17", 80);
				map.put("**R_HAND_MARROW18", 80);
				map.put("**R_HAND_MARROW19", 80);
				map.put("**R_HAND_MARROW20", 80);
				map.put("**R_HAND_MARROW21", 80);
				map.put("**R_HAND_MARROW22", 80);
				map.put("**R_HAND_MARROW23", 80);
				map.put("**R_HAND_MARROW24", 80);
				map.put("**R_HAND_MARROW25", 80);
				map.put("**R_HAND_MARROW26", 80);
				map.put("**L_HAND_MARROW0", 80);
				map.put("**L_HAND_MARROW1", 80);
				map.put("**L_HAND_MARROW2", 80);
				map.put("**L_HAND_MARROW3", 80);
				map.put("**L_HAND_MARROW4", 80);
				map.put("**L_HAND_MARROW5", 80);
				map.put("**L_HAND_MARROW6", 80);
				map.put("**L_HAND_MARROW7", 80);
				map.put("**L_HAND_MARROW8", 80);
				map.put("**L_HAND_MARROW9", 80);
				map.put("**L_HAND_MARROW10", 80);
				map.put("**L_HAND_MARROW11", 80);
				map.put("**L_HAND_MARROW12", 80);
				map.put("**L_HAND_MARROW13", 80);
				map.put("**L_HAND_MARROW14", 80);
				map.put("**L_HAND_MARROW15", 80);
				map.put("**L_HAND_MARROW16", 80);
				map.put("**L_HAND_MARROW17", 80);
				map.put("**L_HAND_MARROW18", 80);
				map.put("**L_HAND_MARROW19", 80);
				map.put("**L_HAND_MARROW20", 80);
				map.put("**L_HAND_MARROW21", 80);
				map.put("**L_HAND_MARROW22", 80);
				map.put("**L_HAND_MARROW23", 80);
				map.put("**L_HAND_MARROW24", 80);
				map.put("**L_HAND_MARROW25", 80);
				map.put("**L_HAND_MARROW26", 80);
			}
			map.put("**R_FOOT_MARROW0", 80);
			map.put("**R_FOOT_MARROW1", 80);
			map.put("**R_FOOT_MARROW2", 80);
			map.put("**R_FOOT_MARROW3", 80);
			map.put("**R_FOOT_MARROW4", 80);
			map.put("**R_FOOT_MARROW5", 80);
			map.put("**R_FOOT_MARROW6", 80);
			map.put("**R_FOOT_MARROW7", 80);
			map.put("**R_FOOT_MARROW8", 80);
			map.put("**R_FOOT_MARROW9", 80);
			map.put("**R_FOOT_MARROW10", 80);
			map.put("**R_FOOT_MARROW11", 80);
			map.put("**R_FOOT_MARROW12", 80);
			map.put("**R_FOOT_MARROW13", 80);
			map.put("**R_FOOT_MARROW14", 80);
			map.put("**R_FOOT_MARROW15", 80);
			map.put("**R_FOOT_MARROW16", 80);
			map.put("**R_FOOT_MARROW17", 80);
			map.put("**R_FOOT_MARROW18", 80);
			map.put("**R_FOOT_MARROW19", 80);
			map.put("**R_FOOT_MARROW20", 80);
			map.put("**R_FOOT_MARROW21", 80);
			map.put("**R_FOOT_MARROW22", 80);
			map.put("**R_FOOT_MARROW23", 80);
			map.put("**R_FOOT_MARROW24", 80);
			map.put("**R_FOOT_MARROW25", 80);
			map.put("**L_FOOT_MARROW0", 80);
			map.put("**L_FOOT_MARROW1", 80);
			map.put("**L_FOOT_MARROW2", 80);
			map.put("**L_FOOT_MARROW3", 80);
			map.put("**L_FOOT_MARROW4", 80);
			map.put("**L_FOOT_MARROW5", 80);
			map.put("**L_FOOT_MARROW6", 80);
			map.put("**L_FOOT_MARROW7", 80);
			map.put("**L_FOOT_MARROW8", 80);
			map.put("**L_FOOT_MARROW9", 80);
			map.put("**L_FOOT_MARROW10", 80);
			map.put("**L_FOOT_MARROW11", 80);
			map.put("**L_FOOT_MARROW12", 80);
			map.put("**L_FOOT_MARROW13", 80);
			map.put("**L_FOOT_MARROW14", 80);
			map.put("**L_FOOT_MARROW15", 80);
			map.put("**L_FOOT_MARROW16", 80);
			map.put("**L_FOOT_MARROW17", 80);
			map.put("**L_FOOT_MARROW18", 80);
			map.put("**L_FOOT_MARROW19", 80);
			map.put("**L_FOOT_MARROW20", 80);
			map.put("**L_FOOT_MARROW21", 80);
			map.put("**L_FOOT_MARROW22", 80);
			map.put("**L_FOOT_MARROW23", 80);
			map.put("**L_FOOT_MARROW24", 80);
			map.put("**L_FOOT_MARROW25", 80);
			map.put("**R_FEMUR_MARROW**", 80);
			map.put("**R_TIBIA_MARROW**", 80);
			map.put("**R_FIBULA_MARROW**", 80);
			map.put("**R_PATELLA_MARROW**", 80);
			map.put("**L_FEMUR_MARROW**", 80);
			map.put("**L_TIBIA_MARROW**", 80);
			map.put("**L_FIBULA_MARROW**", 80);
			map.put("**L_PATELLA_MARROW**", 80);
		}
		// heart
		map.put("****Right-atria-myo*********", 1002);
		map.put("****Right-ventricle-myo*********", 1002);
		map.put("****Right-atria-chamber****", 1049);
		map.put("****Right-ventricle-chamber****", 1055);
		map.put("****Left-atria-myo**********", 1002);
		map.put("****Left-ventricle-myo**********", 1001);
		map.put("****Left-atria-chamber*****", 1006);
		map.put("****Left-ventricle-chamber1*****", 1100);
		map.put("****Left-ventricle-chamber2*****", 1101);

		map.put("****Left-ventricle-chamber3*****", 1102);

		map.put("****Left-ventricle-chamber4*****", 1104);
		
		map.put("****Heartwall*********", 1000);
		//the coronaries have a much higher priority than the chambers
		map.put("/***RCA1***/", 1112);
		map.put("/***RCA2***/", 1112);
		map.put("/***LCA1***/", 1112);
		map.put("/***LCA2***/", 1112);
		map.put("/***LCA3***/", 1112);
		map.put("/***LCA4***/", 1112);
		map.put("/***LCA5***/", 1112);
		map.put("/***LCA6***/", 1112);
		map.put("/***LCA7***/", 1112);
		map.put("/***LCA8***/", 1112);
		map.put("/***LCA9***/", 1112);
		map.put("/***LCA10***/", 1112);
		map.put("Heart Lesion", 1113);
		map.put("Heart Catheter", 1200);
		return map;
	}

	/**
	 * The lookup table for materials according to the spline names.
	 * @return the lut
	 */
	public static HashMap <String, String> getSplineNameMaterialNameLUT(){
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("*****BODY******", "Body (water)");
		map.put("**RightArm", "Body (water)");
		map.put("**LeftArm", "Body (water)");
		map.put("**RightLeg", "Body (water)");
		map.put("**LeftLeg", "Body (water)");
		map.put("*****RLUNG******", "Lung");
		map.put("*****LLUNG******", "Lung");
		map.put("*****LIVER******", "Liver");
		map.put("****Gall-Bladder****", "Intestine");
		map.put("****Left-diaphragm****", "Muscle");
		map.put("*****KIDNEY1******", "Kidney");
		map.put("*****KIDNEY2******", "Kidney");
		map.put("*****STOMACH******", "Intestine");
		map.put("*****SPLEEN******", "Spleen");
		map.put("*****STERNUM******", "Rib Bone");
		map.put("*****thor_cartilage1****", "Cartilage");
		map.put("*****thor_cartilage2****", "Cartilage");
		map.put("*****thor_cartilage3****", "Cartilage");
		map.put("*****thor_cartilage4****", "Cartilage");
		map.put("*****thor_cartilage5****", "Cartilage");
		map.put("*****thor_cartilage6****", "Cartilage");
		map.put("*****thor_cartilage7****", "Cartilage");
		map.put("*****thor_cartilage8****", "Cartilage");
		map.put("*****thor_cartilage9****", "Cartilage");
		map.put("*****thor_cartilage10****", "Cartilage");
		map.put("*****thor_cartilage11****", "Cartilage");
		map.put("*****thor_cartilage12****", "Cartilage");
		map.put("*****lumbar_cartilage1****", "Cartilage");
		map.put("*****lumbar_cartilage2****", "Cartilage");
		map.put("*****lumbar_cartilage3****", "Cartilage");
		map.put("*****lumbar_cartilage4****", "Cartilage");
		map.put("*****lumbar_cartilage5****", "Cartilage");
		map.put("*****lumbar_cartilage6****", "Cartilage");
		map.put("****cerv_vert1a****", "Spine Bone");
		map.put("****cerv_vert1b****", "Spine Bone");
		map.put("****cerv_vert1c****", "Spine Bone");
		map.put("****cerv_vert1d****", "Spine Bone");
		map.put("****cerv_vert1e****", "Spine Bone");
		map.put("****cerv_vert1f****", "Spine Bone");
		map.put("*****cerv_vert2a****", "Spine Bone");
		map.put("*****cerv_vert2b****", "Spine Bone");
		map.put("*****cerv_vert2c****", "Spine Bone");
		map.put("*****cerv_vert3a****", "Spine Bone");
		map.put("*****cerv_vert3b****", "Spine Bone");
		map.put("*****cerv_vert3c****", "Spine Bone");
		map.put("*****cerv_vert4a****", "Spine Bone");
		map.put("*****cerv_vert4b****", "Spine Bone");
		map.put("*****cerv_vert4c****", "Spine Bone");
		map.put("*****cerv_vert5a****", "Spine Bone");
		map.put("*****cerv_vert5b****", "Spine Bone");
		map.put("*****cerv_vert5c****", "Spine Bone");
		map.put("*****cerv_vert6a****", "Spine Bone");
		map.put("*****cerv_vert6b****", "Spine Bone");
		map.put("*****cerv_vert6c****", "Spine Bone");
		map.put("*****cerv_vert7a****", "Spine Bone");
		map.put("*****cerv_vert7b****", "Spine Bone");
		map.put("*****cerv_vert7c****", "Spine Bone");
		map.put("*****thor_vert1a****", "Spine Bone");
		map.put("*****thor_vert1b****", "Spine Bone");
		map.put("*****thor_vert2a****", "Spine Bone");
		map.put("*****thor_vert2b****", "Spine Bone");
		map.put("*****thor_vert3a****", "Spine Bone");
		map.put("*****thor_vert3b****", "Spine Bone");
		map.put("*****thor_vert4a****", "Spine Bone");
		map.put("*****thor_vert4b****", "Spine Bone");
		map.put("*****thor_vert5a****", "Spine Bone");
		map.put("*****thor_vert5b****", "Spine Bone");
		map.put("*****thor_vert6a****", "Spine Bone");
		map.put("*****thor_vert6b****", "Spine Bone");
		map.put("*****thor_vert7a****", "Spine Bone");
		map.put("*****thor_vert7b****", "Spine Bone");
		map.put("*****thor_vert8a****", "Spine Bone");
		map.put("*****thor_vert8b****", "Spine Bone");
		map.put("*****thor_vert9a****", "Spine Bone");
		map.put("*****thor_vert9b****", "Spine Bone");
		map.put("*****thor_vert10a****", "Spine Bone");
		map.put("*****thor_vert10b****", "Spine Bone");
		map.put("*****thor_vert11a****", "Spine Bone");
		map.put("*****thor_vert11b****", "Spine Bone");
		map.put("*****thor_vert12a****", "Spine Bone");
		map.put("*****thor_vert12b****", "Spine Bone");
		map.put("*****lum_vert1a****", "Spine Bone");
		map.put("*****lum_vert1b****", "Spine Bone");
		map.put("*****lum_vert2a****", "Spine Bone");
		map.put("*****lum_vert2b****", "Spine Bone");
		map.put("*****lum_vert3a****", "Spine Bone");
		map.put("*****lum_vert3b****", "Spine Bone");
		map.put("*****lum_vert4a****", "Spine Bone");
		map.put("*****lum_vert4b****", "Spine Bone");
		map.put("*****lum_vert5a****", "Spine Bone");
		map.put("*****lum_vert5b****", "Spine Bone");
		map.put("*****sacrum****", "Spine Bone");
		map.put("*****sacrum_cart****", "Cartilage");
		map.put("***RRIB1-Cartilage", "Cartilage");
		map.put("***RRIB2-Cartilage", "Cartilage");
		map.put("***RRIB3-Cartilage", "Cartilage");
		map.put("***RRIB4-Cartilage", "Cartilage");
		map.put("***RRIB5-Cartilage", "Cartilage");
		map.put("***RRIB6-Cartilage", "Cartilage");
		map.put("***RRIB7-Cartilage", "Cartilage");
		map.put("***RRIB8-Cartilage", "Cartilage");
		map.put("***RRIB9-Cartilage", "Cartilage");
		map.put("***Right-Sternum-Cartilage", "Cartilage");
		map.put("***RRIB1", "Rib Bone");
		map.put("***RRIB2", "Rib Bone");
		map.put("***RRIB3", "Rib Bone");
		map.put("***RRIB4", "Rib Bone");
		map.put("***RRIB5", "Rib Bone");
		map.put("***RRIB6", "Rib Bone");
		map.put("***RRIB7", "Rib Bone");
		map.put("***RRIB8", "Rib Bone");
		map.put("***RRIB9", "Rib Bone");
		map.put("***RRIB10", "Rib Bone");
		map.put("***RRIB11", "Rib Bone");
		map.put("***RRIB12", "Rib Bone");
		map.put("***LRIB1-Cartilage", "Cartilage");
		map.put("***LRIB2-Cartilage", "Cartilage");
		map.put("***LRIB3-Cartilage", "Cartilage");
		map.put("***LRIB4-Cartilage", "Cartilage");
		map.put("***LRIB5-Cartilage", "Cartilage");
		map.put("***LRIB6-Cartilage", "Cartilage");
		map.put("***LRIB7-Cartilage", "Cartilage");
		map.put("***LRIB8-Cartilage", "Cartilage");
		map.put("***LRIB9-Cartilage", "Cartilage");
		map.put("***Left-Sternum-Cartilage", "Cartilage");
		map.put("***LRIB1", "Rib Bone");
		map.put("***LRIB2", "Rib Bone");
		map.put("***LRIB3", "Rib Bone");
		map.put("***LRIB4", "Rib Bone");
		map.put("***LRIB5", "Rib Bone");
		map.put("***LRIB6", "Rib Bone");
		map.put("***LRIB7", "Rib Bone");
		map.put("***LRIB8", "Rib Bone");
		map.put("***LRIB9", "Rib Bone");
		map.put("***LRIB10", "Rib Bone");
		map.put("***LRIB11", "Rib Bone");
		map.put("***LRIB12", "Rib Bone");
		// bones right arm
		map.put("**RightHumerus", "Bone");
		map.put("**RightRadius", "Bone");
		map.put("**RightUlna", "Bone");
		map.put("**RightCapitate", "Bone");
		map.put("**RightHamate", "Bone");
		map.put("**RightLunate", "Bone");
		map.put("**RightMetacarpal", "Bone");
		map.put("**RightMetacarpal", "Bone");
		map.put("**RightMetacarpal", "Bone");
		map.put("**RightMetacarpal", "Bone");
		map.put("**RightMetacarpal", "Bone");
		map.put("**RightPisiform", "Bone");
		map.put("**RightScaphoid", "Bone");
		map.put("**RightTrapezium", "Bone");
		map.put("**RightTriquetrum", "Bone");
		map.put("**RightTrapezoid", "Bone");
		map.put("**RightFinger1", "Bone");
		map.put("**RightFinger2", "Bone");
		map.put("**RightFinger3", "Bone");
		map.put("**RightFinger4", "Bone");
		map.put("**RightFinger5", "Bone");
		map.put("**RightFinger6", "Bone");
		map.put("**RightFinger7", "Bone");
		map.put("**RightFinger8", "Bone");
		map.put("**RightFinger9", "Bone");
		map.put("**RightFinger10", "Bone");
		map.put("**RightFinger11", "Bone");
		map.put("**RightFinger12", "Bone");
		map.put("**RightFinger13", "Bone");
		map.put("**RightFinger14", "Bone");
		// bones left arm
		map.put("**LeftHumerus", "Bone");
		map.put("**LeftRadius", "Bone");
		map.put("**LeftUlna", "Bone");
		map.put("**LeftCapitate", "Bone");
		map.put("**LeftHamate", "Bone");
		map.put("**LeftLunate", "Bone");
		map.put("**LeftMetacarpal", "Bone");
		map.put("**LeftMetacarpal", "Bone");
		map.put("**LeftMetacarpal", "Bone");
		map.put("**LeftMetacarpal", "Bone");
		map.put("**LeftMetacarpal", "Bone");
		map.put("**LeftPisiform", "Bone");
		map.put("**LeftScaphoid", "Bone");
		map.put("**LeftTrapezium", "Bone");
		map.put("**LeftTriquetrum", "Bone");
		map.put("**LeftTrapezoid", "Bone");
		map.put("**LeftFinger1", "Bone");
		map.put("**LeftFinger2", "Bone");
		map.put("**LeftFinger3", "Bone");
		map.put("**LeftFinger4", "Bone");
		map.put("**LeftFinger5", "Bone");
		map.put("**LeftFinger6", "Bone");
		map.put("**LeftFinger7", "Bone");
		map.put("**LeftFinger8", "Bone");
		map.put("**LeftFinger9", "Bone");
		map.put("**LeftFinger10", "Bone");
		map.put("**LeftFinger11", "Bone");
		map.put("**LeftFinger12", "Bone");
		map.put("**LeftFinger13", "Bone");
		map.put("**LeftFinger14", "Bone");
		map.put("**RightFemur", "Bone");
		map.put("**RightTibia", "Bone");
		map.put("**RightFibula", "Bone");
		map.put("**RightPatella", "Bone");
		map.put("**RightTalus", "Bone");
		map.put("**RightCalcaneus", "Bone");
		map.put("**RightCuboid", "Bone");
		map.put("**Right_inter_cuneiform", "Bone");
		map.put("**Right_lat_cuneiform", "Bone");
		map.put("**Right_med_cuneiform", "Bone");
		map.put("**RightTarsal", "Bone");
		map.put("**RightTarsal", "Bone");
		map.put("**RightTarsal", "Bone");
		map.put("**RightTarsal", "Bone");
		map.put("**RightTarsal", "Bone");
		map.put("**RightNavicular", "Bone");
		map.put("RightToe1", "Bone");
		map.put("RightToe2", "Bone");
		map.put("RightToe3", "Bone");
		map.put("RightToe4", "Bone");
		map.put("RightToe5", "Bone");
		map.put("RightToe6", "Bone");
		map.put("RightToe7", "Bone");
		map.put("RightToe8", "Bone");
		map.put("RightToe9", "Bone");
		map.put("RightToe10", "Bone");
		map.put("RightToe11", "Bone");
		map.put("RightToe12", "Bone");
		map.put("RightToe13", "Bone");
		map.put("RightToe14", "Bone");
		map.put("*LeftFemur", "Bone");
		map.put("**LeftTibia", "Bone");
		map.put("**LeftFibula", "Bone");
		map.put("**LeftPatella", "Bone");
		map.put("**LeftTalus", "Bone");
		map.put("**LeftCalcaneus", "Bone");
		map.put("**LeftCuboid", "Bone");
		map.put("**Left_inter_cuneiform", "Bone");
		map.put("**Left_lat_cuneiform", "Bone");
		map.put("**Left_med_cuneiform", "Bone");
		map.put("**LeftTarsal", "Bone");
		map.put("**LeftTarsal", "Bone");
		map.put("**LeftTarsal", "Bone");
		map.put("**LeftTarsal", "Bone");
		map.put("**LeftTarsal", "Bone");
		map.put("**LeftNavicular", "Bone");
		map.put("LeftToe1", "Bone");
		map.put("LeftToe2", "Bone");
		map.put("LeftToe3", "Bone");
		map.put("LeftToe4", "Bone");
		map.put("LeftToe5", "Bone");
		map.put("LeftToe6", "Bone");
		map.put("LeftToe7", "Bone");
		map.put("LeftToe8", "Bone");
		map.put("LeftToe9", "Bone");
		map.put("LeftToe10", "Bone");
		map.put("LeftToe11", "Bone");
		map.put("LeftToe12", "Bone");
		map.put("LeftToe13", "Bone");
		map.put("LeftToe14", "Bone");
		map.put("***RIGHT_COLLARBONE***", "Bone");
		map.put("***RIGHT_SCAPULA***", "Bone");
		map.put("****LEFT_COLLARBONE****", "Bone");
		map.put("****LEFT_SCAPULA****", "Bone");
		map.put("****skull_face1****", "Skull Bone");
		map.put("****skull_face2****", "Skull Bone");
		map.put("****skull1****", "Skull Bone");
		map.put("****skull2****", "Skull Bone");
		map.put("****skull3****", "Skull Bone");
		map.put("****skull4****", "Skull Bone");
		map.put("****skull5****", "Skull Bone");
		map.put("****jaw****", "Skull Bone");
		map.put("****nasal_passage1****", "Skull Bone");
		map.put("****nasal_passage2****", "Skull Bone");
		map.put("****brain****", "Brain");
		map.put("****cerebellum****", "Brain");
		map.put("****brain_stem****", "Brain");
		if (!maleGender){
			if (!supine) {
				map.put("****RBREAST-Prone*****", "Body (water)");
				map.put("****LBREAST-Prone*****", "Body (water)"); 
			} else {
				map.put("****RBREAST-Supine*****", "Body (water)");
				map.put("****LBREAST-Supine*****", "Body (water)");
			}
		}
		map.put("*****RHIP*****", "Bone");
		map.put("*****RPELVIS1*****", "Bone");
		map.put("*****RPELVIS2*****", "Bone");
		map.put("*****RHIPC*****", "Bone");
		map.put("*****RPELVIS1C*****", "Bone");
		map.put("*****RPELVIS2C*****", "Bone");
		map.put("*****LHIP*****", "Bone");
		map.put("*****LPELVIS1*****", "Bone");
		map.put("*****LPELVIS2*****", "Bone");
		map.put("*****LHIPC*****", "Bone");
		map.put("*****LPELVIS1C*****", "Bone");
		map.put("*****LPELVIS2C*****", "Bone");
		map.put("*****R_URETER*****", "Intestine");
		map.put("*****L_URETER*****", "Intestine");
		map.put("*****BLADDER*****", "Intestine");
		map.put("*****R_VAS_DEF*****", "Intestine");
		map.put("*****L_VAS_DEF*****", "Intestine");
		if (maleGender) {
			map.put("*****SEMINAL_VES*****", "Intestine");
			map.put("*****PROSTATE*****", "Intestine");
			map.put("*****URETHRA*****", "Intestine");
			map.put("*****R_TEST*****", "Intestine");
			map.put("*****L_TEST*****", "Body (water)");
			map.put("*****PENIS*****", "Body (water)");
		} else {
			map.put("****UTERUS*****", "Lymph");
			map.put("****RIGHT_OVARY*****", "Lymph");
			map.put("****RIGHT_OVARY2*****", "Lymph");
			map.put("****R_FL_TUBE*****", "Lymph");
			map.put("****LEFT_OVARY*****", "Lymph");
			map.put("****LEFT_OVARY2*****", "Lymph");
			map.put("****L_FL_TUBE*****", "Lymph");
			map.put("****VAGINA1*****", "Body (water)");
			map.put("****VAGINA2*****", "Body (water)");
		}
		map.put("*****ASC_LARGE_INTEST*****", "Intestine");
		map.put("*****TRANS_LARGE_INTEST*****", "Intestine");
		map.put("*****DESC_LARGE_INTEST*****", "Intestine");
		map.put("*****RECTUM*****", "Body (water)");
		map.put("*****ASC_LARGE_INTEST_AIR*****", "Air");
		map.put("*****TRANS_LARGE_INTEST_AIR*****", "Air");
		map.put("*****DESC_LARGE_INTEST_AIR*****", "Air");
		map.put("*****RECTUM_AIR*****", "Air");
		map.put("****small_intest_0****", "Intestine");
		map.put("****small_intest_1****", "Intestine");
		map.put("****small_intest_2****", "Intestine");
		map.put("****small_intest_3****", "Intestine");
		map.put("****small_intest_4****", "Intestine");
		map.put("****small_intest_5****", "Intestine");
		map.put("****small_intest_6****", "Intestine");
		map.put("****small_intest_7****", "Intestine");
		map.put("****small_intest_8****", "Intestine");
		map.put("****small_intest_9****", "Intestine");
		map.put("****small_intest_10****", "Intestine");
		map.put("****small_intest_11****", "Intestine");
		map.put("****small_intest_12****", "Intestine");
		map.put("****small_intest_13****", "Intestine");
		map.put("****small_intest_14****", "Intestine");
		map.put("****small_intest_15****", "Intestine");
		map.put("****small_intest_16****", "Intestine");
		map.put("****small_intest_17****", "Intestine");
		map.put("****small_intest_18****", "Intestine");
		map.put("****small_intest_19****", "Intestine");
		map.put("****small_intest_20****", "Intestine");
		map.put("****small_intest_21****", "Intestine");
		map.put("****small_intest_22****", "Intestine");
		map.put("****small_intest_23****", "Intestine");
		map.put("****small_intest_24****", "Intestine");
		map.put("****small_intest_25****", "Intestine");
		map.put("****small_intest_26****", "Intestine");
		map.put("****small_intest_27****", "Intestine");
		map.put("****small_intest_28****", "Intestine");
		map.put("****small_intest_29****", "Intestine");
		map.put("****small_intest_30****", "Intestine");
		map.put("****small_intest_31****", "Intestine");
		map.put("****small_intest_32****", "Intestine");
		map.put("****small_intest_33****", "Intestine");
		map.put("****small_intest_34****", "Intestine");
		map.put("****small_intest_35****", "Intestine");
		map.put("****small_intest_36****", "Intestine");
		map.put("****small_intest_37****", "Intestine");
		map.put("****small_intest_38****", "Intestine");
		map.put("****small_intest_39****", "Intestine");
		map.put("****small_intest_40****", "Intestine");
		map.put("****small_intest_41****", "Intestine");
		map.put("****small_intest_42****", "Intestine");
		map.put("****small_intest_air_0****", "Air");
		map.put("****small_intest_air_1****", "Air");
		map.put("****small_intest_air_2****", "Air");
		map.put("****small_intest_air_3****", "Air");
		map.put("****small_intest_air_4****", "Air");
		map.put("****small_intest_air_5****", "Air");
		map.put("****small_intest_air_6****", "Air");
		map.put("****small_intest_air_7****", "Air");
		map.put("****small_intest_air_8****", "Air");
		map.put("****small_intest_air_9****", "Air");
		map.put("****small_intest_air_10****", "Air");
		map.put("****small_intest_air_11****", "Air");
		map.put("****small_intest_air_12****", "Air");
		map.put("****small_intest_air_13****", "Air");
		map.put("****small_intest_air_14****", "Air");
		map.put("****small_intest_air_15****", "Air");
		map.put("****small_intest_air_16****", "Air");
		map.put("****small_intest_air_17****", "Air");
		map.put("****small_intest_air_18****", "Air");
		map.put("****small_intest_air_19****", "Air");
		map.put("****small_intest_air_20****", "Air");
		map.put("****small_intest_air_21****", "Air");
		map.put("****small_intest_air_22****", "Air");
		map.put("****small_intest_air_23****", "Air");
		map.put("****small_intest_air_24****", "Air");
		map.put("****small_intest_air_25****", "Air");
		map.put("****small_intest_air_26****", "Air");
		map.put("****small_intest_air_27****", "Air");
		map.put("****small_intest_air_28****", "Air");
		map.put("****small_intest_air_29****", "Air");
		map.put("****small_intest_air_30****", "Air");
		map.put("****small_intest_air_31****", "Air");
		map.put("****small_intest_air_32****", "Air");
		map.put("****small_intest_air_33****", "Air");
		map.put("****small_intest_air_34****", "Air");
		map.put("****small_intest_air_35****", "Air");
		map.put("****small_intest_air_36****", "Air");
		map.put("****small_intest_air_37****", "Air");
		map.put("****small_intest_air_38****", "Air");
		map.put("****small_intest_air_39****", "Air");
		map.put("****small_intest_air_40****", "Air");
		map.put("****small_intest_air_41****", "Air");
		map.put("****small_intest_air_42****", "Air");
		map.put("*****AORTA*****", "Blood (Aorta)");
		map.put("*****Artery1*****", "Blood Artery");
		map.put("*****Artery2*****", "Blood Artery");
		map.put("*****Artery3*****", "Blood Artery");
		map.put("*****Artery4*****", "Blood Artery");
		map.put("*****Artery5*****", "Blood Artery");
		map.put("*****Artery6*****", "Blood Artery");
		map.put("*****Artery7*****", "Blood Artery");
		map.put("*****Artery8*****", "Blood Artery");
		map.put("*****Artery9*****", "Blood Artery");
		map.put("*****Artery10*****", "Blood Artery");
		map.put("*****Artery11*****", "Blood Artery");
		map.put("*****Artery12*****", "Blood Artery");
		map.put("*****Artery13*****", "Blood Artery");
		map.put("*****L_KIDNEY_ART1*****", "Blood Kid Artery");
		map.put("*****L_KIDNEY_ART2*****", "Blood Kid Artery");
		map.put("*****R_KIDNEY_ART1*****", "Blood Kid Artery");
		map.put("*****R_KIDNEY_ART2*****", "Blood Kid Artery");
		map.put("*****INFERIOR_VENA_CAVA*****", "Blood Vena Cava");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****VEIN*****", "Blood Vein");
		map.put("*****LIVER_VEINS*****", "Blood Liver Vein");
		map.put("*****LIVER_VEINS*****", "Blood Liver Vein");
		map.put("*****LIVER_VEINS*****", "Blood Liver Vein");
		map.put("*****LIVER_VEINS*****", "Blood Liver Vein");
		map.put("*****L_KIDNEY_VEIN*****","Blood Kidney Vein");
		map.put("*****R_KIDNEY_VEIN*****", "Blood Kidney Vein");
		map.put("*****PREAORTIC_NODE*****", "Lymph");
		map.put("*****COMMON_ILLIAC_NODE1*****", "Lymph");
		map.put("*****COMMON_ILLIAC_NODE2*****", "Lymph");
		map.put("*****COMMON_ILLIAC_NODE3*****", "Lymph");
		map.put("*****COMMON_ILLIAC_NODE4*****", "Lymph");
		map.put("*****RIGHT_EXT_ILLIAC_NODE1*****", "Lymph");
		map.put("*****RIGHT_EXT_ILLIAC_NODE2*****", "Lymph");
		map.put("*****RIGHT_EXT_ILLIAC_NODE3*****", "Lymph");
		map.put("*****RIGHT_EXT_ILLIAC_NODE4*****", "Lymph");
		map.put("*****RIGHT_EXT_ILLIAC_NODE5*****", "Lymph");
		map.put("*****RIGHT_HYPOGASTRIC_NODE*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE2*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE3*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE4*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE5*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE6*****", "Lymph");
		map.put("*****RIGHT_SUPERF_SUBINGUINAL_NODE7*****", "Lymph");
		map.put("*****LEFT_EXT_ILLIAC_NODE1*****", "Lymph");
		map.put("*****LEFT_EXT_ILLIAC_NODE2*****", "Lymph");
		map.put("*****LEFT_EXT_ILLIAC_NODE3*****", "Lymph");
		map.put("*****LEFT_EXT_ILLIAC_NODE4*****", "Lymph");
		map.put("*****LEFT_EXT_ILLIAC_NODE5*****", "Lymph");
		map.put("*****LEFT_HYPOGASTRIC_NODE*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("*****LEFT_SUPERF_SUBINGUINAL_NODE1*****", "Lymph");
		map.put("****Airway-1*********", "Air");
		map.put("*****PULMONARY_ART1*****", "Blood Pul Artery");
		map.put("*****PULMONARY_VEIN1*****", "Blood Pul Vein");
		if (renderMarrow) {
			map.put("**RRIB1_MARROW**", "Bone Marrow");
			map.put("**RRIB2_MARROW**", "Bone Marrow");
			map.put("**RRIB3_MARROW**", "Bone Marrow");
			map.put("**RRIB4_MARROW**", "Bone Marrow");
			map.put("**RRIB5_MARROW**", "Bone Marrow");
			map.put("**RRIB6_MARROW**", "Bone Marrow");
			map.put("**RRIB7_MARROW**", "Bone Marrow");
			map.put("**RRIB8_MARROW**", "Bone Marrow");
			map.put("**RRIB9_MARROW**", "Bone Marrow");
			map.put("**RRIB10_MARROW**", "Bone Marrow");
			map.put("**RRIB11_MARROW**", "Bone Marrow");
			map.put("**RRIB12_MARROW**", "Bone Marrow");
			map.put("**LRIB1_MARROW**", "Bone Marrow");
			map.put("**LRIB2_MARROW**", "Bone Marrow");
			map.put("**LRIB3_MARROW**", "Bone Marrow");
			map.put("**LRIB4_MARROW**", "Bone Marrow");
			map.put("**LRIB5_MARROW**", "Bone Marrow");
			map.put("**LRIB6_MARROW**", "Bone Marrow");
			map.put("**LRIB7_MARROW**", "Bone Marrow");
			map.put("**LRIB8_MARROW**", "Bone Marrow");
			map.put("**LRIB9_MARROW**", "Bone Marrow");
			map.put("**LRIB10_MARROW**", "Bone Marrow");
			map.put("**LRIB11_MARROW**", "Bone Marrow");
			map.put("**LRIB12_MARROW**", "Bone Marrow");
			map.put("**BACKBONE1_MARROW**", "Bone Marrow");
			map.put("**BACKBONE2_MARROW**", "Bone Marrow");
			map.put("**BACKBONE3_MARROW**", "Bone Marrow");
			map.put("**BACKBONE4_MARROW**", "Bone Marrow");
			map.put("**BACKBONE5_MARROW**", "Bone Marrow");
			map.put("**BACKBONE6_MARROW**", "Bone Marrow");
			map.put("**BACKBONE7_MARROW**", "Bone Marrow");
			map.put("**BACKBONE8_MARROW**", "Bone Marrow");
			map.put("**BACKBONE9_MARROW**", "Bone Marrow");
			map.put("**BACKBONE10_MARROW**", "Bone Marrow");
			map.put("**BACKBONE11_MARROW**", "Bone Marrow");
			map.put("**BACKBONE12_MARROW**", "Bone Marrow");
			map.put("**BACKBONE13_MARROW**", "Bone Marrow");
			map.put("**BACKBONE14_MARROW**", "Bone Marrow");
			map.put("**BACKBONE15_MARROW**", "Bone Marrow");
			map.put("**BACKBONE16_MARROW**", "Bone Marrow");
			map.put("**BACKBONE17_MARROW**", "Bone Marrow");
			map.put("**BACKBONE18_MARROW**", "Bone Marrow");
			map.put("**BACKBONE19_MARROW**", "Bone Marrow");
			map.put("**BACKBONE20_MARROW**", "Bone Marrow");
			map.put("**BACKBONE21_MARROW**", "Bone Marrow");
			map.put("**BACKBONE22_MARROW**", "Bone Marrow");
			map.put("**BACKBONE23_MARROW**", "Bone Marrow");
			map.put("**BACKBONE24_MARROW**", "Bone Marrow");
			map.put("**BACKBONE25_MARROW**", "Bone Marrow");
			map.put("**BACKBONE26_MARROW**", "Bone Marrow");
			map.put("**BACKBONE27_MARROW**", "Bone Marrow");
			map.put("**BACKBONE28_MARROW**", "Bone Marrow");
			map.put("**BACKBONE29_MARROW**", "Bone Marrow");
			map.put("**BACKBONE30_MARROW**", "Bone Marrow");
			map.put("**BACKBONE31_MARROW**", "Bone Marrow");
			map.put("**BACKBONE32_MARROW**", "Bone Marrow");
			map.put("**BACKBONE33_MARROW**", "Bone Marrow");
			map.put("**BACKBONE34_MARROW**", "Bone Marrow");
			map.put("**BACKBONE35_MARROW**", "Bone Marrow");
			map.put("**BACKBONE36_MARROW**", "Bone Marrow");
			map.put("**BACKBONE37_MARROW**", "Bone Marrow");
			map.put("**BACKBONE38_MARROW**", "Bone Marrow");
			map.put("**BACKBONE39_MARROW**", "Bone Marrow");
			map.put("**BACKBONE40_MARROW**", "Bone Marrow");
			map.put("**BACKBONE41_MARROW**", "Bone Marrow");
			map.put("**BACKBONE42_MARROW**", "Bone Marrow");
			map.put("**BACKBONE43_MARROW**", "Bone Marrow");
			map.put("**BACKBONE44_MARROW**", "Bone Marrow");
			map.put("**BACKBONE45_MARROW**", "Bone Marrow");
			map.put("**BACKBONE46_MARROW**", "Bone Marrow");
			map.put("**BACKBONE47_MARROW**", "Bone Marrow");
			map.put("**BACKBONE48_MARROW**", "Bone Marrow");
			map.put("**BACKBONE49_MARROW**", "Bone Marrow");
			map.put("**BACKBONE50_MARROW**", "Bone Marrow");
			map.put("**BACKBONE51_MARROW**", "Bone Marrow");
			map.put("**BACKBONE52_MARROW**", "Bone Marrow");
			map.put("**BACKBONE53_MARROW**", "Bone Marrow");
			map.put("**BACKBONE54_MARROW**", "Bone Marrow");
			map.put("**BACKBONE55_MARROW**", "Bone Marrow");
			map.put("**BACKBONE56_MARROW**", "Bone Marrow");
			map.put("**BACKBONE57_MARROW**", "Bone Marrow");
			map.put("**BACKBONE58_MARROW**", "Bone Marrow");
			map.put("**STERN_MARROW**", "Bone Marrow");
			map.put("**RSCAP_MARROW**", "Bone Marrow");
			map.put("**RCOLLAR_MARROW**", "Bone Marrow");
			map.put("**LSCAP_MARROW**", "Bone Marrow");
			map.put("**LCOLLAR_MARROW**", "Bone Marrow");
			if (renderArmMarrow){
				map.put("**RHUMERUS_MARROW**", "Bone Marrow");
				map.put("**R_RADIUS_MARROW**", "Bone Marrow");
				map.put("**R_ULNA_MARROW**", "Bone Marrow");
				map.put("**LHUMERUS_MARROW**", "Bone Marrow");
				map.put("**L_RADIUS_MARROW**", "Bone Marrow");
				map.put("**L_ULNA_MARROW**", "Bone Marrow");
				map.put("**R_HAND_MARROW0", "Bone Marrow");
				map.put("**R_HAND_MARROW1", "Bone Marrow");
				map.put("**R_HAND_MARROW2", "Bone Marrow");
				map.put("**R_HAND_MARROW3", "Bone Marrow");
				map.put("**R_HAND_MARROW4", "Bone Marrow");
				map.put("**R_HAND_MARROW5", "Bone Marrow");
				map.put("**R_HAND_MARROW6", "Bone Marrow");
				map.put("**R_HAND_MARROW7", "Bone Marrow");
				map.put("**R_HAND_MARROW8", "Bone Marrow");
				map.put("**R_HAND_MARROW9", "Bone Marrow");
				map.put("**R_HAND_MARROW10", "Bone Marrow");
				map.put("**R_HAND_MARROW11", "Bone Marrow");
				map.put("**R_HAND_MARROW12", "Bone Marrow");
				map.put("**R_HAND_MARROW13", "Bone Marrow");
				map.put("**R_HAND_MARROW14", "Bone Marrow");
				map.put("**R_HAND_MARROW15", "Bone Marrow");
				map.put("**R_HAND_MARROW16", "Bone Marrow");
				map.put("**R_HAND_MARROW17", "Bone Marrow");
				map.put("**R_HAND_MARROW18", "Bone Marrow");
				map.put("**R_HAND_MARROW19", "Bone Marrow");
				map.put("**R_HAND_MARROW20", "Bone Marrow");
				map.put("**R_HAND_MARROW21", "Bone Marrow");
				map.put("**R_HAND_MARROW22", "Bone Marrow");
				map.put("**R_HAND_MARROW23", "Bone Marrow");
				map.put("**R_HAND_MARROW24", "Bone Marrow");
				map.put("**R_HAND_MARROW25", "Bone Marrow");
				map.put("**R_HAND_MARROW26", "Bone Marrow");
				map.put("**L_HAND_MARROW0", "Bone Marrow");
				map.put("**L_HAND_MARROW1", "Bone Marrow");
				map.put("**L_HAND_MARROW2", "Bone Marrow");
				map.put("**L_HAND_MARROW3", "Bone Marrow");
				map.put("**L_HAND_MARROW4", "Bone Marrow");
				map.put("**L_HAND_MARROW5", "Bone Marrow");
				map.put("**L_HAND_MARROW6", "Bone Marrow");
				map.put("**L_HAND_MARROW7", "Bone Marrow");
				map.put("**L_HAND_MARROW8", "Bone Marrow");
				map.put("**L_HAND_MARROW9", "Bone Marrow");
				map.put("**L_HAND_MARROW10", "Bone Marrow");
				map.put("**L_HAND_MARROW11", "Bone Marrow");
				map.put("**L_HAND_MARROW12", "Bone Marrow");
				map.put("**L_HAND_MARROW13", "Bone Marrow");
				map.put("**L_HAND_MARROW14", "Bone Marrow");
				map.put("**L_HAND_MARROW15", "Bone Marrow");
				map.put("**L_HAND_MARROW16", "Bone Marrow");
				map.put("**L_HAND_MARROW17", "Bone Marrow");
				map.put("**L_HAND_MARROW18", "Bone Marrow");
				map.put("**L_HAND_MARROW19", "Bone Marrow");
				map.put("**L_HAND_MARROW20", "Bone Marrow");
				map.put("**L_HAND_MARROW21", "Bone Marrow");
				map.put("**L_HAND_MARROW22", "Bone Marrow");
				map.put("**L_HAND_MARROW23", "Bone Marrow");
				map.put("**L_HAND_MARROW24", "Bone Marrow");
				map.put("**L_HAND_MARROW25", "Bone Marrow");
				map.put("**L_HAND_MARROW26", "Bone Marrow");
			}
			map.put("**R_FOOT_MARROW0", "Bone Marrow");
			map.put("**R_FOOT_MARROW1", "Bone Marrow");
			map.put("**R_FOOT_MARROW2", "Bone Marrow");
			map.put("**R_FOOT_MARROW3", "Bone Marrow");
			map.put("**R_FOOT_MARROW4", "Bone Marrow");
			map.put("**R_FOOT_MARROW5", "Bone Marrow");
			map.put("**R_FOOT_MARROW6", "Bone Marrow");
			map.put("**R_FOOT_MARROW7", "Bone Marrow");
			map.put("**R_FOOT_MARROW8", "Bone Marrow");
			map.put("**R_FOOT_MARROW9", "Bone Marrow");
			map.put("**R_FOOT_MARROW10", "Bone Marrow");
			map.put("**R_FOOT_MARROW11", "Bone Marrow");
			map.put("**R_FOOT_MARROW12", "Bone Marrow");
			map.put("**R_FOOT_MARROW13", "Bone Marrow");
			map.put("**R_FOOT_MARROW14", "Bone Marrow");
			map.put("**R_FOOT_MARROW15", "Bone Marrow");
			map.put("**R_FOOT_MARROW16", "Bone Marrow");
			map.put("**R_FOOT_MARROW17", "Bone Marrow");
			map.put("**R_FOOT_MARROW18", "Bone Marrow");
			map.put("**R_FOOT_MARROW19", "Bone Marrow");
			map.put("**R_FOOT_MARROW20", "Bone Marrow");
			map.put("**R_FOOT_MARROW21", "Bone Marrow");
			map.put("**R_FOOT_MARROW22", "Bone Marrow");
			map.put("**R_FOOT_MARROW23", "Bone Marrow");
			map.put("**R_FOOT_MARROW24", "Bone Marrow");
			map.put("**R_FOOT_MARROW25", "Bone Marrow");
			map.put("**L_FOOT_MARROW0", "Bone Marrow");
			map.put("**L_FOOT_MARROW1", "Bone Marrow");
			map.put("**L_FOOT_MARROW2", "Bone Marrow");
			map.put("**L_FOOT_MARROW3", "Bone Marrow");
			map.put("**L_FOOT_MARROW4", "Bone Marrow");
			map.put("**L_FOOT_MARROW5", "Bone Marrow");
			map.put("**L_FOOT_MARROW6", "Bone Marrow");
			map.put("**L_FOOT_MARROW7", "Bone Marrow");
			map.put("**L_FOOT_MARROW8", "Bone Marrow");
			map.put("**L_FOOT_MARROW9", "Bone Marrow");
			map.put("**L_FOOT_MARROW10", "Bone Marrow");
			map.put("**L_FOOT_MARROW11", "Bone Marrow");
			map.put("**L_FOOT_MARROW12", "Bone Marrow");
			map.put("**L_FOOT_MARROW13", "Bone Marrow");
			map.put("**L_FOOT_MARROW14", "Bone Marrow");
			map.put("**L_FOOT_MARROW15", "Bone Marrow");
			map.put("**L_FOOT_MARROW16", "Bone Marrow");
			map.put("**L_FOOT_MARROW17", "Bone Marrow");
			map.put("**L_FOOT_MARROW18", "Bone Marrow");
			map.put("**L_FOOT_MARROW19", "Bone Marrow");
			map.put("**L_FOOT_MARROW20", "Bone Marrow");
			map.put("**L_FOOT_MARROW21", "Bone Marrow");
			map.put("**L_FOOT_MARROW22", "Bone Marrow");
			map.put("**L_FOOT_MARROW23", "Bone Marrow");
			map.put("**L_FOOT_MARROW24", "Bone Marrow");
			map.put("**L_FOOT_MARROW25", "Bone Marrow");
			map.put("**R_FEMUR_MARROW**", "Bone Marrow");
			map.put("**R_TIBIA_MARROW**", "Bone Marrow");
			map.put("**R_FIBULA_MARROW**", "Bone Marrow");
			map.put("**R_PATELLA_MARROW**", "Bone Marrow");
			map.put("**L_FEMUR_MARROW**", "Bone Marrow");
			map.put("**L_TIBIA_MARROW**", "Bone Marrow");
			map.put("**L_FIBULA_MARROW**", "Bone Marrow");
			map.put("**L_PATELLA_MARROW**", "Bone Marrow");
		}
		// heart
		map.put("****Right-atria-myo*********", "Heart");
		map.put("****Right-ventricle-myo*********", "Heart");
		map.put("****Right-atria-chamber****", "Blood");
		map.put("****Right-ventricle-chamber****", "Blood");
		map.put("****Left-atria-myo**********", "Heart");
		map.put("****Left-ventricle-myo**********", "Heart");
		map.put("****Left-atria-chamber*****", "Blood");

		String key = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_VENTRICLE_SELECTION);
		if (key!= null) {
			if (key.contains("1")){
				map.put("****Left-ventricle-chamber1*****", "Blood (LV)");
			} else {
				map.put("****Left-ventricle-chamber1*****", "Heart");
			}
			if (key.contains("2")){
				map.put("****Left-ventricle-chamber2*****", "Blood (LV)");
			} else {
				map.put("****Left-ventricle-chamber2*****", "Heart");
			}
			if (key.contains("3")){
				map.put("****Left-ventricle-chamber3*****", "Blood (LV)");
			} else {
				map.put("****Left-ventricle-chamber3*****", "Heart");
			}
			if (key.contains("4")){
				map.put("****Left-ventricle-chamber4*****", "Blood (LV)");
			} else {
				map.put("****Left-ventricle-chamber4*****", "Heart");
			}
		} else {
			map.put("****Left-ventricle-chamber1*****", "Blood (LV)");
			map.put("****Left-ventricle-chamber2*****", "Blood (LV)");
			map.put("****Left-ventricle-chamber3*****", "Blood (LV)");
			map.put("****Left-ventricle-chamber4*****", "Blood (LV)");
		}
		map.put("****Heartwall*********", "Heart");
		if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED)!= null) {
			if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED).equals("true")) {
			map.put("/***RCA1***/", "Blood");
			map.put("/***RCA2***/", "Blood");
			map.put("/***LCA1***/", "Coronary Artery");
			map.put("/***LCA2***/", "Coronary Artery");
			map.put("/***LCA3***/", "Coronary Artery");
			map.put("/***LCA4***/", "Coronary Artery");
			map.put("/***LCA5***/", "Coronary Artery");
			map.put("/***LCA6***/", "Coronary Artery");
			map.put("/***LCA7***/", "Coronary Artery");
			map.put("/***LCA8***/", "Coronary Artery");
			map.put("/***LCA9***/", "Coronary Artery");
			map.put("/***LCA10***/", "Coronary Artery");
			}
		}else{
			map.put("/***RCA1***/", "Coronary Artery");
			map.put("/***RCA2***/", "Coronary Artery");
			map.put("/***LCA1***/", "Coronary Artery");
			map.put("/***LCA2***/", "Coronary Artery");
			map.put("/***LCA3***/", "Coronary Artery");
			map.put("/***LCA4***/", "Coronary Artery");
			map.put("/***LCA5***/", "Coronary Artery");
			map.put("/***LCA6***/", "Coronary Artery");
			map.put("/***LCA7***/", "Coronary Artery");
			map.put("/***LCA8***/", "Coronary Artery");
			map.put("/***LCA9***/", "Coronary Artery");
			map.put("/***LCA10***/", "Coronary Artery");
		}
		
		map.put("Heart Lesion", "Heart Lesion");
		map.put("Heart Catheter", "Catheter Material");
		return map;
	}

	/**
	 * Lookup Material via the Spline name
	 * @param name
	 * @return the material
	 */
	public static Material generateFromSplineName(String name){
		String materialName = getSplineNameMaterialNameLUT().get(name);
		Material material = XCatMaterialGenerator.generateFromMaterialName(materialName);
		return material;
	}


	public static boolean exclude(String match){
		boolean includeVessels = false;
		if (includeVessels) {
			String materialName = getSplineNameMaterialNameLUT().get(match);
			if (materialName != null) {
				return ! (materialName.contains("Blood"));
			} else return true;
		} else return false;
	}

	
	@Override
	public TimeWarper getTimeWarper() {
		return warper;
	}

	@Override
	public void setTimeWarper(TimeWarper warp) {
		warper = warp;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/