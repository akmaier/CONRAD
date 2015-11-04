package edu.stanford.rsl.conrad.utils;

import java.util.HashMap;

/**
 * This class lists all of Conrad's registry entries with an explanation of their function.
 * New entries may be added here. However, they are not required to be listed here. As Conrad's
 * registry is a HashTable<String, String> any String may be used as key. The list here should
 * be used to prevent double usage of the same term and also explain how the value of the key
 * is intended to be used.
 * 
 * @author akmaier
 *
 */
public abstract class RegKeys {
	
	/**
	 * Registry entry to limit the number threads being used in the parallel processing.<br>
	 * The <b>value</b> of this key is interpreted as <b>int</b> and denotes the maximal number of threads.
	 */
	public static final String MAX_THREADS = "MAX_THREADS";
	
	/**
	 * Entry to set an offset of samples that are skipped in the processing of the VICON data, i.e. the first VICON_SKIP_SAMPLES frames of the data file are not processed.<BR>
	 * The <b>value</b> is a <b>String</b> of an Integer value in [samples]. 
	 */
	public static final String VICON_SKIP_SAMPLES = "VICON_SKIP_SAMPLES";
	
	/**
	 * Entry to set the VICON sampling rate. <BR>
	 * The <b>value</b> is a <b>String</b> of a Double value in [Hz]. 
	 */
	public static final String VICON_SAMPLING_RATE = "VICON_SAMPLING_RATE";
	
	/**
	 * Entry to set the projection sampling rate, i.e. the number of frames that are taken by the scanner in each second. <BR>
	 * The <b>value</b> is a <b>String</b> of a Double value in [Hz]. 
	 */
	public static final String PROJECTION_SAMPLING_RATE = "PROJECTION_SAMPLING_RATE";
	
	/**
	 * Entry to locate the data files required for XCat. <BR>
	 * The <b>value</b> is a <b>String</b> representation of the path to the XCat directory. 
	 */
	public static final String XCAT_PATH = "XCAT_PATH";
	
	/**
	 * Entry to change the contrast settings to artificial MRI for XCat. <BR>
	 * The <b>value</b> is a <b>String</b> representation of the flag, i.e. "true" or "false".
	 * Note that use use of this flag requires monochromatic projections. 
	 */
	public static final String XCAT_USE_MRI_CONTRAST_SETTINGS = "XCAT_USE_MRI_CONTRAST_SETTINGS";
	
	
	/**
	 * Entry to contrast the left ventricle in XCat with 2.5 [g/cm^3]. In addition the aorta is contrasted with 2.0 [g/cm^3] and the heart wall with 1.5 [g/cm^3]<BR>
	 * The <b>value</b> is a <b>String</b> representation of a boolean value. It is either "true" or "false". 
	 */
	public static final String XCAT_CONTRAST_LEFT_VENTRICLE = "XCAT_CONTRAST_LEFT_VENTRICLE";
	
	/**
	 * Entry to export the mesh to file system.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a file name.  
	 */
	public static final String XCAT_WRITE_MESH= "XCAT_WRITE_MESH";
	
	/**
	 * Reduce the number of exported meshes. Only meshes that match the substring are exported.<BR>
	 * The <b>value</b> is a <b>String</b>.  
	 */
	public static final String XCAT_WRITE_MESH_SURFACE_SUBSTRING= "XCAT_WRITE_MESH_SURFACE_SUBSTRING";
	
	
	/**
	 * Entry to configure a motion defect of the heart in XCat.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a two three dimensional points [x y z]. An example is "[1 1 1];[0 0 0]". The scaling is related to Xcat heart world coordinates.
	 * The two points define a bounding box that is aligned with the heart coordinate system of XCat.
	 * The box is given by the maximum and the minimum over the x, y, and z components of the points. 
	 */
	public static final String XCAT_HEART_MOTION_DEFECT = "XCAT_HEART_MOTION_DEFECT";
	
	/**
	 * Entry to configure the flexibility of the motion defect of the heart in XCat.<BR>
	 * The higher the value, the more flexible the defect will be. The parameter adjusts the standard deviation
	 * of the RBF interpolation that is performed at the boundary of the defect box. 
	 * The standard deviation is considered in spline u-v-space, that has a range in between 0 and 1.<BR>
	 * A value of 0.1 gives considerable stiffness. <BR>
	 * 0.05 is very stiff. <BR>
	 * 0.2 is rather flexible. <BR>
	 * The <b>value</b> is a <b>String</b> representation of a double value. 
	 */
	public static final String XCAT_HEART_MOTION_DEFECT_FLEXIBILITY = "XCAT_HEART_MOTION_DEFECT_FLEXIBILITY";

	/**
	 * Entry to configure a phase shift in the motion defect of the heart in XCat.<BR>
	 * The value is given in the interval [-1,1]. The parameter adjusts the phase shift in the motion defect relative to the phase.<BR>
	 * A value of 0.2 means that the motion is delayed by 20% of the heart phase. <BR>
	 * A phase shift will overwrite a stiffness defect.
	 * The <b>value</b> is a <b>String</b> representation of a double value. 
	 */
	public static final String XCAT_HEART_MOTION_DEFECT_PHASE_SHIFT = "XCAT_HEART_MOTION_DEFECT_PHASE_SHIFT";
	
	
	/**
	 * Entry to render moving heart lesions contrasted in XCat with 2.0 [g/cm^3]. 
	 * The <b>value</b> is a <b>String</b> representation of a boolean value. It is either "true" or "false". 
	 */
	public static final String XCAT_RENDER_HEART_LESIONS = "XCAT_RENDER_HEART_LESIONS";
	
	/**
	 * Entry to render constrasted coronary arteries in XCat with 4.0 [g/cm^3]. <BR>
	 * The <b>value</b> is a <b>String</b> representation of a boolean value. It is either "true" or "false". 
	 */
	public static final String XCAT_RENDER_CONTRASTED_CORONARY_ARTERIES = "XCAT_RENDER_CONTRASTED_CORONARY_ARTERIES";
	
	/**
	 * Sequence of three angles in degrees to rotate the heart of XCat. Rotation is meant to be interpreted as series of rotations around the X, Y, and Z axis plus an additional scaling factor. <BR>
	 * The <b>value</b> is a <b>String</b> representation of a four float values separated by ", ". An example would be "90.0, 0.0, 0.0, 1.0". 
	 */
	public static final String XCAT_HEART_ROTATION = "XCAT_HEART_ROTATION";
	
	/**
	 * Entry to allow simulation of contrasted coronary angiography where only the left artery tree is contrasted.
	 * The value is either "true" or "false". 
	 */
	public static final String XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED = "XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED"; 

	/**
	 * This key can be used to DISABLE the automatic centering of 4D phantoms which is applied by default during projection rendering for each projection.
	 * If not set or set to <b>false</b> all four dimensional phantoms, e.g. the XCAT phantom, are automatically centered during projection generation.
	 * If set to <b>true</b> the automatic centering is not applied during the projection rendering process.
	 */
	public static final String DISABLE_CENTERING_4DPHANTOM_PROJECTION_RENDERING = "DISABLE_CENTERING_4DPHANTOM_PROJECTION_RENDERING";
	

	/**
	 * Sequence of three coordinates in mm to apply a global translation to a 4D phantom in world coordinates for the purpose of projection rendering.	  
	 * The <b>value</b> is a <b>String</b> representation of 3 float values separated by a ", ". An example would be "90.0, 0.0, 0.0".
	 * (Note: This translation is valid to define the relative position of XCAT to the source and detector ONLY in 2d projection.)
	 */
	public static final String GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING = "GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING";
	
	/**
	 * If this key is set, the BreathingScene adds a catheter that is approximated from the aorta. The <b>value</b> of the key is of type <b>double</b> and describes the diameter of the catheter in [mm]. 
	 */
	public static final String XCAT_ADD_HEART_CATHETER = "XCAT_ADD_HEART_CATHETER";
	
	/**
	 * This key allows to translate the catheter tip using a 3D vector. The translation will be applied to the tip inside the XCAT heart. The tip is a u = 0. These control points will be affected completely by the translation. In the fourth entry of the vector the control point layer that is no longer affected by the translation is defined. It total the catheter has 30 control point layers. The <b>value</b> of the key is of type <b>String</b> and describes a 4D vector. "[40 -30 0 4]" will translate the tip by 40 mm in x-direction, -30 mm in y-direction and 0 in z-direction while the translation affects control points up to layer 3. These are good values to insert catheter tip into the left ventricle 
	 */
	public static final String XCAT_ADD_HEART_CATHETER_TIP_TRANSLATION = "XCAT_ADD_HEART_CATHETER_TIP_TRANSLATION";
	
	/**
	 * This flag is used in AnalyticPhantom3DVolumeRenderer.
	 * If it is set to true, the phantom is moved to the center for rendering.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a boolean value. "true" means the flag is set. 
	 * 
	 * @see edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantom3DVolumeRenderer
	 */
	public static final String RENDER_PHANTOM_VOLUME_AUTO_CENTER = "RENDER_PHANTOM_VOLUME_AUTO_CENTER";

	/**
	 * This flag is used in XCatScene.
	 * If it is set to true, the phantom is rezised to match the volume to be rendered in.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a boolean value. "true" means the flag is set.
	 * This behavior is set by default and must be disabled. 
	 * 
	 * @see edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantom3DVolumeRenderer
	 */
	public static final String XCAT_AUTO_RESIZE = "XCAT_AUTO_RESIZE";
	
	/**
	 * This flag is used in XCatScene.
	 * This flag allows to select the four different ventricle shapes in XCat.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a list of integer values. "1" selects the first shape. "1, 2, 3, 4" selects all available shapes.
	 * Default is "1, 2, 3, 4". 
	 * 
	 * @see edu.stanford.rsl.conrad.phantom.xcat.XCatScene#getSplineNameMaterialNameLUT()
	 */
	public static final String XCAT_VENTRICLE_SELECTION = "XCAT_VENTRICLE_SELECTION";
	
	/**
	 * Subsampling factor for spline sampling. This factor defaults to 4. The higher, the factor is set, the less points will be generated during tessellation. <BR>
	 * The <b>value</b> is a <b>String</b> representation of an integer number. An example would be "4". Valid numbers range from 1 to Integer.MAX_VALUE. 
	 */
	public static final String SPLINE_SUBSAMPLING_FACTOR = "SPLINE_SUBSAMPLING_FACTOR";
	
	/**
	 * Entry to select a preprocessed 4D time spline.<BR>
	 * The <b>value</b> is a <b>String</b> representation of a file location. The file must contain the 4D spline. 
	 */
	public static final String SPLINE_4D_LOCATION = "SPLINE_4D_LOCATION";
	
	
	/**
	 * Entry to locate the control files of a certain CT scanner. Use is dependent on the manufacturer of the CT scanner.<br>
	 * The <b>value</b> is a <b>String</b> indicating the location of the control directory.
	 */
	public static final String PATH_TO_CONTROL = "PATH_TO_CONTROL";
	
	/**
	 * Entry to locate the calibration files of a certain CT scanner. Use is dependent on the manufacturer of the CT scanner.<br>
	 * The <b>value</b> is a <b>String</b> indicating the location of the calibration directory.
	 */
	public static final String PATH_TO_CALIBRATION = "PATH_TO_CALIBRATION";
	
	/**
	 * Entry to store the relative heart phase for each projection.<br>
	 * The <b>value</b> is a <b>String</b> which describes the sequence of heart phases as numbers between 0 and 1 delimited by space.
	 */
	public static final String HEART_PHASES = "HEART_PHASES";
	
	/**
	 * Entry to store the relative breathing phase for each projection.<br>
	 * The <b>value</b> is a <b>String</b> which describes the sequence of breathing phases as numbers between 0 and 1 delimited by space.
	 */
	public static final String BREATHING_PHASES = "BREATHING_PHASES";
	
	/**
	 * Entry to store the path to the sound file that is played when reconstruction finishes.<br>
	 * The <b>value</b> is a <b>String</b> indicating the path and filename of a supported sound file (WAV, AIFF, AU).
	 */
	public static final String SOUND_FILE = "SOUND_FILE";

	/**
	 * Entry to the file "yacas.jar" which contains the LISP standard.
	 * The <b>value</b> is a <b>String</b> indicating the path and filename of yacas.jar.
	 */
	public static final String YACAS_LOCATION = "YACAS_LOCATION";

	/**
	 * Entry to describe the slow down value, if the memory gets too full during the processing.
	 * The higher the value, the more the system will wait, until the next projection is read.
	 */
	public static final String SLOW_DOWN_MS = "SLOW_DOWN_MS";
	
	/**
	 * Entry to the file which contains the initial beads position in projection [u, v].<br>
	 * This is for the weight-bearing project.<br> 
	 * The <b>value</b> is a <b>String</b> indicating the path and filename of the text file.
	 */
	public static final String INITIAL_BEADS_LOCATION_FILE = "INITIAL_BEADS_LOCATION_FILE";
	
	/**
	 * Entry to the preferred OpenCL device to be used by the application.<br> 
	 * The <b>value</b> is a <b>String</b> indicating the device identifier as reported by the driver.
	 * @see edu.stanford.rsl.conrad.opencl.OpenCLUtil#createContext
	 */
	public static final String OPENCL_DEVICE_SELECTION = "OPENCL_DEVICE_SELECTION";
	
	/**
	 * Entry to set the location of a data file that can be used with Weka. When a classifier is trained, the data is written to this location.<br> 
	 * The <b>value</b> is a <b>String</b> indicating a file location, e.g. "d:\data\data.arff".
	 * 
	 */
	public static final String EVALUATION_OUTPUT_LOCATION = "EVALUATION_OUTPUT_LOCATION";
	/**
	 * Entry to set the location of a data file that can be used with Weka. When a classifier is trained, the data is written to this location.<br> 
	 * The <b>value</b> is a <b>String</b> indicating a file location, e.g. "d:\data\data.arff".
	 * 
	 */
	public static final String CLASSIFIER_DATA_LOCATION = "CLASSIFIER_DATA_LOCATION";
	
	/**
	 * Entry to set the location of the classifier model. When a classifier is trained, the model is written to this location. When a classifier is used the model will be read from this location.<br> 
	 * The <b>value</b> is a <b>String</b> indicating a file location, e.g. "d:\data\classifier.model".
	 * 
	 */
	public static final String CLASSIFIER_MODEL_LOCATION = "CLASSIFIER_MODEL_LOCATION";
	
	/**
	 * Entry to set the location for marginal space learning data. 
	 * The <b>value</b> is a <b>String</b> indicating a file location, e.g. "d:\data\classifier.model".
	 * 
	 */
	public static final String MSL_DATA_LOCATION = "MSL_DATA_LOCATION";
	
	/**
	 * Entry to define a small accuracy tolerance when deciding whether the point of intersection between ray and triangle lies within the triangle.
	 * The <b>value</b> is a <b>Double</b> defining the tolerance.
	 * 
	 * @see edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle
	 */
	public static final String PHANTOM_PROJECTOR_RAYTRACING_EPSILON = "PHANTOM_PROJECTOR_RAYTRACING_EPSILON";
	
	
	/**
	 * For projecting triangle meshes, the watertight phantom projector is used by default. If this flag is set (true), the more established priority raytracer will be used.
	 * The <b>value</b> is a <b>Boolean</b> indicating whether the priority or the watertight ray tracer should be used for generating projections of triangle meshes.
	 */
	public static final String PHANTOM_PROJECTOR_ENFORCE_PRIORITY_RAYTRACER = "PHANTOM_PROJECTOR_ENFORCE_PRIORITY_RAYTRACER";
		
	
	/**
	 * Global configuration of the ED Phantom (CRIS M062 Phantom).
	 * Entry to define a special buffer diameter for the central element.
	 * The <b>value</b> is a <b>Double</b> defining the diameter in [mm].
	 */
	public static final String ED_PHANTOM_CENTERAL_BUFFER_DIAMETER = "ED_PHANTOM_CENTERAL_BUFFER_DIAMETER";

	/**
	 * Global configuration of the ED Phantom (CRIS M062 Phantom).
	 * Entry to define a special buffer diameter for the Insert 1.
	 * The <b>value</b> is a <b>Double</b> defining the diameter in [mm].
	 */
	public static final String ED_PHANTOM_INSERT_1_BUFFER_DIAMETER = "ED_PHANTOM_INSERT_1_BUFFER_DIAMETER";
	
	/**
	 * Global configuration of the ED Phantom (CRIS M062 Phantom).
	 * To configure presence of the bone ring around the central disk
	 * The <b>value</b> is a <b>boolean</b> which is true to activate the bone ring.
	 */
	public static final String ED_PHANTOM_BONE_RING = "ED_PHANTOM_BONE_RING";
	
	

	public static final HashMap<String,String> defaultValues;

	/**
	 * Entry to set the location of the grid feature extractor. The object is written to this location. When a new evaluation task is configured the object is read from this location and used.<br> 
	 * The <b>value</b> is a <b>String</b> indicating a file location, e.g. "d:\data\configparam.gfex".
	 * 
	 */
	public static final String GRID_FEATURE_EXTRACTOR_LOCATION = "GRID_FEATURE_EXTRACTOR_LOCATION";
	
	public static final String DARK_FIELD_RECONSTRUCTION_LOCATION_ANISO = "DARK_FIELD_RECONSTRUCTION_LOCATION_ANISO";
	public static final String DARK_FIELD_RECONSTRUCTION_LOCATION_ISO = "DARK_FIELD_RECONSTRUCTION_LOCATION_ISO";
	
	/**
	 * Entry to set the default location of the position of the CONRAD main window. 
	 * The key is a string that specifies the top left corner in MATLAB notation. "[0,0]" specifies the top left corner 
	 * to be identical with the top left corner of the screen. This is the default setting.
	 */
	public static final String CONRAD_WINDOW_DEFAULT_LOCATION = "CONRAD_WINDOW_DEFAULT_LOCATION";
	
	static {
		defaultValues = new HashMap<String, String>();
		defaultValues.put(SLOW_DOWN_MS, "10");
		defaultValues.put(SPLINE_SUBSAMPLING_FACTOR, "4");
		defaultValues.put(PHANTOM_PROJECTOR_ENFORCE_PRIORITY_RAYTRACER, "false");	
		defaultValues.put(ED_PHANTOM_BONE_RING, "false");
		defaultValues.put(ED_PHANTOM_CENTERAL_BUFFER_DIAMETER, "15");
		defaultValues.put(CONRAD_WINDOW_DEFAULT_LOCATION, "[0,0]");
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/