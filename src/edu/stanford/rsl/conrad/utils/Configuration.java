package edu.stanford.rsl.conrad.utils;

import java.beans.XMLDecoder;
import java.beans.XMLEncoder;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Locale;
import java.util.NoSuchElementException;
import java.util.Set;
import java.beans.ExceptionListener;
import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.filtering.CosineWeightingTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.redundancy.ParkerWeightingTool;
import edu.stanford.rsl.conrad.fitting.Function;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.ConfigFileBasedTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.ExtrapolatedTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.MultiSweepTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.SafeSerializable;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.detector.SimpleMonochromaticDetector;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import ij.ImageJ;


/**
 * Configuration is used to import Conrad configurations more easily and to store them globally.
 * Configuration objects can be used to configure filters and tools faster and more easily as
 * many of them require similar parameters regarding geometry, etc.
 * 
 * @author Andreas Maier
 *
 */

public class Configuration implements SafeSerializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3887275696746174804L;

	private Trajectory geometry = null;

	// Entries for reconstruction:
	private String volumeOfInterestFileName = null;
	private String projectionTableFileName = null;

	// Entries for filtering
	private double cutOffFrequency = 1;
	private String AutomaticExposureControlConfigFile = null;
	private String currentRowWeights = null;

	// Entries for C-arm trajectory:
	private String deviceSerialNumber = null;
	private double [] realtime = null;
	private double [] electroCardioGramm = null;

	// Entries for Automatic Exposure Control
	private double [] voltage = null;
	private double [] current = null;
	private double [] time = null;

	private double dCU = 0;
	private double dose = 0;
	private int intensifierSize = 0;

	// Entries for Utilities
	private String currentPath = null;
	private String recentFileOne = null;
	private String recentFileTwo = null;

	// Entries for Pipeline

	private ImageFilteringTool [] filterPipeline;
	private BufferedProjectionSink sink;

	// Entries for GUI

	public static final int MEDLINE_CITATION_FORMAT = 1;
	public static final int BIBTEX_CITATION_FORMAT = 2;

	private int citationFormat = BIBTEX_CITATION_FORMAT;
	private boolean importFromDicomAutomatically = false;
	private boolean useExtrapolatedGeometry = false;
	private boolean useHounsfieldScaling = false;

	private BilinearInterpolatingDoubleArray beamHardeningLookupTable = null;

	private static Configuration globalConfiguration = null;
	private Function hounsfieldScaling = null;

	private double [] heartPhases = null;
	private int numSweeps = 1;

	private HashMap<String,String> registry;

	// Entries for beads location in 2D or 3D, used for weight-bearing scanning	
	private double [][][] beadPosition2D = null; // [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough searching]]
	private double [][] beadMeanPosition3D = null; // [projection #][bead #][u, v]
	private boolean [] fAccessed = new boolean [500]; // if the projection is accessed
	

	private double[] respiratoryMotionField = null;
	private double[] diaphragmPositionField = null;
	private double[][] diaphragmModelField = null;
	

	private double maxMotion = 0;
	private float lowHyst, highHyst;
	private int seedX, seedY;
	private int roiWidthHalf, roiHeightTop, roiHeightBottom;
	
	private int maxIter;
	private double projTurningAngle;
	private double correctionFactor;
	private double[][] diaphragmCoords = null;
	
	public double[][] getDiaphragmModelField() {
		return diaphragmModelField;
	}

	public void setDiaphragmModelField(double[][] diaphragmModelField) {
		this.diaphragmModelField = diaphragmModelField;
	}
	
	public void setDiaphragmModelEntry(int i, double[] entry) {
		this.diaphragmModelField[i] = entry;
	}
	
	public double[] getDiaphragmModelEntry(int i ) {
		return this.diaphragmModelField[i];
	}
	
	public void setProjTurningAngle(double p){
		this.projTurningAngle = p;
	}
	public double getProjTurningAngle(){
		return projTurningAngle;
	}
	public double getCorrectionFactor() {
		return correctionFactor;
	}
	public void setCorrectionFactor(double correctionFactor) {
		this.correctionFactor = correctionFactor;
	}
	public int getMaxIter() {
		return maxIter;
	}
	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}
	public float getLowHyst() {
		return lowHyst;
	}
	public void setLowHyst(float lowHyst) {
		this.lowHyst = lowHyst;
	}
	public float getHighHyst() {
		return highHyst;
	}
	public void setHighHyst(float highHyst) {
		this.highHyst = highHyst;
	}
	public int getSeedX() {
		return seedX;
	}
	public void setSeedX(int seedX) {
		this.seedX = seedX;
	}
	public int getSeedY() {
		return seedY;
	}
	public void setSeedY(int seedY) {
		this.seedY = seedY;
	}
	public int getRoiWidthHalf() {
		return roiWidthHalf;
	}
	public void setRoiWidthHalf(int roiWidthHalf) {
		this.roiWidthHalf = roiWidthHalf;
	}
	public int getRoiHeightTop() {
		return roiHeightTop;
	}
	public void setRoiHeightTop(int roiHeightTop) {
		this.roiHeightTop = roiHeightTop;
	}
	public int getRoiHeightBottom() {
		return roiHeightBottom;
	}
	public void setRoiHeightBottom(int roiHeightBottom) {
		this.roiHeightBottom = roiHeightBottom;
	}
	public void setRespiratoryMotionField(double[] field) {
		respiratoryMotionField = field;
	}
	public void setRespiratoryMotionFieldEntry(int entry, double value) {
		respiratoryMotionField[entry] = value;
	}
	public double getRespiratoryMotionFieldEntry(int entry) {
		return respiratoryMotionField[entry];
	}
	public double[] getRespiratoryMotionField() {
		return respiratoryMotionField;
	}
	
	public void setDiaphragmPositionField(double[] field) {
		diaphragmPositionField = field;
	}
	public double[] getDiaphragmPositionField() {
		return diaphragmPositionField;
	}
	public double getDiaphragmPositionFieldEntry(int i) {
		return diaphragmPositionField[i];
	}
	
	public void setDiaphragmCoords(double[][] field) {
		this.diaphragmCoords = field;
	}
	public double[][] getDiaphragmCoords() {
		return this.diaphragmCoords;
	}
	public double[] getDiaphragmCoordsEntry(int entry) {
		return this.diaphragmCoords[entry];
	}
	public void setDiaphragmCoordsEntry(int entry, double[] values) {
		this.diaphragmCoords[entry] = values;
	}
	public void setMaxMotion(double m){
		this.maxMotion=m;
	}
	public double getMaxMotion(){
		return maxMotion;
	}
	
	/**
	 * @return the registry
	 */
	public HashMap<String, String> getRegistry() {
		return registry;
	}

	/**
	 * @param registry the registry to set
	 */
	public void setRegistry(HashMap<String, String> registry) {
		this.registry = registry;
	}

	public static void saveConfiguration(){
		String filename = System.getProperty("user.home") + "/Conrad.xml";
		saveConfiguration(getGlobalConfiguration(), filename);
		setGlobalConfiguration(null);
		loadConfiguration();
	}

	public static void saveConfiguration(Configuration config, String filename){
		try {
			Thread.currentThread().setContextClassLoader(Configuration.class.getClassLoader());
			XMLEncoder oos = new XMLEncoder (new FileOutputStream(filename));
			config.prepareForSerialization();
			oos.writeObject(config);
			oos.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} 
	}

	public static void loadConfiguration(){
		if (getGlobalConfiguration() == null){
			String filename = System.getProperty("user.home") + "/Conrad.xml";
			Configuration config = loadConfiguration(filename);
			if (config !=null) {
				setGlobalConfiguration(config);
			}
		}
	}

	public static Configuration loadConfiguration(String filename){
		try {
			Thread.currentThread().setContextClassLoader(Configuration.class.getClassLoader());
			ExceptionListener el = new ExceptionListener() {
				public void exceptionThrown(Exception e) {
					e.printStackTrace();
				};
			};
			XMLDecoder ois = new XMLDecoder (new FileInputStream(filename), null, el);		
			Configuration config = (Configuration) ois.readObject();
			ois.close();
			return config;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("No Conrad.xml found. Creating default config.");
			int revan = JOptionPane.showConfirmDialog(null, "No config file found. Create default config?");
			if (revan == JOptionPane.YES_OPTION) initConfig();
		} catch (ClassCastException e) {
			e.printStackTrace();
			System.out.println("Previous Conrad.xml invalid rewrite forced. Creating default config.");
			int revan = JOptionPane.showConfirmDialog(null, "Could not read config file. Overwrite with default config?");
			if (revan == JOptionPane.YES_OPTION) initConfig();
		} catch (NoSuchElementException e) {
			e.printStackTrace();
			System.out.println("Previous Conrad.xml invalid rewrite forced. Creating default config.");
			int revan = JOptionPane.showConfirmDialog(null, "Could not read config file. Overwrite with default config?");
			if (revan == JOptionPane.YES_OPTION) initConfig();
		}
		return null;
	}

	public static Configuration getGlobalConfiguration(){
		return globalConfiguration;
	}

	public static void setGlobalConfiguration(Configuration config){
		globalConfiguration = config;
	}

	public String getVolumeOfInterestFileName() {
		return volumeOfInterestFileName;
	}
	public void setVolumeOfInterestFileName(String volumeOfInterestFileName) {
		this.volumeOfInterestFileName = volumeOfInterestFileName;
	}
	public String getProjectionTableFileName() {
		return projectionTableFileName;
	}
	public void setProjectionTableFileName(String projectionTableFileName) {
		this.projectionTableFileName = projectionTableFileName;
	}

	public void setCutOffFrequency(double cutOffFrequency) {
		this.cutOffFrequency = cutOffFrequency;
	}
	public double getCutOffFrequency() {
		return cutOffFrequency;
	}

	public void setAutomaticExposureControlConfigFile(
			String automaticExposureControlConfigFile) {
		AutomaticExposureControlConfigFile = automaticExposureControlConfigFile;
	}

	public String getAutomaticExposureControlConfigFile() {
		return AutomaticExposureControlConfigFile;
	}

	public void setCurrentRowWeights(String currentRowWeights) {
		this.currentRowWeights = currentRowWeights;
	}

	public String getCurrentRowWeights() {
		return currentRowWeights;
	}

	public void setDeviceSerialNumber(String deviceSerialNumber) {
		this.deviceSerialNumber = deviceSerialNumber;
	}

	public String getDeviceSerialNumber() {
		return deviceSerialNumber;
	}

	/**
	 * returns the voltage in [kV]
	 * @return the voltage
	 */
	public double[] getVoltage() {
		return voltage;
	}

	public void setVoltage(double[] voltage) {
		this.voltage = voltage;
	}

	/**
	 * Returns the current in [mA]
	 * @return the current
	 */
	public double[] getCurrent() {
		return current;
	}

	public void setCurrent(double[] current) {
		this.current = current;
	}

	/**
	 * Returns exposure time in [s]
	 * @return the exposure time
	 */
	public double[] getTime() {
		return time;
	}

	/**
	 * Sets the exposure time in [s]
	 * @param time
	 */
	public void setTime(double[] time) {
		this.time = time;
	}

	public void setRealtime(double [] realtime) {
		this.realtime = realtime;
	}

	public double [] getRealtime() {
		return realtime;
	}

	public void setElectroCardioGramm(double [] electroCardioGramm) {
		this.electroCardioGramm = electroCardioGramm;
	}

	public double [] getElectroCardioGramm() {
		return electroCardioGramm;
	}

	public double getdCU() {
		return dCU;
	}

	public void setdCU(double dCU) {
		this.dCU = dCU;
	}

	public double getDose() {
		return dose;
	}

	public void setDose(double dose) {
		this.dose = dose;
	}

	public void setIntensifierSize(int intensifierSize) {
		this.intensifierSize = intensifierSize;
	}

	public int getIntensifierSize() {
		return intensifierSize;
	}

	public void setCurrentPath(String currentPath) {
		this.currentPath = currentPath;
	}

	public String getCurrentPath() {
		return currentPath;
	}

	public void setRecentFileOne(String recentFileOne) {
		this.recentFileOne = recentFileOne;
	}

	public String getRecentFileOne() {
		return recentFileOne;
	}

	public void setRecentFileTwo(String recentFileTwo) {
		this.recentFileTwo = recentFileTwo;
	}

	public String getRecentFileTwo() {
		return recentFileTwo;
	}

	public void setFilterPipeline(ImageFilteringTool [] pipline) {
		this.filterPipeline = pipline;
	}

	public ImageFilteringTool [] getFilterPipeline() {
		return filterPipeline;
	}

	public void setSink(BufferedProjectionSink sink) {
		this.sink = sink;
	}

	public BufferedProjectionSink getSink() {
		return sink;
	}
	
	public void setBeadPosition2D(double [][][] beadPosition2D) {
		this.beadPosition2D = beadPosition2D;
	}
	
	public void setBeadPosition2D(int i, int j, int k, double value) {
		this.beadPosition2D[i][j][k] = value;
	}
	
	public double [][][] getBeadPosition2D() {
		return beadPosition2D;
	}
	
	public void setBeadMeanPosition3D(double [][] beadMeanPosition3D) {
		this.beadMeanPosition3D = beadMeanPosition3D;
	}
	
	public double [][] getBeadMeanPosition3D() {
		return beadMeanPosition3D;
	}
	
	public boolean [] getfAccessed() {
		return fAccessed;
	}
	
	public static void initConfig(){
		new ImageJ();
		Configuration config = new Configuration();
		config.geometry = new Trajectory();
		config.detector = new SimpleMonochromaticDetector();
		config.geometry.setDetectorUDirection(CameraAxisDirection.DETECTORMOTION_PLUS);
		config.geometry.setDetectorVDirection(CameraAxisDirection.ROTATIONAXIS_PLUS);
		config.geometry.setDetectorHeight(480);
		config.geometry.setDetectorWidth(620);
		config.geometry.setSourceToAxisDistance(600.0);
		config.geometry.setSourceToDetectorDistance(1200.0);
		config.geometry.setReconDimensions(256, 256, 256);
		config.geometry.setPixelDimensionX(1);
		config.geometry.setPixelDimensionY(1);
		config.geometry.setVoxelSpacingX(1.0);
		config.geometry.setVoxelSpacingY(1.0);
		config.geometry.setVoxelSpacingZ(1.0);
		config.geometry.setAverageAngularIncrement(1.0);
		config.geometry.setProjectionStackSize(200);
		config.registry = new HashMap<String, String>();
		config.registry.put(RegKeys.PATH_TO_CALIBRATION, "C:\\calibration");
		config.registry.put(RegKeys.PATH_TO_CONTROL, "C:\\control");
		config.registry.put(RegKeys.MAX_THREADS, "8");
		config.registry.put(RegKeys.XCAT_PATH, "E:\\phantom data\\numeric phantoms\\xcat\\NCAT2.0_PC");
		ImageFilteringTool[] standardPipeline = new ImageFilteringTool[] {
				new CosineWeightingTool(),
				new ParkerWeightingTool(),
				new RampFilteringTool(),
				new VOIBasedReconstructionFilter()
		};
		config.setFilterPipeline(standardPipeline);
//		config.setFilterPipeline(ImageFilteringTool.getFilterTools());
		config.setCitationFormat(MEDLINE_CITATION_FORMAT);
		config.setSink(new ImagePlusDataSink());
		
		int numProjectionMatrices = config.getGeometry().getProjectionStackSize();
		double sourceToAxisDistance = config.getGeometry().getSourceToAxisDistance();
		double averageAngularIncrement = config.getGeometry().getAverageAngularIncrement();
		double detectorOffsetU = config.getGeometry().getDetectorOffsetU();
		double detectorOffsetV = config.getGeometry().getDetectorOffsetV();
		CameraAxisDirection uDirection = config.getGeometry().getDetectorUDirection();
		CameraAxisDirection vDirection = config.getGeometry().getDetectorVDirection();
		SimpleVector rotationAxis = new SimpleVector(0, 0, 1);
		Trajectory geom = new CircularTrajectory(config.getGeometry());
		geom.setSecondaryAngleArray(null);
		((CircularTrajectory)geom).setTrajectory(numProjectionMatrices, sourceToAxisDistance, averageAngularIncrement, detectorOffsetU, detectorOffsetV, uDirection, vDirection, rotationAxis);
		if (geom != null){
			config.setGeometry(geom);
		}
		setGlobalConfiguration(config);
		saveConfiguration();
	}

	public static Trajectory loadGeometrySource(Configuration config) throws Exception{
		Trajectory loaded = null;
		if (config.getProjectionTableFileName() != null){
			loaded = ConfigFileBasedTrajectory.openAsGeometrySource(config.getProjectionTableFileName(), config.getGeometry());
		} else {
			if (config.getGeometry().getAverageAngularIncrement() != -1){
				// Binning is determined by the detector size in pixels
				String binning = "4x4";
				if (config.getGeometry().getDetectorWidth() == 1240) {
					binning = "2x2";
				}
				if (config.getGeometry().getDetectorWidth() == 960) {
					binning = "2x2";
				}
				
				String filename = "";
				if (config.getDeviceSerialNumber().equals("55242"))	// IR1
					filename = config.registry.get(RegKeys.PATH_TO_CALIBRATION) + "/" + config.getDeviceSerialNumber() + "/active/g_Left_"+ config.getIntensifierSize() + "_" + getStandardNumberFormat().format(config.geometry.getAverageAngularIncrement()) + "0_" + binning +"/projtable.txt";
				else 				
					filename = config.registry.get(RegKeys.PATH_TO_CALIBRATION) + "/" + config.getDeviceSerialNumber() + "/active/-903" + "/g/" + config.getIntensifierSize() + "/" + getStandardNumberFormat().format(config.geometry.getAverageAngularIncrement()) + "/projtable.txt";
				loaded = ConfigFileBasedTrajectory.openAsGeometrySource(filename, config.getGeometry());
			} 
		}
		if ((config.geometry != null)&&(config.getUseExtrapolatedGeometry())) {
			ExtrapolatedTrajectory extra = new ExtrapolatedTrajectory(loaded);
			extra.extrapolateProjectionGeometry();
			loaded = (Trajectory) extra;
		}
		if ((config.geometry != null)&&(config.getNumSweeps() > 1)) {
			MultiSweepTrajectory extra = new MultiSweepTrajectory(loaded);
			extra.extrapolateProjectionGeometry();
			loaded = (Trajectory) extra;
		}
		if (loaded != null){
			System.out.println("Found projection geometry with " + loaded.getNumProjectionMatrices() + " projections.");
		} else {
			System.out.println("No geometry loaded.");
		}
		return loaded;
	}

	public static NumberFormat getStandardNumberFormat(){
		NumberFormat nf = NumberFormat.getInstance(Locale.US);
		nf.setMaximumFractionDigits(1);
		nf.setMinimumFractionDigits(1);
		nf.setMaximumIntegerDigits(1);
		nf.setMinimumIntegerDigits(1);
		return nf;
	}

	public void setCitationFormat(int citationFormat) {
		this.citationFormat = citationFormat;
	}

	public int getCitationFormat() {
		return citationFormat;
	}

	public void setImportFromDicomAutomatically(boolean importFromDicomAutomatically) {
		this.importFromDicomAutomatically = importFromDicomAutomatically;
	}

	public boolean getImportFromDicomAutomatically() {
		return importFromDicomAutomatically;
	}


	public void setUseExtrapolatedGeometry(boolean useExtrapolatedGeometry) {
		this.useExtrapolatedGeometry = useExtrapolatedGeometry;
	}


	public boolean getUseExtrapolatedGeometry() {
		return useExtrapolatedGeometry;
	}


	public void setUseHounsfieldScaling(boolean useHounsfieldScaling) {
		this.useHounsfieldScaling = useHounsfieldScaling;
	}


	public boolean getUseHounsfieldScaling() {
		return useHounsfieldScaling;
	}


	public void setHounsfieldScaling(Function hounsfieldScaling) {
		this.hounsfieldScaling = hounsfieldScaling;
	}

	public Function getHounsfieldScaling(){
		return hounsfieldScaling;
	}

	public void setHeartPhases(double [] heartPhases) {
		this.heartPhases = heartPhases;
	}

	public double [] getHeartPhases() {
		return heartPhases;
	}

	public void setNumSweeps(int numSweeps) {
		this.numSweeps = numSweeps;
	}

	public int getNumSweeps() {
		return numSweeps;
	}

	public Trajectory getGeometry() {
		if (geometry == null){
			geometry = new Trajectory();
		}
		return geometry;
	}

	public void setGeometry(Trajectory geometry) {
		this.geometry = geometry;
	}

	/**
	 * @param beamHardeningLookupTable the beamHardeningLookupTable to set
	 */
	public void setBeamHardeningLookupTable(BilinearInterpolatingDoubleArray beamHardeningLookupTable) {
		this.beamHardeningLookupTable = beamHardeningLookupTable;
	}

	/**
	 * @return the beamHardeningLookupTable
	 */
	public BilinearInterpolatingDoubleArray getBeamHardeningLookupTable() {
		return beamHardeningLookupTable;
	}

	public void setRegistryEntry(String key, String value){
		registry.put(key, value);
	}

	/**
	 * Reads a key from the registry. If the key is not set, the default value will be returned.
	 * @param key
	 * @return the value in the registry
	 */
	public String getRegistryEntry(String key){
		String valueString = registry.get(key);
		if (valueString==null){
			valueString = RegKeys.defaultValues.get(key);
		}
		return valueString;
	}

	public Set<String> getRegistryKeys(){
		if (registry == null){
			resetRegistry();
		}
		return registry.keySet();
	}
	
	public void resetRegistry(){
		registry = new HashMap<String, String>();
	}

	public void prepareForSerialization() {
		if (getSink() != null) getSink().prepareForSerialization();
		if (getGeometry() != null) getGeometry().prepareForSerialization();
		ImageFilteringTool [] pipeline = getFilterPipeline();
		for (ImageFilteringTool tool : pipeline){
			tool.prepareForSerialization();
		}
	}

	/**
	 * Convenience method to access the registry:<BR>
	 * Reads a key form the registry and converts it to int.
	 * A RuntimeException will be thrown, if the number cannot be parsed.
	 * @param key
	 * @return the int value. 
	 */
	public int queryIntFromRegistry(String key) {
		String value = getRegistryEntry(key);
		return Integer.parseInt(value);
	}
	
		//Geometry of crosscalibration phantom
	private double[] originKinect;
	private double[] originKinect2;
	private SimpleMatrix rotationKinectToZeego; //rotation matrix, [0]-[2]=first column, etc.
	private SimpleMatrix rotationKinectToZeego2; //rotation matrix, [0]-[2]=first column, etc.
	private double[] translationKinectToZeego;
	private SimpleVector[][] clickedCentersXYZ = new SimpleVector[3][6];
	private SimpleVector[][] clickedCentersXYZ2 = new SimpleVector[3][6];
	private String pathDepthImageCalibration;
	private String pathDepthImageCalibration2;
	private String pathDepthImagePhantom;
	private String pathDepthImagePhantom2;
	private XRayDetector detector;
	
	public double[] getOriginKinect() {
		return originKinect;
	}
	
	public void setOriginKinect(double[] originKinect) {
		this.originKinect = originKinect;
	}
	
	public SimpleMatrix getRotationKinectToZeego() {
		return rotationKinectToZeego;
	}
	
	public void setRotationKinectToZeego(SimpleMatrix rotationKinectToZeego) {
		this.rotationKinectToZeego = rotationKinectToZeego;
	}
	
	public double[] getTranslationKinectToZeego() {
		return translationKinectToZeego;
	}
	
	public void setTranslationKinectToZeego(double[] translationKinectToZeego) {
		this.translationKinectToZeego = translationKinectToZeego;
	}
	
	public String getPathDepthImageCalibration() {
		return pathDepthImageCalibration;
	}
	
	public void setPathDepthImageCalibration(String pathDepthImage) {
		this.pathDepthImageCalibration = pathDepthImage;
	}
	public String getPathDepthImagePhantom() {
		return pathDepthImagePhantom;
	}
	public void setPathDepthImagePhantom(String pathDepthImagePhantom) {
		this.pathDepthImagePhantom = pathDepthImagePhantom;
	}
	public SimpleVector[][] getClickedCentersXYZ() {
		return clickedCentersXYZ;
	}
	public void setClickedCentersXYZ(SimpleVector[][] clickedCentersXYZ) {
		this.clickedCentersXYZ = clickedCentersXYZ;
	}
	public SimpleMatrix getRotationKinectToZeego2() {
		return rotationKinectToZeego2;
	}
	public void setRotationKinectToZeego2(SimpleMatrix rotationKinectToZeego2) {
		this.rotationKinectToZeego2 = rotationKinectToZeego2;
	}
	public SimpleVector[][] getClickedCentersXYZ2() {
		return clickedCentersXYZ2;
	}
	public void setClickedCentersXYZ2(SimpleVector[][] clickedCentersXYZ2) {
		this.clickedCentersXYZ2 = clickedCentersXYZ2;
	}
	public String getPathDepthImageCalibration2() {
		return pathDepthImageCalibration2;
	}
	public void setPathDepthImageCalibration2(String pathDepthImageCalibration2) {
		this.pathDepthImageCalibration2 = pathDepthImageCalibration2;
	}
	public String getPathDepthImagePhantom2() {
		return pathDepthImagePhantom2;
	}
	public void setPathDepthImagePhantom2(String pathDepthImagePhantom2) {
		this.pathDepthImagePhantom2 = pathDepthImagePhantom2;
	}
	public double[] getOriginKinect2() {
		return originKinect2;
	}
	public void setOriginKinect2(double[] originKinect2) {
		this.originKinect2 = originKinect2;
	}

	public XRayDetector getDetector() {
		return detector;
	}
	
	public void setDetector(XRayDetector detector){
		this.detector = detector;	
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/