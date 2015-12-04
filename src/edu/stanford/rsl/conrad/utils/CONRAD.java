/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.utils;

import java.awt.Point;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.apps.gui.RawDataOpener;

public abstract class CONRAD {
	public static final String VersionString = "Version 1.0.6";
	public static final String CONRADBibtex = "@article{Maier13-CSF," +
	"  author = {A. Maier, H. G. Hofmann, M. Berger, P. Fischer, C. Schwemmer, H. Wu, K. Müller, J. Hornegger, J. H. Choi, C. Riess, A. Keil, and R. Fahrig},\n" +
	"  title={{CONRAD - A software framework for cone-beam imaging in radiology}},\n" +
	"  journal={Medical Physics},\n" +
	"  volume={40},\n" +
	"  number={11},\n" +
	"  pages={111914-1-8},\n" +
	"  year={2013}\n" +
	"}";
	public static final String CONRADMedline = "A. Maier, H. G. Hofmann, M. Berger, P. Fischer, C. Schwemmer, H. Wu, K. Müller, J. Hornegger, J. H. Choi, C. Riess, A. Keil, and R. Fahrig. CONRAD—A software framework for cone-beam imaging in radiology. Medical Physics 40(11):111914-1-8. 2013";
	public static final String CONRADDefinition = "CONe-beam framework for RADiology (CONRAD)";
	public static final double SMALL_VALUE = 1.0e-12;
	public static final double BIG_VALUE = 1.0e12;
	public static final double DOUBLE_EPSILON;
	public static final double FLOAT_EPSILON;
	public static final int INVERSE_SPEEDUP = 0;
	public static boolean useGarbageCollection;
	public static final String EOL = System.getProperty("line.separator");
	public static final long INPUT_QUEUE_DELAY = 0;
	/**
	 * This flag can be used to control debug outputs.
	 * 0: No Debug output
	 * 1: Some Debug output
	 * 2: More debug output
	 * 3: All debug output
	 */
	public static int DEBUGLEVEL = 0;

	static {
		double doubleEps = 1.0d;
		do
			doubleEps /= 2.0d;
		while ((1.0d + (doubleEps/2.0)) != 1.0);
		DOUBLE_EPSILON = doubleEps;
		//		System.out.println( "Calculated double epsilon to " + DOUBLE_EPSILON + ".");

		float floatEps = 1.0f;
		do
			floatEps /= 2.0f;
		while ((1.0f + (floatEps/2.0f)) != 1.0f);
		FLOAT_EPSILON = floatEps;
		//		System.out.println( "Calculated float epsilon to " + FLOAT_EPSILON + ".");
	}

	public static void setup(){
		new ImageJ();
		Configuration.loadConfiguration();	
		RawDataOpener opener = RawDataOpener.getRawDataOpener();
		opener.setVisible(true);
		opener.getjButtonLittle().doClick();
		opener.getjButtonFloat().doClick();
		opener.setLocation(0, 500);
	}
	
	public static Point getWindowTopCorner(){
		String string = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CONRAD_WINDOW_DEFAULT_LOCATION);
		string = string.replace("[","").replace("]", "").replace(" ", "").replace(";", "");
		String [] strings = string.split(",");
		Point location = new Point(Integer.parseInt(strings[0]), Integer.parseInt(strings[1]));
		return location;
	}

	public static void gc(){
		if (useGarbageCollection) { 
			Runtime.getRuntime().gc();
		}
	}

	public static boolean isUseGarbageCollection() {
		return useGarbageCollection;
	}

	public static void setUseGarbageCollection(boolean useGarbageCollection) {
		CONRAD.useGarbageCollection = useGarbageCollection;
	}

	/**
	 * Returns the number of processors which should be used by CONRAD. Equals  Runtime.getRuntime().availableProcessors() if "MAX_THREADS" is not set in CONRAD registry.
	 * Otherwise "MAX_THREADS" is the upper bound for availableProcessors().
	 * @return the maximal number of threads
	 */
	public static int getNumberOfThreads(){
		int numThreads = Runtime.getRuntime().availableProcessors();
		try {
			String regKey = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS);
			int threadLimit = numThreads;
			if (regKey != null) {
				threadLimit = Integer.parseInt(regKey);
			}
			if (threadLimit < numThreads) numThreads = threadLimit;
		} catch (Exception e){
			System.out.println("Key '"+RegKeys.MAX_THREADS+"' was not found in registry.");
		}
		return numThreads;
	}

	/**
	 * CONRAD checks the environment variables for a variable called "CONRAD_HOME". If this is set, the variable will be useed. Otherwise the user home will be set up as CONRAD home.
	 * @return the path to the CONRAD home.
	 */
	public static String getUserHome(){
		Map<String,String> env = System.getenv();
		if (env.containsKey("CONRAD_HOME")){
			return env.get("CONRAD_HOME");
		}
		return System.getProperty("user.home");
	}

	/**
	 * Returns the total free memory as double between 1.0 and 0.0.
	 * 0.0 means that there is no free memory available.
	 * 1.0 means that the memory is completely free.
	 * Note that this function also takes the memory into account that the VM will allocate, if more memory is required.
	 * @return the free memory
	 */
	public static double getFreeMemoryAsDouble(){
		long t = Runtime.getRuntime().totalMemory();
		long m = Runtime.getRuntime().maxMemory();
		long f = Runtime.getRuntime().freeMemory();
		return (m-t+f)/(double)m;
	}

	private static HashMap<Class<? extends Object>, ArrayList<Object>> classLookupTable = new HashMap<Class<? extends Object>, ArrayList<Object>>();
	
	/**
	 * The method will parse the whole current directories currently available to the ClassLoader to find all
	 * classes that are subclasses of cl. The method returns an ArrayList with all objects that could be instantiated using the default constructor.
	 * Introspection is quite useful...
	 * <br><br>
	 * Parsing of the class path will only happen once. Further class will be redirected to an internal HashMap.
	 * <br> We now return clones of the original list as the instances might be changes by the user and we
	 * want default instances here that are suited for user selection.
	 * @param cl the class to find the subclasses for
	 * @return the list of found instances.
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static ArrayList<Object> getInstancesFromConrad(Class<? extends Object> cl) throws ClassNotFoundException, IOException{
		ArrayList<Object> found = classLookupTable.get(cl);
		if (found == null){
			found = getInstancesFromConradFromClasspath(cl);
		}
		// Create a clone of the found array list as instance might be altered by the user...
		ArrayList<Object> cloneArrayList = new ArrayList<Object>();
		for (int i=0; i < found.size(); i++){
			try {
				// uncomment this line to figure out which instantiation takes so long... 
				//long time = System.currentTimeMillis();
				cloneArrayList.add(found.get(i).getClass().newInstance());
				//System.out.println("Copied " + found.get(i) + " in " + (-time+System.currentTimeMillis()) + " ms");
			} catch (InstantiationException e) {
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				e.printStackTrace();
			}
		}
		return cloneArrayList;
	}
	
	/**
	 * Searches the class path for classes that can be cast onto the class cl.
	 * Method will register the found classes in an internal lookup table.
	 * 
	 * @param cl the class to search for
	 * @return the list of matching classes
	 * @throws ClassNotFoundException might happen
	 * @throws IOException may also occur.
	 */
	private synchronized static ArrayList<Object> getInstancesFromConradFromClasspath(Class<? extends Object> cl) throws ClassNotFoundException, IOException{
		ArrayList<Object> found = new ArrayList<Object>();
		ArrayList<Class<? extends Object>> classes = getClasses("edu.stanford.rsl");
		for (Class<? extends Object> match: classes){
			if (cl.isAssignableFrom(match)){
				try {
					found.add(match.newInstance());
				} catch (InstantiationException e) {
					/**
					 * skip over abstract classes. Only the ones which implement the default constructor will be
					 * returned.
					 */
				} catch (IllegalAccessException e) {
					e.printStackTrace();
				}

			}
		}
		classLookupTable.put(cl, found);
		return found;
	}

	/**
	 * Scans all classes accessible from the context class loader which belong to the given package and subpackages.
	 *
	 * @param packageName The base package
	 * @return The classes
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	private static ArrayList<Class<? extends Object>> getClasses(String packageName)
	throws ClassNotFoundException, IOException {
		ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
		assert classLoader != null;
		String path = packageName.replace('.', '/');
		Enumeration<URL> resources = classLoader.getResources(path);
		List<File> dirs = new ArrayList<File>();
		while (resources.hasMoreElements()) {
			URL resource = resources.nextElement();
			String filename = resource.getFile();
			if (filename.contains("%20")) filename = filename.replace("%20", " ");
			dirs.add(new File(filename));
		}
		ArrayList<Class<? extends Object>> classes = new ArrayList<Class<? extends Object>>();
		for (File directory : dirs) {
			classes.addAll(findClasses(directory, packageName));
		}
		return classes;
	}

	/**
	 * Recursive method used to find all classes in a given directory and subdirs.
	 *
	 * @param directory   The base directory
	 * @param packageName The package name for classes found inside the base directory
	 * @return The classes
	 * @throws ClassNotFoundException
	 */
	private static List<Class<? extends Object>> findClasses(File directory, String packageName) throws ClassNotFoundException {
		List<Class<? extends Object>> classes = new ArrayList<Class<? extends Object>>();
		if (!directory.exists()) {
			return classes;
		}
		File[] files = directory.listFiles();
		for (File file : files) {
			if (file.isDirectory()) {
				assert !file.getName().contains(".");
				classes.addAll(findClasses(file, packageName + "." + file.getName()));
			} else if (file.getName().endsWith(".class")) {
				try{
					classes.add(Class.forName(packageName + '.' + file.getName().substring(0, file.getName().length() - 6)));
				} catch (UnsatisfiedLinkError e){
					/**
					 * We skip over classes that cannot be loaded (compilation errors, etc.).
					 */
					//System.out.println("skipping " + file);
				} catch (NoClassDefFoundError e2){
					/**
					 * We skip over classes that cannot be loaded (compilation errors, etc.).
					 */
					//System.out.println("skipping " + file);
				}
			}
		}
		return classes;
	}

	/**
	 * Method to log something in a log file.
	 * Currently it just redirects to IJ.log method.
	 * @param string
	 */
	public static void log(String string) {
		IJ.log(string);
	}

}
