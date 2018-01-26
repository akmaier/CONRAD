/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public abstract class GridFeatureExtractor implements GUIConfigurable,
		Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7113126589428101083L;
	Grid2D dataGrid;
	Grid2D labelGrid;
	Instances instances = null;
	boolean configured = false;
	String className = "class";
	ArrayList<String> classValues;
	ArrayList<Attribute> attribs;
	FileOutputStream out;
	
	public void clearInstances(Attribute classAttribute){
		// leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);
	}

	public abstract double[] extractFeatureAtIndex(int x, int y);
	
	public double[] extractFeatureAtIndex(int c){
		int x = c % dataGrid.getWidth(); 
		int y = c / dataGrid.getWidth();
		return extractFeatureAtIndex(x, y);
	}
	
	public int numberOfZeroSamples() {
		int result = 0;
		for (int j = 0; j < labelGrid.getHeight(); j++) {
			for (int i = 0; i < labelGrid.getWidth(); i++) {
				if (labelGrid.getAtIndex(i, j) == 0)
					result++;
			}
		}
		return result;
	}
	public void prepareForSerialization() {
		dataGrid = null;
		labelGrid = null;
		if (instances != null)
			instances.delete();
	}

	/**
	 * Creates a GridFeatureExtractor. If the corresponding RegKey and the file
	 * pointed at exists, it is loaded from disk. Otherwise the user is queried.
	 * 
	 * @return the GridFeatureExtractor
	 */
	public static GridFeatureExtractor loadDefaultGridFeatureExtractor() {
		GridFeatureExtractor featureExtractor = null;
		String gfexfileloc = Configuration.getGlobalConfiguration()
				.getRegistryEntry(RegKeys.GRID_FEATURE_EXTRACTOR_LOCATION);
		if (gfexfileloc != null) {
			File gfexfile = new File(gfexfileloc);
			if (gfexfile.exists()) {
				ObjectInputStream oos;
				try {
					oos = new ObjectInputStream(new FileInputStream(gfexfile));
					featureExtractor = (GridFeatureExtractor) oos.readObject();
					oos.close();
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (ClassNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		if (featureExtractor == null) {
			try {
				featureExtractor = (GridFeatureExtractor) UserUtil.queryObject(
						"Select Feature Extractor: ", "Feature Selection",
						GridFeatureExtractor.class);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return featureExtractor;
	}

	/**
	 * Set the GridFeatureExtractor as the default GridFeatureExtractor, if the
	 * corresponding registry key is set.
	 * 
	 * @param featureExtractor
	 */
	public static void setDefaultGridFeatureExtractor(
			GridFeatureExtractor featureExtractor) {
		Grid2D labelGrid = featureExtractor.getLabelGrid();
		Grid2D dataGrid = featureExtractor.getDataGrid();
		featureExtractor.prepareForSerialization();
		String gfexfileloc = Configuration.getGlobalConfiguration()
				.getRegistryEntry(RegKeys.GRID_FEATURE_EXTRACTOR_LOCATION);
		if (gfexfileloc != null) {
			File gfexfile = new File(gfexfileloc);
			ObjectOutputStream oos;
			try {
				oos = new ObjectOutputStream(new FileOutputStream(gfexfile));
				oos.writeObject(featureExtractor);
				oos.flush();
				oos.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		featureExtractor.setLabelGrid(labelGrid);
		featureExtractor.setDataGrid(dataGrid);
	}

	public Instances getInstances() {
		return instances;
	}

	public abstract String getName();

	public String toString() {
		if (configured) {
			return getName() + " (" + (attribs.size() - 1) + " features)";
		} else {
			return getName();
		}
	}

	public void saveInstances(String s) throws IOException {
		if (Configuration.getGlobalConfiguration().getRegistryEntry(
				RegKeys.CLASSIFIER_DATA_LOCATION) != null) {
			BufferedWriter bw = new BufferedWriter(new FileWriter(Configuration
					.getGlobalConfiguration().getRegistryEntry(
							RegKeys.CLASSIFIER_DATA_LOCATION)+ "_" +s));
			System.out.println("Saving: "+ s);
			
			//bw.write(getInstances().toString());
			
			Instances inst = getInstances();
			StringBuffer text = new StringBuffer();

		    text.append("@relation").append(" ").append(Utils.quote("testing"))
		      .append("\n\n");
		    for (int i = 0; i < inst.numAttributes(); i++) {
		      text.append(inst.attribute(i)).append("\n");
		    }
		    text.append("\n").append("@data").append("\n");
		    bw.write(text.toString());
		    
		    for (int i = 0; i < inst.numInstances(); i++) {
		    	text = new StringBuffer();
		    	text.append(inst.instance(i));
		    	if (i < inst.numInstances() - 1) {
		        text.append('\n');
		    	}
		    	bw.write(text.toString());
		    }
			bw.flush();
			bw.close();
			System.out.println("Done.");
		}
	}
	
	public void saveInstances() throws IOException {
		if (Configuration.getGlobalConfiguration().getRegistryEntry(
				RegKeys.CLASSIFIER_DATA_LOCATION) != null) {
			BufferedWriter bw = new BufferedWriter(new FileWriter(Configuration
					.getGlobalConfiguration().getRegistryEntry(
							RegKeys.CLASSIFIER_DATA_LOCATION)));
			bw.write(getInstances().toString());
			bw.flush();
			bw.close();
		}
	}

	public Instances loadInstances() throws IOException {
		Instances data = null;
		if (Configuration.getGlobalConfiguration().getRegistryEntry(
				RegKeys.CLASSIFIER_DATA_LOCATION) != null) {
			BufferedReader reader = new BufferedReader(new FileReader(
					Configuration.getGlobalConfiguration().getRegistryEntry(
							RegKeys.CLASSIFIER_DATA_LOCATION)));
			data = new Instances(reader);
			reader.close();
		}
		return data;
	}

	protected Attribute generateClassAttribute() {
		Attribute classAttribute = null;
		if (classValues == null) {
			classAttribute = new Attribute(className);
		} else {
			classAttribute = new Attribute(className, classValues);
		}
		return classAttribute;
	}

	public void extractAllFeatures() {
		for (int j = 0; j < dataGrid.getHeight(); j++) {
			for (int i = 0; i < dataGrid.getWidth(); i++) {
				addInstance(extractFeatureAtIndex(i, j));
			}
		}
	}

	/**
	 * creates a new feature vector (instance in weka language) and adds it to
	 * the local feature vector set.
	 * 
	 * @param attValues
	 */
	public void addInstance(double[] attValues) {
		Instance inst = new DenseInstance(1.0, attValues);
		inst.setDataset(instances);
		instances.add(inst);
	}

	/**
	 * @return the dataGrid
	 */
	public Grid2D getDataGrid() {
		return dataGrid;
	}

	/**
	 * @param dataGrid
	 *            the dataGrid to set
	 */
	public void setDataGrid(Grid2D dataGrid) {
		this.dataGrid = dataGrid;
	}

	/**
	 * @return the labelGrid
	 */
	public Grid2D getLabelGrid() {
		return labelGrid;
	}

	/**
	 * @param labelGrid
	 *            the labelGrid to set
	 */
	public void setLabelGrid(Grid2D labelGrid) {
		this.labelGrid = labelGrid;
	}

	public boolean isConfigured() {
		return configured;
	}

	/**
	 * @return the className
	 */
	public String getClassName() {
		return className;
	}

	/**
	 * @param className
	 *            the className to set
	 */
	public void setClassName(String className) {
		this.className = className;
	}

	/**
	 * @return the classValues
	 */
	public ArrayList<String>  getClassValues() {
		return classValues;
	}

	/**
	 * @param classValues
	 *            the classValues to set
	 */
	public void setClassValues(ArrayList<String>  classValues) {
		this.classValues = classValues;
	}

	/**
	 * @return the attribs
	 */
	public ArrayList<Attribute> getAttribs() {
		return attribs;
	}

	/**
	 * @param attribs
	 *            the attribs to set
	 */
	public void setAttribs(ArrayList<Attribute> attribs) {
		this.attribs = attribs;
	}

}
