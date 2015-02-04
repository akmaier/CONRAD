/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.utils;

import java.util.ArrayList;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients.Material;


/**
 * Class for obtaining information from the user easily.
 * 
 * @author akmaier
 *
 */
public abstract class UserUtil {

	/**
	 * Queries the User for an Integer value using Swing.
	 * @param message
	 * @param initialValue
	 * @return the chosen int
	 * @throws Exception
	 */
	public static int queryInt(String message, int initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + initialValue);
		if (input == null) throw new Exception("Selection aborted");
		return Integer.parseInt(input);
	}

	/**
	 * Queries the User for a Double values using Swing.
	 * @param message
	 * @param initialValue
	 * @return the chosen double
	 * @throws Exception
	 */
	public static double queryDouble(String message, double initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + initialValue);
		if (input == null) throw new Exception("Selection aborted");
		return Double.parseDouble(input);
	}

	/**
	 * Queries the User for a Double array using Swing.
	 * @param message
	 * @param initialValue
	 * @return the chosen double array
	 * @throws Exception
	 */
	public static double[] queryArray(String message, double[] initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + new SimpleVector(initialValue).toString());
		if (input == null) throw new Exception("Selection aborted");
		
		return (new SimpleVector(input)).copyAsDoubleArray();
	}

	/**
	 * Queries the User for a String value.
	 * @param message
	 * @param initialValue
	 * @return the user input
	 * @throws Exception
	 */
	public static String queryString(String message, String initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + initialValue);
		if (input == null) throw new Exception("Selection aborted");
		return input;
	}

	/**
	 * Queries the user for a Material.
	 * @param message
	 * @param messageTitle
	 * @return the chosen Material
	 * @throws Exception
	 */
	public static Material queryMaterial(String message, String messageTitle) throws Exception{
		Material [] materials = Material.values();
		return (Material) chooseObject(message, messageTitle, materials, materials[0]);
	}
		/**
	 * Queries the user for a Phantom.
	 * @param message
	 * @param messageTitle
	 * @return the chosen Phantom
	 * @throws Exception
	 */
	public static AnalyticPhantom queryPhantom(String message, String messageTitle) throws Exception{
		AnalyticPhantom [] materials = AnalyticPhantom.getAnalyticPhantoms();
		return (AnalyticPhantom) chooseObject(message, messageTitle, materials, materials[0]);
	}

	/**
	 * Queries the user for an Object that can be casted onto the Class cl.
	 * Here we parse the source tree and identify all objects that are sub classes of cl.
	 * Next, the subclasses are created using the default constructor.
	 * Only objects that can be created this way are listed. Very convenient way to ask to user for
	 * an instance of a certain class.
	 * @param message the message
	 * @param title the title of the message box
	 * @param cl the super class
	 * @return one instance of the a sub class of class
	 * @throws Exception may happen.
	 */
	public static Object queryObject (String message, String title, Class<? extends Object> cl) throws Exception{
		ArrayList<Object> list = CONRAD.getInstancesFromConrad(cl);
		Object[] obj = new Object[list.size()];
		list.toArray(obj);
		return chooseObject(message, title, obj, obj[0]);
	}
	
	/**
	 * Queries the User for a Boolean value.
	 * @param message
	 * @return the chosen boolean
	 * @throws Exception
	 */
	public static boolean queryBoolean(String message) throws Exception{
		int revan = JOptionPane.showConfirmDialog(null, message);
		return (revan == JOptionPane.YES_OPTION);
	}

	/**
	 * Asks the User to select an Object from a given array of Objects.
	 * @param message
	 * @param messageTitle
	 * @param objects
	 * @param initialObject
	 * @return the chosen object
	 * @throws Exception
	 */
	public static Object chooseObject(String message, String messageTitle, Object [] objects, Object initialObject) throws Exception{
		Object input = JOptionPane.showInputDialog(null, message, messageTitle, JOptionPane.INFORMATION_MESSAGE, null, objects, initialObject);
		if (input == null) throw new Exception("Selection aborted");
		return input;
	}

	/**
	 * Asks the User to select an Object from a given array of Objects.
	 * The index of the selected object is returned.
	 * @param message
	 * @param messageTitle
	 * @param objects
	 * @param initialObject
	 * @return the index of the chosen object
	 * @throws Exception
	 */
	public static int chooseIndex(String message, String messageTitle, Object [] objects, Object initialObject) throws Exception{
		Object input = JOptionPane.showInputDialog(null, message, messageTitle, JOptionPane.INFORMATION_MESSAGE, null, objects, initialObject);
		if (input == null) throw new Exception("Selection aborted");
		int index = -1;
		for (int i = 0; i < objects.length; i++){
			if (objects[i].equals(input)) index = i;
		}
		return index;
	}

}
