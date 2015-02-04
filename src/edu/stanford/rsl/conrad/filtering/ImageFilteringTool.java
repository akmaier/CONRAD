/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.filtering;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.io.SafeSerializable;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Abstract class to describe the blue print of an ImageFilteringTool. In order to be executed in a
 * parallel manner it needs to be Cloneable.
 * 
 * @author Andreas Maier
 *
 */
public abstract class ImageFilteringTool implements Cloneable, SafeSerializable, GUIConfigurable, Citeable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2772279250470371073L;
	/**
	 * 
	 */

	protected boolean configured;


	/**
	 * returns true, if the filter models or compensates device dependent, i.e. physical characteristics.
	 * @return true if the method is device dependent.
	 */
	public abstract boolean isDeviceDependent();
	
	

	
	/**
	 * returns the name of the actual tool which was used.
	 * @return the name of the tool as string.
	 */
	public abstract String getToolName();
	

	/**
	 * returns getToolName();
	 */
	public String toString(){
		return getToolName();
	}
	

	/**
	 * Gives an Array with default instances of all known image filters.
	 * @return the array of known ImageFilteringTools
	 */
	public static ImageFilteringTool [] getFilterTools(){
		ArrayList<Object> found;
		ImageFilteringTool [] list = null;
		try {
			found = CONRAD.getInstancesFromConrad(ImageFilteringTool.class);
			list = new ImageFilteringTool[found.size()];
			found.toArray(list);
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return list;
	}
	

	@Override
	public abstract void prepareForSerialization();
	
	/**
	 * Creates a clone of the filter with the same configuration as the original.
	 */
	public abstract ImageFilteringTool clone();




	/**
	 * @return the configured
	 */
	public boolean isConfigured() {
		return configured;
	}




	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
}

