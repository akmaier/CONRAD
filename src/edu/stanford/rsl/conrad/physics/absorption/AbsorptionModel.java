/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.absorption;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Creates a absorption model for the projection. Note that the absorption model is also able to model
 * multi-channel data for multi-material or multi-energy applications.
 * 
 * @author Rotimi X Ojo
 * @author Andreas Maier
 */
public abstract class AbsorptionModel implements GUIConfigurable, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6082919111796044239L;
	public static int COHERENT_SCATTER = 1;
	public static int INCOHERENT_SCATTER = 2;

	/**
	 * Evaluates the absorption along the line integral according to the model.
	 * @param segments
	 * @return the integral value
	 */
	public abstract double evaluateLineIntegral(ArrayList<PhysicalObject> segments);		

	@Override
	public abstract String toString();

	
	/**
	 * Creates an array of all known absorption models known in the current ClassLoader.
	 * 
	 * @return the absorption model array
	 */
	public static AbsorptionModel[] getAvailableAbsorptionModels(){
		AbsorptionModel[] aList = null;
		if (aList ==null) {
			try {
				ArrayList<Object> list = CONRAD.getInstancesFromConrad(AbsorptionModel.class);
				aList = new AbsorptionModel[list.size()];
				for (int i = 0; i < list.size(); i++){
					aList[i] = (AbsorptionModel) list.get(i);
				}
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return aList;
	}

	/**
	 * Method that is notified once the rendering process using this AbsorptionModel is finished.
	 * This method is intended to be used with AbsorptionModels that collect information about the
	 * rendering process. For example a histogram of path lengths could be stored in an AbsortionModel
	 * and in this call the result could be stored in the registry or a certain file.
	 * <br>
	 * The default implementation just does nothing. 
	 */
	public void notifyEndOfRendering(){	
	}
	
}
