/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * The class AnalyticPhantom defines a scene with different PhysicalObjects, i.e. Objects with geometric shape that consist of a Material.
 * Materials define the density as well as the energy-dependent absorption coefficients of the object.
 * <BR><BR>
 * Everything that is an AnalyticPhantom can be projected in a monochromatic or polychromatic fashion.
 *  
 * @author akmaier
 *
 */
public abstract class AnalyticPhantom extends PrioritizableScene implements Citeable, GUIConfigurable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 9211271452251474622L;
	private boolean configured = false;
	
	public void prepareForSerialization(){
		
	}

	@Override
	public boolean isConfigured(){
		return configured;
	}

	/**
	 * Reads all available phantoms from the current class path.
	 * @return an array of phantoms.
	 */
	public static AnalyticPhantom [] getAnalyticPhantoms(){
		AnalyticPhantom[] aList = null;
		if (aList ==null) {
			try {
				ArrayList<Object> list = CONRAD.getInstancesFromConrad(AnalyticPhantom.class);
				aList = new AnalyticPhantom[list.size()];
				for (int i = 0; i < list.size(); i++){
					aList[i] = (AnalyticPhantom) list.get(i);
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

	public abstract String getName();

	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

	public void configure() throws Exception{
		configured = true;
	}

	@Override
	public String toString(){
		return getName();
	}
	
	public PrioritizableScene tessellatePhantom(double accuracy){
		PrioritizableScene scene = new PrioritizableScene();
		for (PhysicalObject o : this){
			PhysicalObject o2 = new PhysicalObject(o);
			o2.setShape(((AbstractSurface)o.getShape()).tessellate(accuracy));
			o2.setMaterial(o.getMaterial());
			scene.add(o2);
		}
		
		return scene;
		
	}
	
	/**
	 * Computes a Translation that will center the phantom around the origin, if applied to the phantom.
	 * @return the translation 
	 */
	public Translation computeCenterTranslation(){
		// convert to internal notation
		SimpleVector center = SimpleOperators.add(getMax().getAbstractVector(), getMin().getAbstractVector()).dividedBy(2);
		Translation centerTranslation = new Translation(center.negated());
		return centerTranslation;
	}
	
	public static AnalyticPhantom getCurrentPhantom() throws Exception{
		AnalyticPhantom phantom = null;
		String defaultPhantom = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.SPLINE_4D_LOCATION);
		if (defaultPhantom != null) {
			try {
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(defaultPhantom));
				phantom = (AnalyticPhantom) ois.readObject();
				ois.close();
			} catch (FileNotFoundException e){
			} catch (IOException e) {
			} catch (ClassNotFoundException e) {
			}
		}
		if (phantom == null) {
			phantom = UserUtil.queryPhantom("Select Phantom", "Select Phantom");
			Material mat = null;		
			do{
				String materialstr = UserUtil.queryString("Enter Background Medium:", "vacuum");
				mat = MaterialsDB.getMaterialWithName(materialstr);
			} while(mat == null);
			phantom.setBackground(mat);
			phantom.configure();
			if (defaultPhantom != null) {
				phantom.prepareForSerialization();
				try {
					ObjectOutputStream ooStream = new ObjectOutputStream(new FileOutputStream(defaultPhantom));
					ooStream.writeObject(phantom);
					ooStream.flush();
					ooStream.close();
				} catch (Exception e){
					e.printStackTrace();
					return null;
				}
			}
		}
		return phantom;
	}
}
