/*
 * Copyright (C) 2010-2014 Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.File;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.MaterialUtils;
import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">The material
 * database provides access to materials commonly used in medical physics. The
 * database has 154 preloaded materials, consisting of the first 92 elements,17
 * common compounds, and 45 mixtures.</span><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Materials in this
 * database are stored as editable XML files.Mixtures are retrieved using a name
 * and chemical formula. While elements and compounds can be retrieved using a
 * name or chemical formula. </span>
 * </p>
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">This is a list of
 * preloaded compounds :</span>
 * </p>
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt"></span>
 * </p>
 * <p>
 * <table>
 * <tbody>
 * <tr>
 * <td><strong><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Name</span></strong></td>
 * <td><strong><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Fomula</span></strong></td>
 * <td><strong><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Density</span></strong>
 * </td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Polyethylene</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">H2C</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.96</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Water</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">H2O</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.00</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Scinti-C9H10</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">C9H10</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.032</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Plastic</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">C5H8</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.18</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">PVC</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">H3C2Cl</span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.65</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">PTFE</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">CF2</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.18</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Quartz</span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">SiO2</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.2</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">NaI</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">NaI</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">3.67</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">YAP</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">YAlO3</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">5.55</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">CZT</span></td>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Cd9ZnTe10</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">5.68</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">GSO</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Gd2SO5</span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">6.7</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LuYAP-70</span></td>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Lu8Y2Al10O30</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">7.1 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">BGO</span></td>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">B4Ge3O12</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">7.13</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LSO</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Lu2SO5</span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">7.4 </span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LuYAP-80</span></td>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Lu8Y2Al10O30</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">7.5</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">PWO</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">PbWO4</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">8.28</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LuAP</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LuAlO3</span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">8.34</span></td>
 * </tr>
 * </tbody>
 * </table>
 * </p>
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">New compounds can
 * be created and added to the database by executing the code below</span>
 * </p>
 * <blockquote dir="ltr">
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Material material
 * = MaterialUtils.newMaterial(name,density,formula);<br />
 * MaterialDB.put(material);</span>
 * </p>
 * </blockquote>
 * <p dir="ltr">
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">This is a list of
 * preloaded mixtures:</span>
 * </p>
 * <p dir="ltr">
 * <table>
 * <tbody>
 * <tr>
 * <td><strong><span style=
 * "FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt; TEXT-DECORATION: underline"
 * >Name</span></strong></td>
 * <td><strong><span style=
 * "FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt; TEXT-DECORATION: underline"
 * >Density</span></strong></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Air</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.0129</span>
 * </td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Lung </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.26</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LungMoby</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.30</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Adipose</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.92 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Fat </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">0.92 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Body </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.00</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Urine </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.020</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Breast
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.020</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Intestine
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.03</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Lymph </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.03</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Pancreas
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.04</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Brain </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.04 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Testis
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.04</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Muscle
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.05</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Heart </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.05</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Aorta </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.050</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Kidney
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.05 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Liver </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.06 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Blood </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.06 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Spleen
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.06 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Cartilage
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.10 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">BoneMarrow
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.120</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Spongiosa
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.180</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Plexiglass
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.19</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">PMMA </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.195</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Sternum
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.250</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Rips_2nd
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.410</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">SpineBone
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.42</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Femur </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.430</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Humerus
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.460</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Claviculum
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.460</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Skapulum
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.460</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Rips_10th
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.510</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Skull </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.61</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Cranium
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.61 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Mandibula
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.680</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">RibBone
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.92</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Bone </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.92</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Kortikalis
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">1.920</span></td>
 * </tr>
 * <tr>
 * <td><span
 * style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">CoronaryArtery</span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.06</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">HeartLesion
 * </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.1</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Glass </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.5 </span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Teeth </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">2.5</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">LYSO </span></td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">5.37</span></td>
 * </tr>
 * <tr>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">SS304 </span>
 * </td>
 * <td><span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">7.92</span></td>
 * </tr>
 * </tbody>
 * </table>
 * </p>
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">New mixtures can
 * be created and added to the database by executing the code below</span>
 * </p>
 * <blockquote style="MARGIN-RIGHT: 0px" dir="ltr">
 * <p>
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">
 * WeightedAtomicComposition wac = new WeightedAtomicComposition();<br />
 * For all elements in mixture do:<br />
 * wac.add(element,proportion);<br />
 * Material material = MaterialUtils.newMaterial(name,density,wac);<br />
 * MaterialDB.put(material);</span>
 * </p>
 * </blockquote>
 * <p dir="ltr">
 * <span style="FONT-FAMILY: Times New Roman; FONT-SIZE: 10pt">Where element is
 * the chemical formula an elemental component of the mixture and proportion is
 * its weighted proportion in the mixture.</span>
 * </p>
 * 
 * @author Rotimi X Ojo
 * @author Andreas Maier
 */
public class MaterialsDB {

	private static String knownMaterials[] = null;
	
	private static String databaseLoc = System.getProperty("user.dir")
			+ "/data/";
	private static String materials = databaseLoc + "materials/";
	
	/**
	 * Returns a list of all preconfigured polychromatic materials. (See table at the beginning of this
	 * class)
	 * @return the list of materials as array
	 */
	public static String[] getPredefinedMaterials(){
		return NameToFormulaMap.map.keySet().toArray(new String[FormulaToNameMap.map.size()]);
	}
	
	/**
	 * Returns a list of all currently known polychromatic materials. (See table at the beginning of this
	 * class)
	 * @return the list of materials as array
	 */
	public static String[] getMaterials(){
		if (knownMaterials == null){
			ArrayList<String> matList = new ArrayList<String>();
			File list = new File(materials);
			for (String candidate:list.list()){
				if (candidate.endsWith(".xml")){
					matList.add(candidate.replace(".xml", ""));
				}
			}
			knownMaterials = matList.toArray(new String[matList.size()]);
		}
		return knownMaterials;
	}
	
	
	/**
	 * Retrieves material associated with given identifier. 	 * 
	 * @param identifier
	 *            Name or Formula of material
	 * @return null if material is not defined
	 */
	public static Material getMaterial(String identifier) {
		identifier = identifier.trim();
		Material material = getMaterialWithFormula(identifier);
		if (material == null) {
			material = getMaterialWithName(identifier);
		}
		return material;
	}

	/**
	 * Retrieves material associated with given formula
	 * 
	 * @param formula
	 *            is chemical formula of compound or mixture defining material
	 *            to be retrieved
	 * @return null if material is undefined
	 */
	public static Material getMaterialWithFormula(String formula) {
		formula = formula.trim();
		String name = FormulaToNameMap.getName(formula);
		if (name == null) {
			return null;
		}
		return getMaterialWithName(name);
	}

	/**
	 * Retrieves material associated with name
	 * 
	 * @param name
	 *            is name of material to be retrieved
	 * @return null if material material does not exist and cannot be generated
	 *         on the fly
	 */
	public static Material getMaterialWithName(String name) {
		name = name.toLowerCase().trim();
		File file = new File(materials + name + ".xml");
		return (Material) XmlUtils.deserializeObject(file);
	}

	/**
	 * Adds a new material to database. New Materials are created by
	 * {@link MaterialUtils}
	 * 
	 * @param material
	 * @return false if material cannot be added to database
	 */
	public static boolean put(Material material) {
		File file = new File(materials + material.getName() + ".xml");
		return XmlUtils.serializeObject(file, material);
	}

	/**
	 * Removes material with given identifier from database
	 * 
	 * @param identifier
	 *            is name or formula of material to be removed
	 * @return false if removal is unsuccessful
	 */
	public static boolean removeMaterial(String identifier) {
		identifier = identifier.trim();
		if (removeMaterialWithFormula(identifier) == false) {
			return removeMaterialWithName(identifier);
		}
		return false;
	}

	/**
	 * Removes material defined by given formula from database
	 * 
	 * @param formula
	 *            is chemical formula of material of interest
	 * @return false if removal is unsuccessful
	 */
	public static boolean removeMaterialWithFormula(String formula) {
		formula = formula.trim();
		String name = FormulaToNameMap.getName(formula);
		if (name == null) {
			return false;
		}
		return removeMaterialWithName(name);
	}

	/**
	 * Removes material associated with given name from database
	 * 
	 * @param name
	 *            is name of material of interest
	 * @return false if removal is unsuccessful
	 */
	public static boolean removeMaterialWithName(String name) {
		name = name.trim().toLowerCase();
		
		File file = new File(materials + name + ".xml");
		if (file.exists()) {
			return file.delete();
		}
		return false;
	}

	/**
	 * Absolute file path of database
	 * 
	 * @return the location of databases
	 */
	public static String getDatabaseLocation() {
		return databaseLoc;
	}

	/**
	 * Absolute file path of materials database
	 * 
	 * @return the location of material database
	 */
	public static String getMaterialsLocation() {
		return materials;
	}



}
