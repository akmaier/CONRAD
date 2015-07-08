/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.io.STLFileUtil;
import edu.stanford.rsl.conrad.io.SelectionCancelledException;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Class to read a single STL mesh from a file and to load it as phantom. Currently only ASCII format is supported.
 * @author akmaier
 *
 */
public class AsciiSTLMeshPhantom extends AnalyticPhantom {

	protected int debug = 1;
	/**
	 * 
	 */
	private static final long serialVersionUID = -7401895160541333397L;
	String filenameString = null;
	
	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public String getName() {
		if (filenameString == null){
			return "ASCII STL Mesh Phantom";
		} else {
			return "ASCII STL Mesh Phantom " + filenameString;
		}
	}

	@Override
	public void configure() {
		int meshes = 0;
		try {
			while(true) {
				// Only an exception breaks the loop
				
				// select file
				filenameString = FileUtil.myFileChoose(".stl", false);
				// read mesh from file
				CompoundShape mesh = STLFileUtil.readSTLMesh(filenameString);
				
				// select material
				Object materialString = UserUtil.chooseObject("Please select a material: ", "Material Selection", MaterialsDB.getMaterials(), "water");
				Material material = MaterialsDB.getMaterial(materialString.toString());
				
				// determine scaling
				double scaling = UserUtil.queryDouble("Enter scaling:", 1);
				// apply scaling
				mesh.applyTransform(new ScaleRotate(SimpleMatrix.I_3.multipliedBy(scaling)));
				
				// wrap information to a physical object
				PhysicalObject po = new PhysicalObject();
				po.setMaterial(material);
				po.setNameString(filenameString);
				po.setShape(mesh);
				// add object to scene
				add(po);
				meshes++;
			}
		} catch (SelectionCancelledException e) {
			// User clicked on "cancel"
			// Do not print stack trace
			if (debug > 0 && 0 == meshes) {
				System.err.println("No STL files have been selected");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
