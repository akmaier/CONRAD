/*
 * Copyright (C) 2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.filtering;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Filter to apply an absorption model to a sequence of multi channel grids. With this class you only need to render your phantoms once. Then you can apply any absorption model to the data.
 * This is very useful, if you want to investigate the effects of different detector configurations. Detector configuration is read from global configuration.
 * @author akmaier
 *
 */
public class SimulateXRayDetector extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3973031220832239032L;
	
	@Override
	public void configure() throws Exception {
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		SimulateXRayDetector clone = new SimulateXRayDetector();
		clone.configured = this.configured;
		return clone;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		Grid2D result = imageProcessor;
		XRayDetector model = Configuration.getGlobalConfiguration().getDetector();
		if (imageProcessor instanceof MultiChannelGrid2D){
			MultiChannelGrid2D multiChannelGrid2D = (MultiChannelGrid2D) imageProcessor;
			result = model.createDetectorGrid(multiChannelGrid2D.getWidth(), multiChannelGrid2D.getHeight());
			Material[] materials = new Material[multiChannelGrid2D.getNumberOfChannels()];
			ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();
			for (int c=0;c<materials.length; c++){
				String name = multiChannelGrid2D.getChannelNames()[c];
				if (name.contains("water w rho=")) {
					// Exception for XCAT.
					materials[c] = MaterialsDB.getMaterial("water");
					String [] parts = name.split("water w rho=");
					String [] number = parts[1].split(" ");
					double density = Double.parseDouble(number[0]);
					materials[c].setDensity(density);
					materials[c].setName(name);
				} else {
					// Normal materials.
					materials[c] = MaterialsDB.getMaterial(name);
				}
				PhysicalObject object = new PhysicalObject();
				object.setMaterial(materials[c]);
				object.setNameString(name);
				segments.add(object);
			}
			for (int j=0; j<multiChannelGrid2D.getHeight(); j++){
				for (int i=0; i<multiChannelGrid2D.getWidth();i++){
					for (int c=0; c<materials.length; c++){
						segments.get(c).setShape(new Edge(new PointND(0), new PointND(multiChannelGrid2D.getPixelValue(i, j, c))));
					}
					model.writeToDetector(result, i, j, segments);
				}
			}
		} else {
			// Only one material assuming water.
			// See http://lists.fau.de/pipermail/project-conrad/2014-November/000069.html for discussion in the mailing list.
			Material material = MaterialsDB.getMaterial("water");
			for (int j=0; j<result.getHeight(); j++){
				for (int i=0; i<result.getWidth();i++){
					ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();
					PhysicalObject object = new PhysicalObject();
					object.setMaterial(material);
					object.setNameString("water");
					object.setShape(new Edge(new PointND(0), new PointND(result.getPixelValue(i, j))));
					segments.add(object);
					model.writeToDetector(result, i, j, segments);
				}
			}
		}
		if (imageIndex == Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize() -1) model.notifyEndOfRendering();
		return result;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getToolName() {
		String name = "Simulate X-Ray Detector " + Configuration.getGlobalConfiguration().getDetector().toString();
		return name;
	}

}
