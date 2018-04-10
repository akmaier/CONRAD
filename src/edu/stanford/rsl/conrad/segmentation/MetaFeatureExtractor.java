/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import ij.gui.GenericDialog;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import weka.core.Attribute;
import weka.core.Instances;

public class MetaFeatureExtractor extends GridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7493565581383426078L;
	ArrayList<Object> knownExtractors;

	boolean[] selection;

	public Instances getInstances() {
		return instances;
	}

	protected Attribute generateClassAttribute() {
		Attribute classAttribute = null;
		if (classValues == null) {
			classAttribute = new Attribute("Class");
		} else {
			classAttribute = new Attribute(className, classValues);
		}
		classAttribute = new Attribute(className);
		return classAttribute;
	}

	@Override
	public void prepareForSerialization() {
		super.prepareForSerialization();
		if (knownExtractors != null) {
			for (int i = 0; i < knownExtractors.size(); i++) {
				GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
						.get(i);
				gridFex.prepareForSerialization();
			}
		}
	}

	@Override
	public void setLabelGrid(Grid2D labelGrid) {
		super.setLabelGrid(labelGrid);
		if (knownExtractors != null) {
			for (int i = 0; i < knownExtractors.size(); i++) {
				if (selection[i]) {
					GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
							.get(i);
					gridFex.setLabelGrid(labelGrid);
				}
			}
		}
	}

	@Override
	public void setDataGrid(Grid2D dataGrid) {
		super.setDataGrid(dataGrid);
		if (knownExtractors != null) {
			for (int i = 0; i < knownExtractors.size(); i++) {
				if (selection[i]) {
					GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
							.get(i);
					gridFex.setDataGrid(dataGrid);
					;
				}
			}
		}
	}

	public void configure() throws Exception {

		knownExtractors = CONRAD
				.getInstancesFromConrad(GridFeatureExtractor.class);
		selection = new boolean[knownExtractors.size()];

		GenericDialog gDialog = new GenericDialog("Feature Configuration");
		gDialog.addMessage("Select features:");

		for (int i = 0; i < knownExtractors.size(); i++) {
			GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
					.get(i);
			if (!(gridFex instanceof MetaFeatureExtractor)) {
				gDialog.addCheckbox(gridFex.toString(), true);
			}
		}

		gDialog.showDialog();

		for (int i = 0; i < knownExtractors.size(); i++) {
			GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
					.get(i);
			if (!(gridFex instanceof MetaFeatureExtractor)) {
				selection[i] = gDialog.getNextBoolean();
				if (selection[i]) {
					gridFex.labelGrid = this.labelGrid;
					gridFex.dataGrid = this.dataGrid;
					gridFex.configure();
					this.attribs = combineAttribs(this.attribs, gridFex.attribs);
				}
			} else {
				selection[i] = false;
			}
		}

		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);

		// leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);
		configured = true;
	}

	@Override
	public double[] extractFeatureAtIndex(int x, int y) {

		double[] vector = new double[attribs.size()];
		int currentCount = 0;

		for (int i = 0; i < selection.length; i++) {
			if (selection[i]) {
				GridFeatureExtractor gridFex = (GridFeatureExtractor) knownExtractors
						.get(i);
				double[] localFeatureVector = gridFex.extractFeatureAtIndex(x,
						y);
				for (int j = 0; j < localFeatureVector.length - 1; j++) {
					vector[j + currentCount] = localFeatureVector[j];
				}
				currentCount += localFeatureVector.length - 1;
			}
		}

		vector[vector.length - 1] = this.labelGrid.getAtIndex(x, y);
		return vector;

	}

	public static ArrayList<Attribute> combineAttribs(ArrayList<Attribute> a, ArrayList<Attribute> b) {

		if (a == null) {
			a = new ArrayList<Attribute>(0);
		}
		for (int i = 0; i < b.size() - 1; i++) {
			a.add(b.get(i));
		}
		return a;
	}

	@Override
	public String getName() {
		return "Meta Features (any combination of features)";
	}

}
