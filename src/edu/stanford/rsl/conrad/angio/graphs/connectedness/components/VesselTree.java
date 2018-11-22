/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness.components;

import java.util.ArrayList;

public class VesselTree extends ArrayList<VesselBranch>{

	private static final long serialVersionUID = 8579091409641832177L;
	
	public VesselTree() {
		super();
	}
	
	public void addVesselBranch(VesselBranch branch) {
		this.add(branch);
	}
	
	

}
