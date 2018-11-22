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
