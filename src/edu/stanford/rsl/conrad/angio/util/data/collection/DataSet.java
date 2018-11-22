package edu.stanford.rsl.conrad.angio.util.data.collection;

import java.io.Serializable;

public class DataSet implements Serializable{

	private static final long serialVersionUID = 1792276958072132147L;
	//
	private String dir = null;
	private String projectionFile = null;
	
	private PreproSettings preproSet;
	private ReconSettings recoSet;
	private PostproSettings postproSet;
	
	public DataSet(String dir, String projFile){
		this.setDir(dir);
		this.setProjectionFile(dir+projFile);
	}

	
	// Getter and setter
	
	public String getProjectionFile() {
		return projectionFile;
	}

	public void setProjectionFile(String projectionFile) {
		this.projectionFile = projectionFile;
	}

	public PreproSettings getPreproSet() {
		return preproSet;
	}

	public void setPrepSegSet(PreproSettings preproSet) {
		this.preproSet = preproSet;
	}

	public ReconSettings getRecoSet() {
		return recoSet;
	}

	public void setRecoSet(ReconSettings recoSet) {
		this.recoSet = recoSet;
	}


	public PostproSettings getPostproSet() {
		return postproSet;
	}


	public void setPostproSet(PostproSettings postproSet) {
		this.postproSet = postproSet;
	}


	public String getDir() {
		return dir;
	}


	public void setDir(String projectionDir) {
		this.dir = projectionDir;
	}
	
	
}
