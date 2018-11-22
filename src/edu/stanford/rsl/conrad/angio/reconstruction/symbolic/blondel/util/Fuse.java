package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class Fuse {
	public PointND point;
	public int numberOfResponses;
	public Fuse(){
	}
	public Fuse(PointND p, int nr){
		this.point = p;
		this.numberOfResponses = nr;
	}
	public Fuse(PointND p){
		this.point = p;
	}
	public PointND getPoint() {
		return point;
	}
	public void setPoint(PointND point) {
		this.point = point;
	}
	public int getNumberOfResponses() {
		return numberOfResponses;
	}
	public void setNumberOfResponses(int numberOfResponses) {
		this.numberOfResponses = numberOfResponses;
	}
	
}
