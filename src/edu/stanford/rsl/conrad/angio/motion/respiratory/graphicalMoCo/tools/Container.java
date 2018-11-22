package edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo.tools;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;

public class Container {
	public Grid3D img;
	public Projection[] pMats;
	public ArrayList<float[]> shifts;
	
	public Container(Grid3D img, Projection[] ps, ArrayList<float[]> sh){
		this.img = img;
		this.pMats = ps;
		this.shifts = sh;
	}
	
	public Container(Grid3D img, Projection[] ps){
		this.img = img;
		this.pMats = ps;
		this.shifts = null;
	}
}
