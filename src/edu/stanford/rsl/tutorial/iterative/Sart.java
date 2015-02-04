package edu.stanford.rsl.tutorial.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public interface Sart {
	
	public void iterate(final int iter) throws Exception;
	public void iterate() throws Exception;
	
	public Grid3D getVol();	
}