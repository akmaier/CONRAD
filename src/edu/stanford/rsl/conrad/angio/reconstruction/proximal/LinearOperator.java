package edu.stanford.rsl.conrad.angio.reconstruction.proximal;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;

public abstract class LinearOperator {

	/**
	 * Norm for approximation of parameters |L* L|
	 * @return
	 */
	public abstract float getForwBackOperatorNorm();
	
	public abstract int getNumberOfComponents();
	
	public abstract void apply(ArrayList<MultiChannelGrid3D> x, ArrayList<MultiChannelGrid3D> xForw);
	
	public abstract void applyAdjoint(ArrayList<MultiChannelGrid3D> u, ArrayList<MultiChannelGrid3D> uBack);
}
