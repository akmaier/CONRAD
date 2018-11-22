/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;

public abstract class SmoothFunction {

	
	public abstract float evaluate(ArrayList<MultiChannelGrid3D> x);
	
	public abstract void evaluateGradient(ArrayList<MultiChannelGrid3D> x, ArrayList<MultiChannelGrid3D> xGrad);
	
}
