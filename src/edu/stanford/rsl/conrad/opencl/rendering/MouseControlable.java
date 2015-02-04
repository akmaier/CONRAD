package edu.stanford.rsl.conrad.opencl.rendering;

public interface MouseControlable {

	public void updateRotationX(double increment);
	public void updateRotationY(double increment);
	public void updateTranslationX(double increment);
	public void updateTranslationY(double increment);
	public void updateTranslationZ(double increment);
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/