/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.apodization;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public abstract class ApodizationWindow {
	
	protected Grid2D window = null;
	protected int width = 0;
	protected int height = 0;
	
	public ApodizationWindow(int w, int h){
		this.width = w;
		this.height = h;
		this. window = setupWindow();
	}
	
	protected abstract Grid2D setupWindow();

	public Grid2D getWindow(){
		return window;
	}
}
