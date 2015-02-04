package edu.stanford.rsl.tutorial.phantoms;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class Phantom extends Grid2D{

	private String title;

	public Phantom(int x,int y){
		this(x, y, "phantom");
	}

	public Phantom(int x, int y, String title) {
		// TODO Auto-generated constructor stub
		super(new float[x*y], x, y);
		this.setSpacing(1,1);
		this.setOrigin(x/2.d,y/2.d);
		this.title = title;
	}

	public void setTitle(String t) {
		this.title = t;
	}
	
	public String getTitle() {
		return this.title;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/