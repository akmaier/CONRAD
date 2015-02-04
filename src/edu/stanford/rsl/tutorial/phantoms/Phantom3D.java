package edu.stanford.rsl.tutorial.phantoms;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class Phantom3D extends Grid3D{

	@SuppressWarnings("unused")
	private String title;

	public Phantom3D(int x,int y, int z){
		this(x, y, z,"phantom");
	}

	public Phantom3D(int x, int y, int z, String title) {
		// TODO Auto-generated constructor stub
		super(x, y, z, true);
		//this.setOrigin(x/2.d,y/2.d,z/2.d);
		this.title = title;
	}

	public void setTitle(String t) {
		this.title = t;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/