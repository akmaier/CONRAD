/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.skeleton.util;

public class BranchPoint extends Point{
	
	private boolean JUNCTION = false;
	private boolean END = false;
	
	private boolean MATCHED = false;
	
	public BranchPoint(Point p){
		super(p.x,p.y,p.z);
	}
	
	public BranchPoint(int x, int y, int z, boolean isEnd, boolean isJunction){
		super(x,y,z);
		this.setEND(isEnd);
		this.setJUNCTION(isJunction);
	}
	
	public BranchPoint(Point p, boolean isEnd, boolean isJunction){
		super(p.x,p.y,p.z);
		this.setEND(isEnd);
		this.setJUNCTION(isJunction);
	}

	public boolean isMATCHED(){
		return MATCHED;
	}
	
	public void setMATCHED(boolean matched){
		this.MATCHED = matched;
	}
	
	public boolean isJUNCTION() {
		return JUNCTION;
	}

	public void setJUNCTION(boolean iS_JUNCTION) {
		JUNCTION = iS_JUNCTION;
		if(JUNCTION)END = true;
	}

	public boolean isEND() {
		return END;
	}

	public void setEND(boolean eND) {
		END = eND;
		if(!END)JUNCTION = false;
	}

	
}
