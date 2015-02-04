package edu.stanford.rsl.conrad.phantom.electrondensity;

import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * <p>This class models <a href = "http://www.cirsinc.com/pdfs/062cp.pdf"> CRIS's Electron Density Phantom Model 062 </a>.<br/>
 * This phantom enables precise correlation of CT data in hounsfield units to electron density and includes eight different tissue references.<br/>
 * Model 062 consists of a small cylindrical disk (Inner) nested within a large spherical disk (Outer).<br/>
 * Phantom can be configured to simulate head or abdomen, by positioning tissue equivalent samples at 17 different locations within the scan field.
 * 
 * <p>The outer disks inserts are evenly spaced and labeled from 0 - 7. Assuming the face of the outer disk is a perfect circle, then Insert 0 represents the insert at (x,y,theta) = (x,0,0)<br/>
 * The inner disk inserts are evenly spaced and  labeled from 0 - 8. Assuming the face of the inner disk is a perfect circle, then Insert 0 represents the insert at (x,y,theta) = (x,0,0), while Insert 9 represents the insert at (x,y,theta) = (0,0,0).
 * 
 * [TO BE COMPLETED] * 
 * @author Rotimi X Ojo
 */

public class CrisEDPhantomM062 extends AnalyticPhantom {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7299055837928247210L;
	public static int OUTER_RING = 1;
	public static int INNER_RING = 2;
	
	private Insert[] innerDiskIns = new Insert[9];
	private Insert[] outerDiskIns  = new Insert[8];
	
	private boolean useInnerDisk= true;
	private boolean useOuterDisk = true;
	private EDInnerDisk inner = new EDInnerDisk();
	private EDOuterDisk outer = new EDOuterDisk();

	
		
	public CrisEDPhantomM062(){
			
	}
	
	@Override
	public void configure() throws Exception{
		super.configure();		
		for(int i = 0; i < 9; i++){
			innerDiskIns[i] = new Insert( MaterialsDB.getMaterial("vacuum"), Insert.UNBUFFERED_INSERT);
		}
		for(int i = 0; i < 8; i++){
			outerDiskIns[i] = new Insert( MaterialsDB.getMaterial("vacuum"), Insert.UNBUFFERED_INSERT);
		}
		new CrisEDPhantomGUI(this);

		initDisks();
	}
	
	public void setRingState(int ring, boolean state){
		if(ring == OUTER_RING){
			useOuterDisk = state;
		}else{
			useInnerDisk = state;
		}
	}
	
	public boolean getRingState(int ring){
		if(ring == OUTER_RING){
			return useOuterDisk;
		}else{
			return useInnerDisk;
		}
	}
	
	public String getInsertValue(int ring, int index){
		if(ring == OUTER_RING){
			return outerDiskIns[index].toString();
		}else{
			return innerDiskIns[index].toString();
		}
	}
	
	public int getInsertBufferState(int ring, int index) {
		if(ring == OUTER_RING){
			 return outerDiskIns[index].getBufferState();
		}else{
			 return innerDiskIns[index].getBufferState();
		}
	}

	
	public void setInsert(int ring, int index, Insert ins){
		if(ring == OUTER_RING){
			outerDiskIns[index] = ins;
		}else{
			innerDiskIns[index] = ins;
		}
	}
	
	private void initDisks() {
		if(useOuterDisk){
			for(int i = 0; i < outerDiskIns.length; i++){
				outer.addInsert(outerDiskIns[i], i);
			}
			addAll(outer);
		}
		 if(useInnerDisk){
			for(int i = 0; i < innerDiskIns.length; i++){
				inner.addInsert(innerDiskIns[i], i);
			}
			addAll(inner);
		}
	}
	
	

	@Override
	public String getName() {
		return "CRIS Phantom M062";
	}
	

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	


}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/