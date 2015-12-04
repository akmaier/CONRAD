package edu.stanford.rsl.conrad.phantom.electrondensity;

import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;


/**
 * Models the insert of an ED Phantom. Inserts can be either buffered or unbuffered. The default buffer material is water. 
 * @author Rotimi X Ojo
 *
 */
public class Insert extends PrioritizableScene {

	private static final long serialVersionUID = -6549158504106309805L;
	public static int BUFFERED_INSERT = 1;
	public static int UNBUFFERED_INSERT = 2;
	
	private double dx = 30.5/2,dy = 30.5/2, dz = 50/2;	
	private QuadricDisk buffer = new QuadricDisk(dx, dy, dz);
	private QuadricDisk disk;
	private Material bufferMaterial = MaterialsDB.getMaterial("water");
	private SimpleMatrix mat = new SimpleMatrix(3,3);
	private int bufferState = UNBUFFERED_INSERT;
	
	/**
	 * 
	 * @param material
	 * @param bufferState
	 */
	public Insert(Material material, int bufferState, double diameter) {
		mat.identity();
		if(material == null){
			material =  MaterialsDB.getMaterialWithName("air");
		}
		
		this.bufferState = bufferState;
		
		if(bufferState== BUFFERED_INSERT){
			buffer.setMaterial(bufferMaterial);
			buffer.setNameString("InsertBuffer");
			disk = new QuadricDisk(diameter, diameter, dz);
			disk.setMaterial(material);	
			disk.setNameString("Insert");
		}else{
			buffer.setMaterial(material);
			buffer.setNameString("InsertBuffer");
		}
		if(disk != null){
			add(disk, 10);
		}
		add(buffer, 9);
		
	}
	
	public Insert(Material material, int bufferState) {
		this(material, bufferState, 3);
	}
	
	/**
	 * 
	 * @param material
	 */
	public void setBufferMaterial(Material material){
		this.bufferMaterial = material;
	}
	/**
	 * Gets the material of the buffer.
	 * @return the Material
	 */
	public Material getBufferMaterial() {
		return bufferMaterial;
	}
	
	/**
	 * 
	 * @param index
	 * @param distanceFromOrigin
	 */
	public void setLocation(int index, double distanceFromOrigin) {
		double angle = (index)*Math.PI/4;
		double x = distanceFromOrigin*Math.cos(angle);
		double y = distanceFromOrigin*Math.sin(angle);
		double z = 0;
		buffer.applyTransform(new AffineTransform(mat, new SimpleVector(x,y,z)));
		if(disk != null){
			disk.applyTransform(new AffineTransform(mat, new SimpleVector(x,y,z)));
		}
	}
	
	@Override
	public String toString(){
		if(disk != null){
			return disk.getMaterial().getName();
		}
		return buffer.getMaterial().getName();
	}

	/**
	 * 
	 * @return the state of the buffer.
	 */
	public int getBufferState() {
		return bufferState;
	}


}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/