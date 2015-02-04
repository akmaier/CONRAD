package edu.stanford.rsl.conrad.volume3d.operations;

import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.conrad.parallel.ParallelizableRunnable;
import edu.stanford.rsl.conrad.volume3d.Volume3D;

public abstract class ParallelVolumeOperation implements ParallelizableRunnable, Cloneable {

	protected CountDownLatch latch;
	// for binary operations:
	protected Volume3D vol1;
	protected Volume3D vol2;
	// for uniary operations;
	protected Volume3D vol;
	protected int beginIndexX;
	protected int endIndexX;
	protected Object result;
	// for scalar operations;
	protected float scalar1;
	protected float scalar2;
	
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}
	
	public Volume3D getVol1() {
		return vol1;
	}

	public void setVol1(Volume3D vol1) {
		this.vol1 = vol1;
	}

	public Volume3D getVol2() {
		return vol2;
	}

	public void setVol2(Volume3D vol2) {
		this.vol2 = vol2;
	}

	public Volume3D getVol() {
		return vol;
	}

	public void setVol(Volume3D vol) {
		this.vol = vol;
	}

	public Object getResult(){
		return result;
	}

	public float getScalar1() {
		return scalar1;
	}

	public void setScalar1(float scalar1) {
		this.scalar1 = scalar1;
	}

	public float getScalar2() {
		return scalar2;
	}

	public void setScalar2(float scalar2) {
		this.scalar2 = scalar2;
	}

	public int getBeginIndexX() {
		return beginIndexX;
	}

	public void setBeginIndexX(int beginIndexX) {
		this.beginIndexX = beginIndexX;
	}

	public int getEndIndexX() {
		return endIndexX;
	}

	public void setEndIndexX(int endIndexX) {
		this.endIndexX = endIndexX;
	}

	public void run(){
		performOperation();
		latch.countDown();
	}
	
	public abstract ParallelVolumeOperation clone();
	
	public abstract void performOperation();
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/