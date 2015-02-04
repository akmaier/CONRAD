package edu.stanford.rsl.conrad.volume3d.operations;

public abstract class VoxelOperation extends ParallelVolumeOperation {

	protected int indexX;
	protected int indexY;
	protected int indexZ;
	
	@Override
	public void performOperation() {
		for (indexX=beginIndexX; indexX<endIndexX; indexX++) {
			for (indexY=0; indexY<vol.size[1]; indexY++) {
				for (indexZ=0; indexZ<vol.size[2]; indexZ++) {
					performVoxelOperation();
				}
			}
		}
	}

	protected abstract void performVoxelOperation();

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/