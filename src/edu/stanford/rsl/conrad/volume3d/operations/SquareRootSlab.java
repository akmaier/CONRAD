package edu.stanford.rsl.conrad.volume3d.operations;

public class SquareRootSlab extends VoxelOperation {

	@Override
	protected void performVoxelOperation() {
		vol.data[indexX][indexY][indexZ] = (float) Math.sqrt(vol.data[indexX][indexY][indexZ]);
	}

	@Override
	public ParallelVolumeOperation clone() {
		return new SquareRootSlab();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/