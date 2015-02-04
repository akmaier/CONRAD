package edu.stanford.rsl.conrad.volume3d.operations;

public class UpperLimitSlab extends VoxelOperation {

	@Override
	protected void performVoxelOperation() {
		if (vol.data[indexX][indexY][indexZ] > scalar1)
			vol.data[indexX][indexY][indexZ] = scalar1;

	}

	@Override
	public ParallelVolumeOperation clone() {
		return new UpperLimitSlab();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/