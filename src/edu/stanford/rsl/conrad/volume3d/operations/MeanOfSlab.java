package edu.stanford.rsl.conrad.volume3d.operations;

public class MeanOfSlab extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		float m = 0f;
		for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]*vol.getInternalDimension(); indexZ++) {
					m += vol.data[indexX][indexY][indexZ];
				}
			}
		}
		result = new Float(m);
	}

	@Override
	public ParallelVolumeOperation clone() {
		return new MeanOfSlab();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/