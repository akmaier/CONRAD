package edu.stanford.rsl.conrad.volume3d.operations;

public class MultiplySlabScalar extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
			switch (vol.getInternalDimension()) {
			case 1:
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						vol.data[indexX][indexY][indexZ] *= scalar1;
					}
				}
				break;

			case 2:
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						float tmp_re = vol.data[indexX][indexY][indexZ*2];
						vol.data[indexX][indexY][indexZ*2]   = scalar1 * vol.data[indexX][indexY][indexZ*2] - scalar2 * vol.data[indexX][indexY][indexZ*2+1];
						vol.data[indexX][indexY][indexZ*2+1] = scalar2 * tmp_re + scalar1 * vol.data[indexX][indexY][indexZ*2+1];
					}
				}
				break;
			default:
				System.out.println("vol_mult_sc: Invalid dimension\n");
			}
		}
	}

	@Override
	public ParallelVolumeOperation clone() {
		return new MultiplySlabScalar();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/