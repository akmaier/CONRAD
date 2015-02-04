package edu.stanford.rsl.conrad.volume3d.operations;

public class AddSlabScalar extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		switch (vol.getInternalDimension()) {
		case 1:
			for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						vol.data[indexX][indexY][indexZ] += scalar1;
					}
				}
			}
			break;
		case 2:   
			for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
				for (int indexY=0; indexY<vol.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol.size[2]; indexZ++) {
						vol.data[indexX][indexY][indexZ*2]   += scalar1;
						vol.data[indexX][indexY][indexZ*2+1] += scalar2;
					}
				}
			}
			break;
		default:
			System.out.println("vol_add_sc: Invalid dimension\n");
		}
	}

	@Override
	public ParallelVolumeOperation clone() {
		return new AddSlabScalar();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/