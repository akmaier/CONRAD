package edu.stanford.rsl.conrad.volume3d.operations;

public class CopySlabData extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
			//System.out.println(indexX);
			switch (vol1.getInternalDimension()) {
			case 1:
				if (vol2.getInternalDimension() == 1) {
					// 1 to 1
					if ((scalar1 == scalar2)&&(scalar1 == 0f)){
						// Absolute Value
						for (int indexY=0; indexY<vol1.size[1]; indexY++) {
							for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
								vol1.data[indexX][indexY][indexZ] = Math.abs(vol2.data[indexX][indexY][indexZ]);
							}
						}
					} else {
						// Plain Copy
						for (int indexY=0; indexY<vol1.size[1]; indexY++) {
							for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
								vol1.data[indexX][indexY][indexZ] = vol2.data[indexX][indexY][indexZ];
							}
						}	
					}
				} else {
					// from 2 to 1:
					if ((scalar1 == 0f) && (scalar2 == 0f)){
						// compute power spectrum
						for (int indexY=0; indexY<vol1.size[1]; indexY++) {
							for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
								vol1.data[indexX][indexY][indexZ] = (float) Math.sqrt(Math.pow(vol2.data[indexX][indexY][indexZ*2],2 ) + Math.pow(vol2.data[indexX][indexY][indexZ*2+1],2));
							}
						}
					} else {
						if (scalar1 != 0f) {
							// get real part
							for (int indexY=0; indexY<vol1.size[1]; indexY++) {
								for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
									vol1.data[indexX][indexY][indexZ] = vol2.data[indexX][indexY][indexZ*2];
								}
							}
						} else {
							// get imaginary part
							for (int indexY=0; indexY<vol1.size[1]; indexY++) {
								for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
									vol1.data[indexX][indexY][indexZ] = vol2.data[indexX][indexY][indexZ*2+1];
								}
							}

						}
					}
				}
				break;
			case 2:
				if (vol2.getInternalDimension() == 1) {
					// from 1 to 2
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						//System.out.println(indexY);
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
							vol1.data[indexX][indexY][indexZ*2]   = vol2.data[indexX][indexY][indexZ];
						}
					}
				} else {
					// from 2 to 2
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {
							vol1.data[indexX][indexY][indexZ*2]   = vol2.data[indexX][indexY][indexZ*2];
							vol1.data[indexX][indexY][indexZ*2+1] = vol2.data[indexX][indexY][indexZ*2+1];
						}
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
		return new CopySlabData();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/