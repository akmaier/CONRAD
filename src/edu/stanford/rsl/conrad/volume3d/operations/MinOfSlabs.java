package edu.stanford.rsl.conrad.volume3d.operations;

public class MinOfSlabs extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
			
			switch (vol1.getInternalDimension()) {

			case 1:

				for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
							float tmp1 = vol1.data[indexX][indexY][indexZ];
							float tmp2 = vol2.data[indexX][indexY][indexZ];
							if (tmp1 > tmp2)
								vol1.data[indexX][indexY][indexZ] = tmp2;
						}
					}
				}

				break;

			case 2:

				for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
							float tmp1 = vol1.data[indexX][indexY][indexZ*2];
							float tmp2 = vol2.data[indexX][indexY][indexZ*2];
							if (tmp1 > tmp2)
								vol1.data[indexX][indexY][indexZ*2] = tmp2;
							tmp1 = vol1.data[indexX][indexY][indexZ*2+1];
							tmp2 = vol2.data[indexX][indexY][indexZ*2+1];
							if (tmp1 > tmp2)
								vol1.data[indexX][indexY][indexZ*2+1] = tmp2;
						}
					}
				}

				break;

			default:
				System.out.println("vol_div: Invalid dimension\n");
			}
		
	}

	@Override
	public ParallelVolumeOperation clone() {
		return new MinOfSlabs();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/