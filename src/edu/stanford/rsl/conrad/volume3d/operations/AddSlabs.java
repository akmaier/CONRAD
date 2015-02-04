package edu.stanford.rsl.conrad.volume3d.operations;

public class AddSlabs extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
			
			switch (vol1.getInternalDimension()) {
			case 1:
				for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
							vol1.data[indexX][indexY][indexZ] +=  vol2.data[indexX][indexY][indexZ] * scalar1;
						}
					}
				}
				break;

			case 2:

				for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
					for (int indexY=0; indexY<vol1.size[1]; indexY++) {
						for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
							//System.out.println(indexY + " " + indexZ);
							vol1.data[indexX][indexY][indexZ*2] +=  vol2.data[indexX][indexY][indexZ*2] * scalar1;
							vol1.data[indexX][indexY][(indexZ*2)+1] +=  vol2.data[indexX][indexY][(indexZ*2)+1] * scalar1;
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
		return new AddSlabs();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/