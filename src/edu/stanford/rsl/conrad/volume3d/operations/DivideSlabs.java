package edu.stanford.rsl.conrad.volume3d.operations;

public class DivideSlabs extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		boolean div_z_mess = false;
		switch (vol1.getInternalDimension()) {

		case 1:

			for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	
						if (vol2.data[indexX][indexY][indexZ]==0 && div_z_mess==false) {
							System.out.println( "DivideSlabs: Division by zero in one volume element.");							
							div_z_mess=true;
						} 
						vol1.data[indexX][indexY][indexZ] = vol1.data[indexX][indexY][indexZ] / vol2.data[indexX][indexY][indexZ];
					}
				}
			}

			break;

		case 2:   

			for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
				for (int indexY=0; indexY<vol1.size[1]; indexY++) {
					for (int indexZ=0; indexZ<vol1.size[2]; indexZ++) {	

						float tmp_abs_sq =
							vol2.data[indexX][indexY][indexZ*2]   * vol2.data[indexX][indexY][indexZ*2]+
							vol2.data[indexX][indexY][indexZ*2+1] * vol2.data[indexX][indexY][indexZ*2+1];

						if (tmp_abs_sq==0 && div_z_mess==false) {
							System.out.println("vol_div: Division by zero\n");
							div_z_mess=true;
						}

						float tmp_re1 = vol1.data[indexX][indexY][indexZ*2];
						float tmp_re2 = vol2.data[indexX][indexY][indexZ*2];

						vol1.data[indexX][indexY][indexZ*2]   = 
							(vol1.data[indexX][indexY][indexZ*2]   * vol2.data[indexX][indexY][indexZ*2] +
									vol1.data[indexX][indexY][indexZ*2+1] * vol2.data[indexX][indexY][indexZ*2+1]) / tmp_abs_sq;

						vol1.data[indexX][indexY][indexZ*2+1] = 
							(vol1.data[indexX][indexY][indexZ*2+1] * tmp_re2 -
									vol2.data[indexX][indexY][indexZ*2+1] * tmp_re1) / tmp_abs_sq;

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
		return new DivideSlabs();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/