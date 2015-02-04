package edu.stanford.rsl.conrad.volume3d.operations;

public class FFTShifter extends ParallelVolumeOperation {

	@Override
	public void performOperation() {
		float [] tmp_buffer = new float[vol.getInternalDimension()];
		
		for (int indexX=beginIndexX; indexX<endIndexX; indexX++) {
			for (int indexY=0; indexY<vol.size[1]; indexY++) {
				for (int indexZ=0; indexZ<vol.size[2]/2; indexZ++) {
					int newX = (indexX + vol.size[0]/2) % vol.size[0];
					int newY = (indexY + vol.size[1]/2) % vol.size[1];
					int newZ = (indexZ + vol.size[2]/2) % vol.size[2];
					int in_dim = vol.getInternalDimension();
					for(int i=0;i<in_dim;i++){
						tmp_buffer[i]=vol.data[indexX][indexY][indexZ*in_dim+i];
					}
					for(int i=0;i<in_dim;i++){
						vol.data[indexX][indexY][indexZ*in_dim+i] = vol.data[newX][newY][newZ*in_dim+i];
					}
					for(int i=0;i<in_dim;i++){
						vol.data[newX][newY][newZ*in_dim+i] = tmp_buffer[i];
					}
				}
			}
		}

	}

	@Override
	public ParallelVolumeOperation clone() {
		return new FFTShifter();
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/