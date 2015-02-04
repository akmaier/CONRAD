package edu.stanford.rsl.conrad.cuda;

import java.util.Random;

import org.junit.Test;

import edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.volume3d.ParallelVolumeOperator;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator;

public class CUDAVolumeTest {
	
	protected int [] size = {130, 152, 124};
	protected float [] dim = {1f, 1f, 1f};
	protected CUDAVolumeOperator cuop = new CUDAVolumeOperator();
	protected VolumeOperator op = new ParallelVolumeOperator();

	protected void assertVolumeEquality(Volume3D one, Volume3D two){
		op.subtractVolume(one, two);
		op.abs(one);
		double maxAbsoluteError = op.max(one);
		System.out.println("Max Error: " + maxAbsoluteError);
		org.junit.Assert.assertTrue(maxAbsoluteError < 0.00001);
	}
	
	protected void assertWeakVolumeEquality(Volume3D one, Volume3D two){
		op.subtractVolume(one, two);
		op.abs(one);
		double maxAbsoluteError = op.mean(one);
		System.out.println("Mean Error: " + maxAbsoluteError);
		org.junit.Assert.assertTrue(maxAbsoluteError < 0.001);
	}
	
	protected void assertVolumeDifference(Volume3D one, Volume3D two){
		op.subtractVolume(one, two);
		op.abs(one);
		double maxAbsoluteError = op.max(one);
		org.junit.Assert.assertFalse(maxAbsoluteError < 0.000001);
	}
	
	public CUDAVolume3D createCUDACopy (Volume3D vol){
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createVolume(size, dim, 1);
		float [][][] temp = cudaVol.data;
		cudaVol.data = vol.data;
		cudaVol.updateOnDevice();
		cudaVol.data = temp;
		return cudaVol;
	}
	
	@Test
	public void testMax(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		float cpu = op.max(one);
		float gpu = cuop.max(cudaVol);
		cudaVol.fetch();
		org.junit.Assert.assertTrue(cpu == gpu);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testMin(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		float cpu = op.min(one);
		float gpu = cuop.min(cudaVol);
		cudaVol.fetch();
		org.junit.Assert.assertTrue(cpu == gpu);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testMean(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		float cpu = op.mean(one);
		float gpu = cuop.mean(cudaVol);
		cudaVol.fetch();
		//System.out.println(cpu + " " + gpu);
		org.junit.Assert.assertTrue(Math.abs(cpu - gpu) < 0.00001);
		cudaVol.destroy();
		one.destroy();
	}
	
	
	@Test
	public void testAbs(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		op.abs(one);
		cuop.abs(cudaVol);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testAddScalar(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		op.addScalar(one, 1.0f, 0.0f);
		cuop.addScalar(cudaVol, 1.0f, 0.0f);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testAddVolume(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		Volume3D two = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		CUDAVolume3D cudaVol2 = createCUDACopy(two);
		op.addVolume(one, two);
		cuop.addVolume(cudaVol, cudaVol2);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		cudaVol2.destroy();
		one.destroy();
		two.destroy();
	}
	
	@Test
	public void testAddVolumeWeight(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		Volume3D two = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		CUDAVolume3D cudaVol2 = createCUDACopy(two);
		op.addVolume(one, two, 2.0);
		cuop.addVolume(cudaVol, cudaVol2, 2.0);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		cudaVol2.destroy();
		one.destroy();
		two.destroy();
	}
	
	@Test
	public void testCreateHighPassFilter(){
		Configuration.loadConfiguration();
		Volume3D one = op.createHighPassFilter(3, size, dim, 2, 1.5f);
		cuop.initCUDA();
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createHighPassFilter(3, size, dim, 2, 1.5f);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testCreateLowPassFilter(){
		Configuration.loadConfiguration();
		Volume3D one = op.createLowPassFilter(3, size, dim, 1.5f);
		cuop.initCUDA();
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createLowPassFilter(3, size, dim, 1.5f);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testCreateVolume(){
		Configuration.loadConfiguration();
		Volume3D vol = op.createVolume(size , dim, 1);
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createVolume(size, dim, 1);
		cudaVol.fetch();
		assertVolumeEquality(vol, cudaVol);
		cudaVol.destroy();
		vol.destroy();
	}
	
	@Test
	public void testDivideByVolume(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		Volume3D two = createRandomVolume();
		op.multiplyScalar(one, 1000, 0);
		op.addScalar(two, 10, 0);
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		CUDAVolume3D cudaVol2 = createCUDACopy(two);
		op.divideByVolume(one, two);
		cuop.divideByVolume(cudaVol, cudaVol2);
		cudaVol.fetch();
		assertWeakVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		cudaVol2.destroy();
		one.destroy();
		two.destroy();
	}
	
	@Test
	public void testDivideScalar(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		op.divideScalar(one, 3.0f, 0.0f);
		cuop.divideScalar(cudaVol, 3.0f, 0.0f);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testFFTShift(){
		Configuration.loadConfiguration();
		Volume3D one = createRandomVolume();
		cuop.initCUDA();
		CUDAVolume3D cudaVol = createCUDACopy(one);
		op.fftShift(one);
		cuop.fftShift(cudaVol);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	
	@Test
	public void testfilt_cos2_quad(){
		Configuration.loadConfiguration();
		cuop.initCUDA();
		float [] [] dirs = new float [6][3];
		AnisotropicFilterFunction.filt_get_filt_dirs(3, dirs);
		Volume3D one = op.createDirectionalWeights(3, size, dim, dirs[0], 1, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createDirectionalWeights(3, size, dim, dirs[0], 1, VolumeOperator.FILTER_TYPE.QUADRATIC);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testfilt_cos2r(){
		Configuration.loadConfiguration();
		cuop.initCUDA();
		float [] [] dirs = new float [6][3];
		AnisotropicFilterFunction.filt_get_filt_dirs(3, dirs);
		Volume3D one = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.NORMAL);
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.NORMAL);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testfilt_cos2r_quad(){
		Configuration.loadConfiguration();
		cuop.initCUDA();
		float [] [] dirs = new float [6][3];
		AnisotropicFilterFunction.filt_get_filt_dirs(3, dirs);
		Volume3D one = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testfilt_gauss(){
		Configuration.loadConfiguration();
		cuop.initCUDA();
		Volume3D one = op.createGaussLowPassFilter(3, size, dim, 2);
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.createGaussLowPassFilter(3, size, dim, 2);
		cudaVol.fetch();
		assertVolumeEquality(one, cudaVol);
		cudaVol.destroy();
		one.destroy();
	}
	
	@Test
	public void testfilt_solve_max_eigenValue(){
		Configuration.loadConfiguration();
		cuop.initCUDA();
		int [] size = {30, 30, 30};
		this.size = size;
		float [] [] dirs = new float [6][3];
		AnisotropicFilterFunction.filt_get_filt_dirs(3, dirs);
		
		Volume3D a11 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		Volume3D a12 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[1], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		Volume3D a13 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[2], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		Volume3D a22 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[3], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		Volume3D a23 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[4], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		Volume3D a33 = op.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[5], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		op.real(a11);
		op.real(a12);
		op.real(a13);
		op.real(a22);
		op.real(a23);
		op.real(a33);
		Volume3D [][] st = new Volume3D[3][3];
		st[0][0] = a11;
		st[0][1] = a12;
		st[0][2] = a13;
		st[1][0] = a12;
		st[1][1] = a22;
		st[1][2] = a23;
		st[2][0] = a13;
		st[2][1] = a23;
		st[2][2] = a33;
		CUDAVolume3D ca11 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[0], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D ca12 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[1], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D ca13 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[2], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D ca22 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[3], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D ca23 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[4], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		CUDAVolume3D ca33 = (CUDAVolume3D) cuop.createExponentialDirectionalHighPassFilter(3, size, dim, dirs[5], 1, 2.0f, 1.5f, VolumeOperator.FILTER_TYPE.QUADRATIC);
		cuop.real(ca11);
		cuop.real(ca12);
		cuop.real(ca13);
		cuop.real(ca22);
		cuop.real(ca23);
		cuop.real(ca33);
		CUDAVolume3D [][] cst = new CUDAVolume3D[3][3];
		cst[0][0] = ca11;
		cst[0][1] = ca12;
		cst[0][2] = ca13;
		cst[1][1] = ca22;
		cst[1][2] = ca23;
		cst[2][2] = ca33;
		
		//new ImageJ();
		Volume3D one = op.solveMaximumEigenvalue(st);
		//one.getImagePlus("CPU Result").show();
		CUDAVolume3D cudaVol = (CUDAVolume3D) cuop.solveMaximumEigenvalue(cst);
		cudaVol.fetch();
		//cudaVol.getImagePlus("CUDA Result").show();
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		assertWeakVolumeEquality(one, cudaVol);
		a11.destroy();
		a12.destroy();
		a13.destroy();
		a22.destroy();
		a23.destroy();
		a33.destroy();
		ca11.destroy();
		ca12.destroy();
		ca13.destroy();
		ca22.destroy();
		ca23.destroy();
		ca33.destroy();
		cudaVol.destroy();
		one.destroy();
	}
	
	public Volume3D createRandomVolume(){
		Volume3D vol = op.createVolume(size , dim, 1);
		Random rand = new Random();
		for (int i = 0; i < size[0]; i++)
			for (int j = 0; j < size[1]; j++)
				for (int k = 0; k < size[2]; k++) {
					vol.data[i][j][k] = (rand.nextFloat() - 0.5f)*2;
				}
		return vol;
	}

	
	
	@Test
	public void testRandomVolume(){
		Volume3D one = createRandomVolume();
		Volume3D two = createRandomVolume();
		assertVolumeDifference(one, two);
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
