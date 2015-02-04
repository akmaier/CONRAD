/*
 * Copyright (C) 2010-2014 - Andreas Maier, Magdalena Herbst, Frank Schebesch, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.generic.complex;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class OpenCLGridTest {

	private static double smallReferenceValue = CONRAD.FLOAT_EPSILON*20;
	
	void fillBufferRandom(float[] buffer) {
		for (int i = 0; i < buffer.length; i++) {
			buffer[i] = (float) Math.random()*4 + 2f;
		}
	}

	@Test
	public void startup() {
		OpenCLUtil.getStaticContext();
	}

	@Test
	public void allMethods1DTest() {

		ComplexGrid1D grid1D;
		ComplexGrid1D grid1D2;
		ComplexGrid1D clgrid1D;
		ComplexGrid1D clgrid1D2;

		float[] buffer1 = new float[302];
		float[] buffer2 = new float[302];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);

		String parGenericGridCplx = "edu.stanford.rsl.conrad.data.generic.GenericGrid";
		String parClassGrid = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassType = "edu.stanford.rsl.conrad.data.generic.datatypes.Complex";

		ComplexPointwiseOperators pop = new ComplexPointwiseOperators();
		Method[] methods1 = pop.getClass().getDeclaredMethods();
		Method[] methods2 = pop.getClass().getSuperclass().getDeclaredMethods();
		Method[] methods = new Method[methods1.length+methods2.length];
		System.arraycopy(methods1, 0, methods, 0, methods1.length);
		System.arraycopy(methods2, 0, methods, methods1.length, methods2.length);

		for (int i = 0; i < methods.length; i++) {

			grid1D = new ComplexGrid1D(buffer1.length/2);
			grid1D.setAslinearMemory(buffer1);
			grid1D2 = new ComplexGrid1D(buffer1.length/2);
			grid1D2.setAslinearMemory(buffer2);

			clgrid1D = (ComplexGrid1D) grid1D.clone();
			clgrid1D2 = (ComplexGrid1D) grid1D2.clone();
			clgrid1D.activateCL();
			clgrid1D2.activateCL();

			int nrElements = clgrid1D.getNumberOfElements();

			System.out.println(methods[i]);

			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet1D = new Object[paramLength];
			Object[] clparamSet1D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();
				// Class<?> paramClass = Class.forName(argstr);

				if (argstr.equals(parClassGrid) || argstr.equals(parGenericGridCplx)) {
					if (c == 0) {
						paramSet1D[j] = grid1D.clone();
						clparamSet1D[j] = clgrid1D.clone();

						c++;
					}
					else {
						paramSet1D[j] = grid1D2.clone();
						clparamSet1D[j] = clgrid1D2.clone();

					}

				} else if (argstr.equals("edu.stanford.rsl.conrad.data.generic.datatypes.Gridable")) {
					paramSet1D[j] = new Complex(2.f,1.f);
					clparamSet1D[j] = new Complex(2.f,1.f);


				}

			}

			try {
				Object erg1D = methods[i].invoke(pop, paramSet1D);
				Object clerg1D = methods[i].invoke(pop, clparamSet1D);

				ComplexPointwiseOperators op = new ComplexPointwiseOperators();

				// wenn erg = null dann ist methode void
				if(erg1D == null && clerg1D == null) {
					ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet1D[0]), (ComplexGrid) (clparamSet1D[0]));
					op.abs(diff);
					Complex out = op.sum(diff).div(nrElements);
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);
				} else if (erg1D != null && clerg1D != null) {

					String ergStr1D = erg1D.getClass().getName();
					String clergStr1D = clerg1D.getClass().getName();

					//					Ergebnis ist ein Grid
					if (ergStr1D.equals(parClassGrid) && clergStr1D.equals(parClassGrid)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet1D[0]), (ComplexGrid) (clparamSet1D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 1DGrid		
					} else if (ergStr1D.equals(parClassGrid1D) && clergStr1D.equals(parClassCLGrid1D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid1D)(paramSet1D[0]), (ComplexGrid1D) (clparamSet1D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 2DGrid		
					} else if (ergStr1D.equals(parClassGrid2D) && clergStr1D.equals(parClassCLGrid2D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid2D)(paramSet1D[0]), (ComplexGrid2D) (clparamSet1D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 3DGrid		
					} else if (ergStr1D.equals(parClassGrid3D) && clergStr1D.equals(parClassCLGrid3D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid3D)(paramSet1D[0]), (ComplexGrid3D) (clparamSet1D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein Complex
					} else if (ergStr1D.equals(parClassType) && clergStr1D.equals(parClassType)) {
						double val = ((Complex)(erg1D)).sub((Complex) (clerg1D)).getMagn();
						if(methods[i].toString().contains("dotProduct(") || methods[i].toString().contains("sum(")){
							val = val/(double)nrElements/10;
						}
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + val, val < smallReferenceValue);
					} else {
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED.  Result types do not agree or are not defined.", false);
					}

				} else {
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Result types do not agree.", false);
				}
			} catch(IllegalAccessException e){
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}



	@Test
	public void allMethods2DTest() {

		ComplexGrid2D grid2D;
		ComplexGrid2D grid2D2;
		ComplexGrid2D clgrid2D;
		ComplexGrid2D clgrid2D2;

		float[] buffer1 = new float[56*302];
		float[] buffer2 = new float[56*302];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);

		String parGenericGridCplx = "edu.stanford.rsl.conrad.data.generic.GenericGrid";
		String parClassGrid = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassType = "edu.stanford.rsl.conrad.data.generic.datatypes.Complex";

		ComplexPointwiseOperators pop = new ComplexPointwiseOperators();
		Method[] methods1 = pop.getClass().getDeclaredMethods();
		Method[] methods2 = pop.getClass().getSuperclass().getDeclaredMethods();
		Method[] methods = new Method[methods1.length+methods2.length];
		System.arraycopy(methods1, 0, methods, 0, methods1.length);
		System.arraycopy(methods2, 0, methods, methods1.length, methods2.length);

		for (int i = 0; i < methods.length; i++) {

			grid2D = new ComplexGrid2D(151,56);
			grid2D.setAslinearMemory(buffer1);
			grid2D2 = new ComplexGrid2D(151,56);
			grid2D2.setAslinearMemory(buffer2);

			clgrid2D = (ComplexGrid2D) grid2D.clone();
			clgrid2D2 = (ComplexGrid2D) grid2D2.clone();
			clgrid2D.activateCL();
			clgrid2D2.activateCL();

			int nrElements = clgrid2D.getNumberOfElements();

			System.out.println(methods[i]);

			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet2D = new Object[paramLength];
			Object[] clparamSet2D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();
				// Class<?> paramClass = Class.forName(argstr);

				if (argstr.equals(parClassGrid) || argstr.equals(parGenericGridCplx)) {
					if (c == 0) {
						paramSet2D[j] = grid2D.clone();
						clparamSet2D[j] = clgrid2D.clone();

						c++;
					}
					else {
						paramSet2D[j] = grid2D2.clone();
						clparamSet2D[j] = clgrid2D2.clone();

					}

				} else if (argstr.equals("edu.stanford.rsl.conrad.data.generic.datatypes.Gridable")) {
					paramSet2D[j] = new Complex(2.f,1.f);
					clparamSet2D[j] = new Complex(2.f,1.f);


				}

			}

			try {
				Object erg2D = methods[i].invoke(pop, paramSet2D);
				Object clerg2D = methods[i].invoke(pop, clparamSet2D);

				ComplexPointwiseOperators op = new ComplexPointwiseOperators();

				// wenn erg = null dann ist methode void
				if(erg2D == null && clerg2D == null) {
					ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet2D[0]), (ComplexGrid) (clparamSet2D[0]));
					op.abs(diff);
					Complex out = op.sum(diff).div(nrElements);
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);
				} else if (erg2D != null && clerg2D != null) {

					String ergStr2D = erg2D.getClass().getName();
					String clergStr2D = clerg2D.getClass().getName();

					//					Ergebnis ist ein Grid
					if (ergStr2D.equals(parClassGrid) && clergStr2D.equals(parClassGrid)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet2D[0]), (ComplexGrid) (clparamSet2D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 1DGrid		
					} else if (ergStr2D.equals(parClassGrid1D) && clergStr2D.equals(parClassCLGrid1D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid1D)(paramSet2D[0]), (ComplexGrid1D) (clparamSet2D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 2DGrid		
					} else if (ergStr2D.equals(parClassGrid2D) && clergStr2D.equals(parClassCLGrid2D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid2D)(paramSet2D[0]), (ComplexGrid2D) (clparamSet2D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 3DGrid		
					} else if (ergStr2D.equals(parClassGrid3D) && clergStr2D.equals(parClassCLGrid3D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid3D)(paramSet2D[0]), (ComplexGrid3D) (clparamSet2D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein Complex
					} else if (ergStr2D.equals(parClassType) && clergStr2D.equals(parClassType)) {
						double val = ((Complex)(erg2D)).sub((Complex) (clerg2D)).getMagn();
						if(methods[i].toString().contains("dotProduct(") || methods[i].toString().contains("sum(")){
							val = val/(double)nrElements/10;
						}
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + val, val < smallReferenceValue);
					} else {
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED.  Result types do not agree or are not defined.", false);
					}

				} else {
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Result types do not agree.", false);
				}
			} catch(IllegalAccessException e){
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

	@Test
	public void allMethods3DTest() {

		ComplexGrid3D grid3D;
		ComplexGrid3D grid3D2;
		ComplexGrid3D clgrid3D;
		ComplexGrid3D clgrid3D2;

		float[] buffer1 = new float[35*56*150];
		float[] buffer2 = new float[35*56*150];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);

		String parGenericGridCplx = "edu.stanford.rsl.conrad.data.generic.GenericGrid";
		String parClassGrid = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D";
		String parClassType = "edu.stanford.rsl.conrad.data.generic.datatypes.Complex";

		ComplexPointwiseOperators pop = new ComplexPointwiseOperators();
		Method[] methods1 = pop.getClass().getDeclaredMethods();
		Method[] methods2 = pop.getClass().getSuperclass().getDeclaredMethods();
		Method[] methods = new Method[methods1.length+methods2.length];
		System.arraycopy(methods1, 0, methods, 0, methods1.length);
		System.arraycopy(methods2, 0, methods, methods1.length, methods2.length);

		for (int i = 0; i < methods.length; i++) {

			grid3D = new ComplexGrid3D(75,56,35);
			grid3D.setAslinearMemory(buffer1);
			grid3D2 = new ComplexGrid3D(75,56,35);
			grid3D2.setAslinearMemory(buffer2);

			clgrid3D = (ComplexGrid3D) grid3D.clone();
			clgrid3D2 = (ComplexGrid3D) grid3D2.clone();
			clgrid3D.activateCL();
			clgrid3D2.activateCL();

			int nrElements = clgrid3D.getNumberOfElements();

			System.out.println(methods[i]);

			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet3D = new Object[paramLength];
			Object[] clparamSet3D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();
				// Class<?> paramClass = Class.forName(argstr);

				if (argstr.equals(parClassGrid) || argstr.equals(parGenericGridCplx)) {
					if (c == 0) {
						paramSet3D[j] = grid3D.clone();
						clparamSet3D[j] = clgrid3D.clone();

						c++;
					}
					else {
						paramSet3D[j] = grid3D2.clone();
						clparamSet3D[j] = clgrid3D2.clone();

					}

				} else if (argstr.equals("edu.stanford.rsl.conrad.data.generic.datatypes.Gridable")) {
					paramSet3D[j] = new Complex(2.f,1.f);
					clparamSet3D[j] = new Complex(2.f,1.f);


				}

			}

			try {
				Object erg3D = methods[i].invoke(pop, paramSet3D);
				Object clerg3D = methods[i].invoke(pop, clparamSet3D);

				ComplexPointwiseOperators op = new ComplexPointwiseOperators();

				// wenn erg = null dann ist methode void
				if(erg3D == null && clerg3D == null) {
					ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet3D[0]), (ComplexGrid) (clparamSet3D[0]));
					op.abs(diff);
					Complex out = op.sum(diff).div(nrElements);
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);
				} else if (erg3D != null && clerg3D != null) {

					String ergStr3D = erg3D.getClass().getName();
					String clergStr3D = clerg3D.getClass().getName();

					//					Ergebnis ist ein Grid
					if (ergStr3D.equals(parClassGrid) && clergStr3D.equals(parClassGrid)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid)(paramSet3D[0]), (ComplexGrid) (clparamSet3D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 1DGrid		
					} else if (ergStr3D.equals(parClassGrid1D) && clergStr3D.equals(parClassCLGrid1D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid1D)(paramSet3D[0]), (ComplexGrid1D) (clparamSet3D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 2DGrid		
					} else if (ergStr3D.equals(parClassGrid2D) && clergStr3D.equals(parClassCLGrid2D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid2D)(paramSet3D[0]), (ComplexGrid2D) (clparamSet3D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein 3DGrid		
					} else if (ergStr3D.equals(parClassGrid3D) && clergStr3D.equals(parClassCLGrid3D)) {
						ComplexGrid diff = (ComplexGrid)op.subtractedBy((ComplexGrid3D)(paramSet3D[0]), (ComplexGrid3D) (clparamSet3D[0]));
						op.abs(diff);
						Complex out = op.sum(diff).div(nrElements);;
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + out, Math.abs(out.getReal()) < smallReferenceValue);

						//					Ergebnis ist ein Complex
					} else if (ergStr3D.equals(parClassType) && clergStr3D.equals(parClassType)) {
						double val = ((Complex)(erg3D)).sub((Complex) (clerg3D)).getMagn();
						if(methods[i].toString().contains("dotProduct(") || methods[i].toString().contains("sum(")){
							val = val/(double)nrElements/10;
						}
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Value = " + val, val < smallReferenceValue);
					} else {
						Assert.assertTrue("The test for the method " + methods[i] +" FAILED.  Result types do not agree or are not defined.", false);
					}

				} else {
					Assert.assertTrue("The test for the method " + methods[i] +" FAILED. Result types do not agree.", false);
				}
			} catch(IllegalAccessException e){
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

}
