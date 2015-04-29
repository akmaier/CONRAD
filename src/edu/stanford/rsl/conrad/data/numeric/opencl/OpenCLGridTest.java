/*
 * Copyright (C) 2010-2014 - Andreas Maier, Magdalena Herbst, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.data.numeric.opencl;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class OpenCLGridTest {
	
	private static final double smallValue = 1e-6;

	void fillBufferRandom(float[] buffer) {
		for (int i = 0; i < buffer.length; i++) {
			buffer[i] = (float) Math.random();
		}
	}
	
	void fillBufferOnes(float[] buffer) {
		for (int i = 0; i < buffer.length; i++) {
			buffer[i] = 1.0f;
		}
	}


	@Test
	public void allMethods1DTest() {
		
		Grid1D grid1D;
		Grid1D grid1D2;
		OpenCLGrid1D clgrid1D;
		OpenCLGrid1D clgrid1D2;
		
		float[] buffer1 = new float[300];
		float[] buffer2 = new float[300];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);

		String parClassGrid = "edu.stanford.rsl.conrad.data.numeric.NumericGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.numeric.Grid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.numeric.Grid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.numeric.Grid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D";
		String parClassFloat = "java.lang.Float";
		String parClassDouble = "java.lang.Double";


		Method[] methods = NumericPointwiseOperators.class.getDeclaredMethods();
		
		for (int i = 0; i < methods.length; i++) {
			
			grid1D = new Grid1D(buffer1);
			grid1D2 = new Grid1D(buffer2);
			clgrid1D = new OpenCLGrid1D(grid1D);
			clgrid1D2 = new OpenCLGrid1D(grid1D2);

			
			//Check if its a auxiliary method? Yes, ignore it!
			if (methods[i].getName().toLowerCase().contains(new String("selectGridOperator").toLowerCase()))
				continue;
			
			System.out.println(formatMethodName(methods[i]));
			
			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet1D = new Object[paramLength];
			Object[] clparamSet1D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();
				// Class<?> paramClass = Class.forName(argstr);

				if (argstr.equals(parClassGrid)) {
					if (c == 0) {
						paramSet1D[j] = grid1D.clone();
						clparamSet1D[j] = clgrid1D.clone();

						c++;
					}
					else {
						paramSet1D[j] = grid1D2.clone();
						clparamSet1D[j] = clgrid1D2.clone();
						
					}

				} else if (argstr.equals("float")) {
					paramSet1D[j] = 2.f;
					clparamSet1D[j] = 2.f;

					
				} else if (argstr.equals("double")) {
					paramSet1D[j] = 2.0;
					clparamSet1D[j] = 2.0;

				}

			}

			try {
				Object erg1D = methods[i].invoke(null, paramSet1D);
				Object clerg1D = methods[i].invoke(null, clparamSet1D);

				
				// wenn erg = null dann ist methode void
				if(erg1D == null && clerg1D == null) {
					double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(paramSet1D[0]), (NumericGrid) (clparamSet1D[0]))));
					Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
				} else if (erg1D != null && clerg1D != null) {
					
					String ergStr1D = erg1D.getClass().getName();
					String clergStr1D = clerg1D.getClass().getName();
					
//					Ergebnis ist ein Grid
					if (ergStr1D.equals(parClassGrid) && clergStr1D.equals(parClassGrid)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(erg1D), (NumericGrid) (clerg1D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);

//					Ergebnis ist ein 1DGrid		
					} else if (ergStr1D.equals(parClassGrid1D) && clergStr1D.equals(parClassCLGrid1D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid1D)(erg1D), (Grid1D) (clerg1D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//					Ergebnis ist ein 2DGrid		
					} else if (ergStr1D.equals(parClassGrid2D) && clergStr1D.equals(parClassCLGrid2D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid2D)(erg1D), (Grid2D) (clerg1D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
						
//					Ergebnis ist ein 3DGrid		
					} else if (ergStr1D.equals(parClassGrid3D) && clergStr1D.equals(parClassCLGrid3D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid3D)(erg1D), (Grid3D) (clerg1D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//					Ergebnis ist ein float
					} else if (ergStr1D.equals(parClassFloat) && clergStr1D.equals(parClassFloat)) {
						double diff = Math.pow((float)erg1D - (float)clerg1D, 2);
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);						
//					Ergebnis ist ein double
					} else if (ergStr1D.equals(parClassDouble) && clergStr1D.equals(parClassDouble)) {
						double diff = Math.pow((Double)erg1D - (Double)clerg1D, 2);
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);

					} else {
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED.  Result types do not agree or are not defined.", false);
					}
					
				} else {
					Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Result types do not agree.", false);
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

		Grid2D grid2D;
		Grid2D grid2D2;
		OpenCLGrid2D clgrid2D;
		OpenCLGrid2D clgrid2D2;
		
		float[] buffer1 = new float[300];
		float[] buffer2 = new float[300];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);

		String parClassGrid = "edu.stanford.rsl.conrad.data.numeric.NumericGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.numeric.Grid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.numeric.Grid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.numeric.Grid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D";
		String parClassFloat = "java.lang.Float";
		String parClassDouble = "java.lang.Double";

		Method[] methods = NumericPointwiseOperators.class.getDeclaredMethods();

		for (int i = 0; i < methods.length; i++) {

			grid2D = new Grid2D(buffer1, 10, 30);
			grid2D2 = new Grid2D(buffer2, 10, 30);
			clgrid2D = new OpenCLGrid2D(grid2D);
			clgrid2D2 = new OpenCLGrid2D(grid2D2);

			//Check if its a auxiliary method? Yes, ignore it!
			if (methods[i].getName().toLowerCase().contains(new String("selectGridOperator").toLowerCase()))
				continue;
			
			System.out.println(formatMethodName(methods[i]));
			
			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet2D = new Object[paramLength];
			Object[] clparamSet2D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();

				if (argstr.equals(parClassGrid)) {
					if (c == 0) {
						paramSet2D[j] = grid2D.clone();
						clparamSet2D[j] = clgrid2D.clone();
						c++;
					}
					else {
						paramSet2D[j] = grid2D2.clone();
						clparamSet2D[j] = clgrid2D2.clone();
					}
				} else if (argstr.equals("float")) {
					paramSet2D[j] = 2.f;
					clparamSet2D[j] = 2.f;					
				} else if (argstr.equals("double")) {
					paramSet2D[j] = 2.0;
					clparamSet2D[j] = 2.0;
				}
			}

			try {
				Object erg2D = methods[i].invoke(null, paramSet2D);
				Object clerg2D = methods[i].invoke(null, clparamSet2D);

				
				// wenn erg = null dann ist methode void
				if(erg2D == null && clerg2D == null) {
					double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(paramSet2D[0]), (NumericGrid) (clparamSet2D[0]))));
					if ((methods[i].toString().equals("public static void edu.stanford.rsl.conrad.data.NumericalPointwiseOperators.exp(edu.stanford.rsl.conrad.data.numeric.NumericGrid)"))
					   || (methods[i].toString().equals("public static void edu.stanford.rsl.conrad.data.NumericalPointwiseOperators.log(edu.stanford.rsl.conrad.data.numeric.NumericGrid)"))
					   || (methods[i].toString().equals("public static void edu.stanford.rsl.conrad.data.NumericalPointwiseOperators.pow(edu.stanford.rsl.conrad.data.numeric.NumericGrid,double)"))
					   || (methods[i].toString().equals("public static void edu.stanford.rsl.conrad.data.NumericalPointwiseOperators.sqr(edu.stanford.rsl.conrad.data.numeric.NumericGrid)"))
					   || (methods[i].toString().equals("public static void edu.stanford.rsl.conrad.data.NumericalPointwiseOperators.sqrt(edu.stanford.rsl.conrad.data.numeric.NumericGrid)"))
						)
					{
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
					} else {
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
					}
				} else if (erg2D != null && clerg2D != null) {
					
					String ergStr2D = erg2D.getClass().getName();
					String clergStr2D = clerg2D.getClass().getName();
					
//					Ergebnis ist ein Grid
					if (ergStr2D.equals(parClassGrid) && clergStr2D.equals(parClassGrid)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(erg2D), (NumericGrid) (clerg2D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);

//					Ergebnis ist ein 1DGrid		
					} else if (ergStr2D.equals(parClassGrid1D) && clergStr2D.equals(parClassCLGrid1D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid1D)(erg2D), (Grid1D) (clerg2D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//					Ergebnis ist ein 2DGrid		
					} else if (ergStr2D.equals(parClassGrid2D) && clergStr2D.equals(parClassCLGrid2D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid2D)(erg2D), (Grid2D) (clerg2D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
						
//					Ergebnis ist ein 3DGrid		
					} else if (ergStr2D.equals(parClassGrid3D) && clergStr2D.equals(parClassCLGrid3D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid3D)(erg2D), (Grid3D) (clerg2D))));
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//						Ergebnis ist ein float
						} else if (ergStr2D.equals(parClassFloat) && clergStr2D.equals(parClassFloat)) {
							double diff = Math.pow((float)erg2D - (float)clerg2D, 2);
							Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);						
//						Ergebnis ist ein double
						} else if (ergStr2D.equals(parClassDouble) && clergStr2D.equals(parClassDouble)) {
							double diff = Math.pow((Double)erg2D - (Double)clerg2D, 2);
							Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);
					} else {
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED.  Result types do not agree or are not defined.", false);
					}
				} else {
					Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Result types do not agree.", false);
				}
			} catch (IllegalAccessException e) {
				// TODO Auto-generated catch block
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

		Grid2D grid2D;
		Grid2D grid2D2;
		//OpenCLGrid2D clgrid2D;
		//OpenCLGrid2D clgrid2D2;
		
		Grid3D grid3D;
		Grid3D grid3D2;
//		OpenCLGrid3D clgrid3D;
//		OpenCLGrid3D clgrid3D2;
		
		float[] buffer1 = new float[300];
		float[] buffer2 = new float[300];
		fillBufferRandom(buffer1);
		fillBufferRandom(buffer2);


		String parClassGrid = "edu.stanford.rsl.conrad.data.numeric.NumericGrid";
		String parClassGrid2D = "edu.stanford.rsl.conrad.data.numeric.Grid2D";
		String parClassGrid1D = "edu.stanford.rsl.conrad.data.numeric.Grid1D";
		String parClassGrid3D = "edu.stanford.rsl.conrad.data.numeric.Grid3D";
		String parClassCLGrid2D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D";
		String parClassCLGrid1D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D";
		String parClassCLGrid3D = "edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D";
		String parClassFloat = "java.lang.Float";
		String parClassDouble = "java.lang.Double";

		Method[] methods = NumericPointwiseOperators.class.getDeclaredMethods();

		for (int i = 0; i < methods.length; i++) {

			grid2D = new Grid2D(buffer1, 10, 30);
			grid2D2 = new Grid2D(buffer2, 10, 30);
			//clgrid2D = new OpenCLGrid2D(grid2D, context, device);
			//clgrid2D2 = new OpenCLGrid2D(grid2D2, context, device);
			
			grid3D = new Grid3D(10, 30, 2);
			grid3D2 = new Grid3D(10, 30, 2);
			grid3D.setSubGrid(0, grid2D);
			grid3D.setSubGrid(1, grid2D);
			grid3D2.setSubGrid(0, grid2D2);
			grid3D2.setSubGrid(1, grid2D2);
			OpenCLGrid3D clgrid3D = new OpenCLGrid3D(grid3D);
			OpenCLGrid3D clgrid3D2 = new OpenCLGrid3D(grid3D2);
			
			//Check if its a auxiliary method? Yes, ignore it!
			if (methods[i].getName().toLowerCase().contains(new String("selectGridOperator").toLowerCase()))
				continue;

			System.out.println(formatMethodName(methods[i]));
			
			Class<?>[] param = methods[i].getParameterTypes();

			int paramLength = param.length;
			int c = 0;
			Object[] paramSet3D = new Object[paramLength];
			Object[] clparamSet3D = new Object[paramLength];


			for (int j = 0; j < paramLength; j++) {

				String argstr = param[j].getName();
				// Class<?> paramClass = Class.forName(argstr);

				if (argstr.equals(parClassGrid)) {
					if (c == 0) {
						paramSet3D[j] = grid3D.clone();
						clparamSet3D[j] = clgrid3D.clone();

						c++;
					}
					else {
						paramSet3D[j] = grid3D2.clone();
						clparamSet3D[j] = clgrid3D2.clone();
						
					}

				} else if (argstr.equals("float")) {
					paramSet3D[j] = 2.f;
					clparamSet3D[j] = 2.f;

					
				} else if (argstr.equals("double")) {
					paramSet3D[j] = 2.0;
					clparamSet3D[j] = 2.0;

				}

			}

			try {
				Object erg3D = methods[i].invoke(null, paramSet3D);
				Object clerg3D = methods[i].invoke(null, clparamSet3D);

				
				// wenn erg = null dann ist methode void
				if(erg3D == null && clerg3D == null) {
					double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(paramSet3D[0]), (NumericGrid) (clparamSet3D[0]))))/(double)grid3D.getNumberOfElements();
					Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
				} else if (erg3D != null && clerg3D != null) {
					
					String ergStr3D = erg3D.getClass().getName();
					String clergStr3D = clerg3D.getClass().getName();
					
//					Ergebnis ist ein Grid
					if (ergStr3D.equals(parClassGrid) && clergStr3D.equals(parClassGrid)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((NumericGrid)(erg3D), (NumericGrid) (clerg3D))))/(double)((NumericGrid)(erg3D)).getNumberOfElements();
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);

//					Ergebnis ist ein 1DGrid		
					} else if (ergStr3D.equals(parClassGrid1D) && clergStr3D.equals(parClassCLGrid1D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid1D)(erg3D), (Grid1D) (clerg3D))))/(double)((NumericGrid)(erg3D)).getNumberOfElements();
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//					Ergebnis ist ein 2DGrid		
					} else if (ergStr3D.equals(parClassGrid2D) && clergStr3D.equals(parClassCLGrid2D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid2D)(erg3D), (Grid2D) (clerg3D))))/(double)((NumericGrid)(erg3D)).getNumberOfElements();
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
						
//					Ergebnis ist ein 3DGrid		
					} else if (ergStr3D.equals(parClassGrid3D) && clergStr3D.equals(parClassCLGrid3D)) {
						double value = NumericPointwiseOperators.sum(NumericPointwiseOperators.sqrcopy(NumericPointwiseOperators.subtractedBy((Grid3D)(erg3D), (Grid3D) (clerg3D))))/(double)((NumericGrid)(erg3D)).getNumberOfElements();
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Value = " + value, Math.abs(value) < smallValue);
							
//					Ergebnis ist ein float
					} else if (ergStr3D.equals(parClassFloat) && clergStr3D.equals(parClassFloat)) {
						double diff = Math.pow((float)erg3D - (float)clerg3D, 2);
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);						
//					Ergebnis ist ein double
					} else if (ergStr3D.equals(parClassDouble) && clergStr3D.equals(parClassDouble)) {
						double diff = Math.pow((Double)erg3D - (Double)clerg3D,2);
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. diff = " + diff, diff < smallValue);

					} else {
						Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED.  Result types do not agree or are not defined.", false);
					}
					
				} else {
					Assert.assertTrue("The test for the method " + formatMethodName(methods[i]) +" FAILED. Result types do not agree.", false);
				}
			} catch (IllegalAccessException  e) {
				// TODO Auto-generated catch block
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
	
	private String formatMethodName(Method method) {
		
		String name = method.getName();
		int parameterCount = method.getParameterTypes().length;
		Class<?>[] parameters = method.getParameterTypes();
		
		String parameterString = parameters[0].getSimpleName();
		
		for (int i = 1; i<parameterCount; i++) {
			parameterString += ", " + parameters[i].getSimpleName();
		}
		
		return name + "(" + parameterString + ")";
		
	}

}
