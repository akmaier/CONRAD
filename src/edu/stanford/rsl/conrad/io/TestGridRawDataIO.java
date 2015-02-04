/*
 * Copyright (C) 2010-2014  Andreas Maier, Kerstin Müller
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.io;

import ij.io.FileInfo;

import java.io.File;
import java.io.IOException;

import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class TestGridRawDataIO {

	@Test
	public void testGrid1DIO(){
		int length = 400;
		Grid1D grid = new Grid1D(length);
		for (int i=0;i<length;i++){
			grid.setAtIndex(i, (float) Math.random());
		}
		try {
			FileInfo fi = GridRawIOUtil.getDefaultFloat32LittleEndianFileInfo();
			fi.width = length;
			fi.height = 1;
			fi.nImages = 1;
			String sep = System.getProperty("file.separator");
			String filename = CONRAD.getUserHome()+sep+"test.raw";
			GridRawIOUtil.saveRawDataGrid(grid, fi, filename);
			Grid2D load = (Grid2D) GridRawIOUtil.loadFromRawData(fi, filename);
			NumericPointwiseIteratorND iter1 = new NumericPointwiseIteratorND(grid);
			NumericPointwiseIteratorND iter2 = new NumericPointwiseIteratorND(load);
			double sum = 0;
			while (iter1.hasNext()){
				sum += Math.abs(iter1.getNext()-iter2.getNext());
			}
			File file = new File(filename);
			file.delete();
			Assert.assertTrue(sum < CONRAD.FLOAT_EPSILON);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Test
	public void testGrid2DIO(){
		int length = 400;
		Grid2D grid = new Grid2D(length,length*2);
		NumericPointwiseIteratorND iter = new NumericPointwiseIteratorND(grid);
		while(iter.hasNext()){
			iter.setNext((float)Math.random()); 
		}
		try {
			FileInfo fi = GridRawIOUtil.getDefaultFloat32LittleEndianFileInfo();
			fi.width = length;
			fi.height = length*2;
			fi.nImages = 1;
			String sep = System.getProperty("file.separator");
			String filename = CONRAD.getUserHome()+sep+"test.raw";
			GridRawIOUtil.saveRawDataGrid(grid, fi, filename);
			Grid2D load = (Grid2D) GridRawIOUtil.loadFromRawData(fi, filename);
			NumericPointwiseIteratorND iter1 = new NumericPointwiseIteratorND(grid);
			NumericPointwiseIteratorND iter2 = new NumericPointwiseIteratorND(load);
			double sum = 0;
			while (iter1.hasNext()){
				sum += Math.abs(iter1.getNext()-iter2.getNext());
			}
			File file = new File(filename);
			file.delete();
			Assert.assertTrue(sum < CONRAD.FLOAT_EPSILON);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Test
	public void testGrid3DIO(){
		int length = 40;
		Grid3D grid = new Grid3D(length, length*2, length*3);
		NumericPointwiseIteratorND iter = new NumericPointwiseIteratorND(grid);
		while(iter.hasNext()){
			iter.setNext((float)Math.random()); 
		}
		try {
			FileInfo fi = GridRawIOUtil.getDefaultFloat32LittleEndianFileInfo();
			fi.width = length;
			fi.height = length*2;
			fi.nImages = length*3;
			String sep = System.getProperty("file.separator");
			String filename = CONRAD.getUserHome()+sep+"test.raw";
			GridRawIOUtil.saveRawDataGrid(grid, fi, filename);
			Grid3D load = (Grid3D) GridRawIOUtil.loadFromRawData(fi, filename);
			NumericPointwiseIteratorND iter1 = new NumericPointwiseIteratorND(grid);
			NumericPointwiseIteratorND iter2 = new NumericPointwiseIteratorND(load);
			double sum = 0;
			while (iter1.hasNext()){
				sum += Math.abs(iter1.getNext()-iter2.getNext());
			}
			File file = new File(filename);
			file.delete();
			Assert.assertTrue(sum < CONRAD.FLOAT_EPSILON);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
