package edu.stanford.rsl.conrad.calibration.crossratios;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;
import edu.stanford.rsl.conrad.calibration.GeometricCalibration;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class CreateCalibration {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		ArrayList<CalibrationBead> list = new ArrayList<CalibrationBead>();
		
		// read 2-D and 3-D positions
		
		try {
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(new FileReader(args[0]));
			String line = reader.readLine();
			while (line != null){
				String [] numbers = line.split(";");
				double u = Double.parseDouble(numbers[0]);
				double v = Double.parseDouble(numbers[1]);
				double x = Double.parseDouble(numbers[2]);
				double y = Double.parseDouble(numbers[3]);
				double z = Double.parseDouble(numbers[4]);
				list.add(new CalibrationBead(u, v, x, y, z));
				line = reader.readLine();
			};
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		// compute matrix
		SimpleMatrix matrix = GeometricCalibration.computePMatrix(list);
		
		// write matrix somewhere
		System.out.println(matrix);
	}

}
