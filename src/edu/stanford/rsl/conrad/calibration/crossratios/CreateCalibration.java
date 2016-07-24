package edu.stanford.rsl.conrad.calibration.crossratios;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;
import edu.stanford.rsl.conrad.calibration.GeometricCalibration;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class CreateCalibration {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		ArrayList<CalibrationBead> list = new ArrayList<CalibrationBead>();
		
		// read 2-D and 3-D positions
	
		SimpleMatrix matrix = GeometricCalibration.computePMatrix(list);
		
		// write matrix somewhere
		
	}

}
