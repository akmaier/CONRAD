import java.io.File;
import java.util.ArrayList;

import javax.swing.JFileChooser;

import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.ProjectionTableFileTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.DoublePrecisionPointUtil;
import edu.stanford.rsl.conrad.utils.StringFileFilter;
import ij.plugin.PlugIn;


public class Check_Trajectory_Consistency implements PlugIn {

	@Override
	public void run(String arg) {
		try {
			JFileChooser FC = new JFileChooser();
			if (arg != null){
				File dir = new File(arg);
				FC.setCurrentDirectory(dir);
			}
			FC.setVisible(true);
			FC.setFileFilter(new StringFileFilter(".txt"));
			FC.setMultiSelectionEnabled(true);
			int result = -1;
			result = FC.showOpenDialog(null);
			if (result == JFileChooser.CANCEL_OPTION) {
				throw new Exception("Cancelled");
			}
			File [] files = FC.getSelectedFiles();
			Trajectory [] trajectories = new Trajectory[files.length];
			
			int numProj = -1;
			for (int i = 0; i < files.length; i++){
				trajectories[i]= new ProjectionTableFileTrajectory(files[i].getAbsolutePath(), new Trajectory());
				if (numProj == -1) numProj = trajectories[i].getProjectionMatrices().length;
				else {
					if (numProj != trajectories[i].getProjectionMatrices().length) throw new RuntimeException("Number of matrices does not match!");
				}
			}
			// compute reference iso center
			PointND referenceCenter = trajectories[0].computeIsoCenter();
			// compute reference source positions centered around (0, 0, 0);
			ArrayList<PointND> refPositions = computeCenteredSourcePositions(trajectories[0], referenceCenter);
			ArrayList<PointND> refPoints = computePrincipalPoints(trajectories[0]);
			// estimate normalization rotation matrix
			SimpleMatrix referenceRotation = estimateRotationMatrix(refPositions);
			// apply rotation to all source postions
			applyRotation(refPositions, referenceRotation);
			
			System.out.println("Isocenter\tmean deviation [mm] \tcoordinates");
			printSourcePositions(files[0].getAbsolutePath(), referenceCenter, 0, refPositions, refPoints);
			for (int i = 1; i < files.length; i++){
				// estimate the iso center
				PointND otherIsoCenter = trajectories[i].computeIsoCenter();
				ArrayList<PointND> otherTrajectory = computeCenteredSourcePositions(trajectories[i], otherIsoCenter);
				SimpleMatrix otherRotation = estimateRotationMatrix(otherTrajectory);
				applyRotation(otherTrajectory, otherRotation);
				double deviation = computeMeanDeviation(refPositions, otherTrajectory);
				printSourcePositions(files[i].getAbsolutePath(), otherIsoCenter, deviation, otherTrajectory, computePrincipalPoints(trajectories[i]));
			}
			printSourcePositions(files[0].getAbsolutePath(), referenceCenter, 0, refPositions, refPoints);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public double computeMeanDeviation(ArrayList<PointND> pointsA, ArrayList<PointND> pointsB){
		double sum = 0;
		for (int i = 0; i < pointsA.size(); i++){
			sum += (pointsA.get(i).euclideanDistance(pointsB.get(i)));
		}
		return sum / pointsA.size();
	}
	
	/**
	 * Prints the important features of the trajectory
	 * @param isocenter the iso center 
	 * @param deviation the mean deviation
	 * @param points the actual source positions
	 * @param principalPoints 
	 */
	public void printSourcePositions(String file, PointND isocenter, double deviation, ArrayList<PointND> points, ArrayList<PointND> principalPoints){
		System.out.print(file +"\t"+isocenter + "\t" + deviation+"\t");
		System.out.print(DoublePrecisionPointUtil.getGeometricCenter(principalPoints) + "\t" + DoublePrecisionPointUtil.getStandardDeviation(principalPoints)+"\t");
		for (int i =0; i < points.size(); i++){
			System.out.print(points.get(i)+"\t");
		}
		System.out.println();
	}
	
	/**
	 * Method to generate all the principal points of the detectors in pixels.
	 * @param traj the trajectory
	 * @return the source positions centered around (0, 0, 0);
	 */
	public ArrayList<PointND> computePrincipalPoints(Trajectory traj){
		ArrayList<PointND> refPositions = new ArrayList<PointND>();
		for (int i = 0; i < traj.getProjectionMatrices().length; i++){
			SimpleVector vector = traj.getProjectionMatrices()[i].getPrincipalPoint();
			PointND point = new PointND(vector);
			refPositions.add(point);
		}
		return refPositions;
	}
	
	/**
	 * Method to move points around the iso center, i.e. a translation is applied.
	 * @param traj the trajectory
	 * @param isoCenter the iso center
	 * @return the source positions centered around (0, 0, 0);
	 */
	public ArrayList<PointND> computeCenteredSourcePositions(Trajectory traj, PointND isoCenter){
		ArrayList<PointND> refPositions = new ArrayList<PointND>();
		for (int i = 0; i < traj.getProjectionMatrices().length; i++){
			SimpleVector vector = traj.getProjectionMatrices()[i].computeCameraCenter();
			traj.getProjectionMatrices()[i].computeRayDirection(new SimpleVector(0,0));
			vector.add(isoCenter.getAbstractVector().negated());
			PointND point = new PointND(vector);
			refPositions.add(point);
		}
		return refPositions;
	}
	
	/**
	 * estimates the rotation that needs to be applied to move the first source position to the vector<br>
	 * ( normL2(firstSourcePosition), 0, 0) <br>
	 * Using this transformation the trajectory can be standardized. Coordinates will still me metric, but centered around (0, 0, 0)
	 * 
	 * @param centeredPoints
	 * @return the rotation matrix
	 */
	public SimpleMatrix estimateRotationMatrix (ArrayList<PointND> centeredPoints){
		SimpleVector other = centeredPoints.get(0).getAbstractVector();
		double len = other.normL2();
		SimpleVector e1 = new SimpleVector(len, 0, 0);
		return Rotations.getRotationMatrixFromAtoB(other, e1);
	}
	
	/**
	 * Method to apply a rotation matrix to a set of points.
	 * @param points
	 * @param rotation
	 */
	public void applyRotation(ArrayList<PointND> points, SimpleMatrix rotation){
		for (PointND point:points){
			SimpleVector newCoord = SimpleOperators.multiply(rotation, point.getAbstractVector());
			point.setCoordinates(newCoord);
		}
	}
	
	public static void main(String [] args){
		new Check_Trajectory_Consistency().run("C:\\Users\\z002xsyz\\Documents\\4yu\\5s\\forward");
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
