/*
 * Copyright (C) 2010-2014 - Andreas Maier, Marco Bögel 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Rotations.BasicAxis;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * This code was edited or generated using CloudGarden's Jigloo SWT/Swing GUI
 * Builder, which is free for non-commercial use. If Jigloo is being used
 * commercially (ie, by a corporation, company or business for any purpose
 * whatever) then you should purchase a license for each developer using Jigloo.
 * Please visit www.cloudgarden.com for details. Use of Jigloo implies
 * acceptance of these licensing terms. A COMMERCIAL LICENSE HAS NOT BEEN
 * PURCHASED FOR THIS MACHINE, SO JIGLOO OR THIS CODE CANNOT BE USED LEGALLY FOR
 * ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
public class TrajectoryEditor extends JFrame {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5844637548975765171L;
	private JLabel jLabelLeft;
	private JLabel jLabelMatrixRight;
	private JLabel JLabelPMatrix;
	private JTextField jA11;
	private JButton jApplyToPMatrices;
	private JButton jDeleteMatrix;
	private JButton jDefineRotation;
	private JButton jDefineNoise;
	private JButton jFixPrimaryAngles;
	private JTextField jB33;
	private JTextField jB32;
	private JTextField jB31;
	private JTextField jB23;
	private JTextField jB22;
	private JTextField jB21;
	private JTextField jB13;
	private JTextField jB12;
	private JTextField jB11;
	private JTextField jA31;
	private JTextField jA32;
	private JTextField jA33;
	private JTextField jA34;
	private JTextField jA41;
	private JTextField jA42;
	private JTextField jA43;
	private JTextField jA44;
	private JTextField jA21;
	private JTextField jA22;
	private JTextField jA23;
	private JTextField jA24;
	private JTextField jA14;
	private JTextField jA13;
	private JTextField jA12;
	private JLabel jLabelP;

	private void initGUI() {
		try {
			{
				this.setBackground(Color.WHITE);
				GridBagLayout thisLayout = new GridBagLayout();
				thisLayout.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 };
				thisLayout.rowHeights = new int[] { 36, 12, 61, 65, 240 };
				thisLayout.columnWeights = new double[] { 0.0, 0.0, 0.1 };
				thisLayout.columnWidths = new int[] { 167, 201, 7 };
				getContentPane().setLayout(thisLayout);
				{
					jLabelLeft = new JLabel();
					getContentPane().add(jLabelLeft, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
							GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jLabelLeft.setText("Matrix Left");
				}
				{
					jLabelMatrixRight = new JLabel();
					getContentPane().add(jLabelMatrixRight, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0,
							GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jLabelMatrixRight.setText("Matrix Right");
				}
				{
					JLabelPMatrix = new JLabel();
					getContentPane().add(JLabelPMatrix, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
							GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					JLabelPMatrix.setText("Projection Matrix");
				}
				{
					jLabelP = new JLabel();
					getContentPane().add(jLabelP, new GridBagConstraints(1, 2, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, -10, 0), 0, 0));
					jLabelP.setText("x P x");
				}
				{
					jA11 = new JTextField();
					getContentPane().add(jA11, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, new Insets(5, 5, 0, 0), 0, 0));
					jA11.setText("1");
				}
				{
					jA12 = new JTextField();
					getContentPane().add(jA12, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, 40), 0, 0));
					jA12.setText("0");
				}
				{
					jA13 = new JTextField();
					getContentPane().add(jA13, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, -40), 0, 0));
					jA13.setText("0");
				}
				{
					jA14 = new JTextField();
					getContentPane().add(jA14, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHEAST, GridBagConstraints.NONE, new Insets(5, 0, 0, 5), 0, 0));
					jA14.setText("0");
				}
				{
					jA24 = new JTextField();
					getContentPane().add(jA24, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHEAST, GridBagConstraints.NONE, new Insets(0, 0, 5, 5), 0, 0));
					jA24.setText("0");
				}
				{
					jA23 = new JTextField();
					getContentPane().add(jA23, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, 5, -40), 0, 0));
					jA23.setText("0");
				}
				{
					jA22 = new JTextField();
					getContentPane().add(jA22, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, 5, 40), 0, 0));
					jA22.setText("1");
				}
				{
					jA21 = new JTextField();
					getContentPane().add(jA21, new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHWEST, GridBagConstraints.NONE, new Insets(0, 5, 5, 0), 0, 0));
					jA21.setText("0");
				}
				{
					jA44 = new JTextField();
					getContentPane().add(jA44, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHEAST, GridBagConstraints.NONE, new Insets(0, 0, 5, 5), 0, 0));
					jA44.setText("1");
				}
				{
					jA43 = new JTextField();
					getContentPane().add(jA43, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, 5, -40), 0, 0));
					jA43.setText("0");
				}
				{
					jA42 = new JTextField();
					getContentPane().add(jA42, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, 5, 40), 0, 0));
					jA42.setText("0");
				}
				{
					jA41 = new JTextField();
					getContentPane().add(jA41, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHWEST, GridBagConstraints.NONE, new Insets(0, 5, 5, 0), 0, 0));
					jA41.setText("0");
				}
				{
					jA34 = new JTextField();
					getContentPane().add(jA34, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHEAST, GridBagConstraints.NONE, new Insets(5, 0, 0, 5), 0, 0));
					jA34.setText("0");
				}
				{
					jA33 = new JTextField();
					getContentPane().add(jA33, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, -40), 0, 0));
					jA33.setText("1");
				}
				{
					jA32 = new JTextField();
					getContentPane().add(jA32, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, 40), 0, 0));
					jA32.setText("0");
				}
				{
					jA31 = new JTextField();
					getContentPane().add(jA31, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, new Insets(5, 5, 0, 0), 0, 0));
					jA31.setText("0");
				}
				{
					jB11 = new JTextField();
					getContentPane().add(jB11, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, new Insets(5, 5, 0, 0), 0, 0));
					jB11.setText("1");
				}
				{
					jB12 = new JTextField();
					getContentPane().add(jB12, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, 0), 0, 0));
					jB12.setText("0");
				}
				{
					jB13 = new JTextField();
					getContentPane().add(jB13, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHEAST, GridBagConstraints.NONE, new Insets(5, 0, 0, 5), 0, 0));
					jB13.setText("0");
				}
				{
					jB21 = new JTextField();
					getContentPane().add(jB21, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHWEST, GridBagConstraints.NONE, new Insets(0, 5, 5, 0), 0, 0));
					jB21.setText("0");
				}
				{
					jB22 = new JTextField();
					getContentPane().add(jB22, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTH,
							GridBagConstraints.NONE, new Insets(0, 0, 5, 0), 0, 0));
					jB22.setText("1");
				}
				{
					jB23 = new JTextField();
					getContentPane().add(jB23, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
							GridBagConstraints.SOUTHEAST, GridBagConstraints.NONE, new Insets(0, 0, 5, 5), 0, 0));
					jB23.setText("0");
				}
				{
					jB31 = new JTextField();
					getContentPane().add(jB31, new GridBagConstraints(0, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, new Insets(5, 5, 0, 0), 0, 0));
					jB31.setText("0");
				}
				{
					jB32 = new JTextField();
					getContentPane().add(jB32, new GridBagConstraints(0, 3, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH,
							GridBagConstraints.NONE, new Insets(5, 0, 0, 0), 0, 0));
					jB32.setText("0");
				}
				{
					jB33 = new JTextField();
					getContentPane().add(jB33, new GridBagConstraints(0, 3, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTHEAST, GridBagConstraints.NONE, new Insets(5, 0, 0, 5), 0, 0));
					jB33.setText("1");
				}
				{
					int insetY = 20;

					jApplyToPMatrices = new JButton();
					jApplyToPMatrices.addActionListener(new ActionListener() {

						public void actionPerformed(ActionEvent e) {
							if (e.getSource() != null) {
								if (e.getSource().equals(jApplyToPMatrices)) {
									applyToPMatrices();
								}
							}
						}
					});
					getContentPane().add(jApplyToPMatrices, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(insetY, 0, 0, 0), 0, 0));
					jApplyToPMatrices.setText("Apply to all Projection Matrices");
					jDeleteMatrix = new JButton();
					jDeleteMatrix.addActionListener(new ActionListener() {

						public void actionPerformed(ActionEvent e) {
							if (e.getSource() != null) {
								if (e.getSource().equals(jDeleteMatrix)) {
									try {
										deleteSome();
									} catch (Exception e1) {
										// TODO Auto-generated catch block
										e1.printStackTrace();
									}
								}
							}
						}
					});

					insetY += 40;

					getContentPane().add(jDeleteMatrix, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(insetY, 0, 0, 0), 0, 0));
					jDeleteMatrix.setText("Delete Some Projection Matrices");

					jDefineRotation = new JButton();
					jDefineRotation.addActionListener(new ActionListener() {

						public void actionPerformed(ActionEvent e) {
							if (e.getSource() != null) {
								if (e.getSource().equals(jDefineRotation)) {
									try {
										defineRotation();
									} catch (Exception e1) {
										// TODO Auto-generated catch block
										e1.printStackTrace();
									}
								}
							}
						}
					});

					insetY += 40;

					getContentPane().add(jDefineRotation, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(insetY, 0, 0, 0), 0, 0));
					jDefineRotation.setText("Define A Source Rotation");

					jDefineNoise = new JButton();
					jDefineNoise.addActionListener(new ActionListener() {

						public void actionPerformed(ActionEvent e) {
							if (e.getSource() != null) {
								if (e.getSource().equals(jDefineNoise)) {
									try {
										defineNoise();
									} catch (Exception e1) {
										// TODO Auto-generated catch block
										e1.printStackTrace();
									}
								}
							}
						}
					});

					insetY += 40;

					getContentPane().add(jDefineNoise, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(insetY, 0, 0, 0), 0, 0));
					jDefineNoise.setText("Define Trajectory Noise");

					jFixPrimaryAngles = new JButton();
					jFixPrimaryAngles.addActionListener(new ActionListener() {

						public void actionPerformed(ActionEvent e) {
							if (e.getSource() != null) {
								if (e.getSource().equals(jFixPrimaryAngles)) {
									try {
										fixPriamries();
									} catch (Exception e1) {
										// TODO Auto-generated catch block
										e1.printStackTrace();
									}
								}
							}
						}
					});
					insetY += 40;

					getContentPane().add(jFixPrimaryAngles, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0,
							GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(insetY, 0, 0, 0), 0, 0));
					jFixPrimaryAngles.setText("Fix Primary Angles");
				}
			}
			{
				this.setSize(512, 375);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void fixPriamries() throws Exception {
		double maxAngle = UserUtil.queryDouble("Enter Maximal Angular Range:", 200);
		int numProj = Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices();
		double angleStep = maxAngle / (numProj - 1);
		double[] newPrimaries = new double[numProj];
		double[] secondaries = null;
		if (Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles() != null)
			secondaries = new double[numProj];
		for (int i = 0; i < numProj; i++) {
			if (Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles() != null) {
				secondaries[i] = Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles()[i];
			}
			newPrimaries[i] = angleStep * i;
		}
		Configuration.getGlobalConfiguration().getGeometry().setPrimaryAngleArray(newPrimaries);
		Configuration.getGlobalConfiguration().getGeometry().setSecondaryAngleArray(secondaries);

	}

	private void deleteSome() throws Exception {
		int maxPrimaryAngle = UserUtil.queryInt("Enter Maximum Primary Angle:", 180);
		int lastValid = 0;
		double[] primaries = Configuration.getGlobalConfiguration().getGeometry().getPrimaryAngles();
		double min = DoubleArrayUtil.minOfArray(primaries);
		while (primaries[lastValid] - min < maxPrimaryAngle) {
			if (lastValid < primaries.length - 1)
				lastValid++;
			else {
				CONRAD.log("Angular range is already less or equal. Nothing to be done.");
				return;
			}
		}
		double[] newPrimaries = new double[lastValid];
		Projection[] newMatrices = new Projection[lastValid];
		double[] secondaries = null;
		if (Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles() != null)
			secondaries = new double[lastValid];
		for (int i = 0; i < lastValid; i++) {
			if (Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles() != null) {
				secondaries[i] = Configuration.getGlobalConfiguration().getGeometry().getSecondaryAngles()[i];
			}
			newPrimaries[i] = primaries[i];
			newMatrices[i] = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(i);
		}
		Configuration.getGlobalConfiguration().getGeometry().setProjectionMatrices(newMatrices);
		Configuration.getGlobalConfiguration().getGeometry().setPrimaryAngleArray(newPrimaries);
		Configuration.getGlobalConfiguration().getGeometry().setSecondaryAngleArray(secondaries);
		Configuration.getGlobalConfiguration().getGeometry().setNumProjectionMatrices(lastValid);
		CONRAD.log("Reduced to " + lastValid + " matrices");
	}

	private void applyToPMatrices() {
		double[][] b = new double[3][3];
		b[0][0] = Double.valueOf(jB11.getText());
		b[0][1] = Double.valueOf(jB12.getText());
		b[0][2] = Double.valueOf(jB13.getText());
		b[1][0] = Double.valueOf(jB21.getText());
		b[1][1] = Double.valueOf(jB22.getText());
		b[1][2] = Double.valueOf(jB23.getText());
		b[2][0] = Double.valueOf(jB31.getText());
		b[2][1] = Double.valueOf(jB32.getText());
		b[2][2] = Double.valueOf(jB33.getText());

		double[][] a = new double[4][4];
		a[0][0] = Double.valueOf(jA11.getText());
		a[0][1] = Double.valueOf(jA12.getText());
		a[0][2] = Double.valueOf(jA13.getText());
		a[0][3] = Double.valueOf(jA14.getText());
		a[1][0] = Double.valueOf(jA21.getText());
		a[1][1] = Double.valueOf(jA22.getText());
		a[1][2] = Double.valueOf(jA23.getText());
		a[1][3] = Double.valueOf(jA24.getText());
		a[2][0] = Double.valueOf(jA31.getText());
		a[2][1] = Double.valueOf(jA32.getText());
		a[2][2] = Double.valueOf(jA33.getText());
		a[2][3] = Double.valueOf(jA34.getText());
		a[3][0] = Double.valueOf(jA41.getText());
		a[3][1] = Double.valueOf(jA42.getText());
		a[3][2] = Double.valueOf(jA43.getText());
		a[3][3] = Double.valueOf(jA44.getText());

		SimpleMatrix B = new SimpleMatrix(b);
		SimpleMatrix A = new SimpleMatrix(a);

		Projection[] pMats = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices();
		for (int i = 0; i < pMats.length; i++) {
			SimpleMatrix tmp1 = SimpleOperators.multiplyMatrixProd(B, pMats[i].computeP());
			SimpleMatrix tmp2 = SimpleOperators.multiplyMatrixProd(tmp1, A);
			pMats[i].setPMatrixSerialization(tmp2.toString());
		}
		Configuration.getGlobalConfiguration().getGeometry().setProjectionMatrices(pMats);
		System.out.println("Done");

	}

	public static BasicAxis queryAxis(String message, String messageTitle, BasicAxis[] axes) throws Exception {
		return (BasicAxis) UserUtil.chooseObject(message, messageTitle, axes, axes[0]);
	}

	private void defineRotation() {

		try {
			BasicAxis axis = queryAxis("Define Rotation Axis", "Rotation Axis",
					new BasicAxis[] { BasicAxis.X_AXIS, BasicAxis.Y_AXIS, BasicAxis.Z_AXIS });
			double angle = Math.PI * UserUtil.queryDouble("Rotation angle (degree)", 0.0) / 180.0;
			SimpleMatrix rotation = Rotations.createBasicRotationMatrix(axis, angle);
			System.out.println(rotation);

			jA11.setText("" + rotation.getElement(0, 0));
			jA12.setText("" + rotation.getElement(0, 1));
			jA13.setText("" + rotation.getElement(0, 2));
			jA21.setText("" + rotation.getElement(1, 0));
			jA22.setText("" + rotation.getElement(1, 1));
			jA23.setText("" + rotation.getElement(1, 2));
			jA31.setText("" + rotation.getElement(2, 0));
			jA32.setText("" + rotation.getElement(2, 1));
			jA33.setText("" + rotation.getElement(2, 2));

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static double queryDistance(String message, String messageTitle, String[] distances) throws Exception {
		return Double.parseDouble((String) UserUtil.chooseObject(message, messageTitle, distances, distances[0]));
	}

	private void defineNoise() {
		try {
			double d = queryDistance("Choose distance", "Distance", new String[] { "1.0", "100.0" });
			double maxAngle = Math.atan(
					d / (2.0 * Configuration.getGlobalConfiguration().getGeometry().getSourceToDetectorDistance()));
			Projection[] pMats = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices();
			System.out.println(Configuration.getGlobalConfiguration().getGeometry().getSourceToDetectorDistance());
			System.out.println(maxAngle);

			for (int i = 0; i < pMats.length; i++) {

				double alpha = 2.0 * maxAngle * Math.random() - maxAngle;
				double beta = 2.0 * maxAngle * Math.random() - maxAngle;
				// Math.atan(2.0 * d / 2.0 * Math.random() - d / 2.0)

				SimpleMatrix rotation1 = Rotations.createBasicRotationMatrix(BasicAxis.X_AXIS, alpha);
				SimpleMatrix rotation2 = Rotations.createBasicRotationMatrix(BasicAxis.Y_AXIS, beta);

				SimpleMatrix rotation = SimpleOperators.multiplyMatrixProd(rotation1, rotation2);

				SimpleVector translation = new SimpleVector(3);

				translation.setElementValue(0, 2.0 * d / 2.0 * Math.random() - d / 2.0);
				translation.setElementValue(1, 2.0 * d / 2.0 * Math.random() - d / 2.0);
				translation.setElementValue(2, 2.0 * d / 2.0 * Math.random() - d / 2.0);

				jA11.setText("" + rotation.getElement(0, 0));
				jA12.setText("" + rotation.getElement(0, 1));
				jA13.setText("" + rotation.getElement(0, 2));
				jA21.setText("" + rotation.getElement(1, 0));
				jA22.setText("" + rotation.getElement(1, 1));
				jA23.setText("" + rotation.getElement(1, 2));
				jA31.setText("" + rotation.getElement(2, 0));
				jA32.setText("" + rotation.getElement(2, 1));
				jA33.setText("" + rotation.getElement(2, 2));
				jA14.setText("" + translation.getElement(0));
				jA24.setText("" + translation.getElement(1));
				jA34.setText("" + translation.getElement(2));

				double[][] a = new double[4][4];
				a[0][0] = Double.valueOf(jA11.getText());
				a[0][1] = Double.valueOf(jA12.getText());
				a[0][2] = Double.valueOf(jA13.getText());
				a[0][3] = Double.valueOf(jA14.getText());
				a[1][0] = Double.valueOf(jA21.getText());
				a[1][1] = Double.valueOf(jA22.getText());
				a[1][2] = Double.valueOf(jA23.getText());
				a[1][3] = Double.valueOf(jA24.getText());
				a[2][0] = Double.valueOf(jA31.getText());
				a[2][1] = Double.valueOf(jA32.getText());
				a[2][2] = Double.valueOf(jA33.getText());
				a[2][3] = Double.valueOf(jA34.getText());
				a[3][0] = Double.valueOf(jA41.getText());
				a[3][1] = Double.valueOf(jA42.getText());
				a[3][2] = Double.valueOf(jA43.getText());
				a[3][3] = Double.valueOf(jA44.getText());

				SimpleMatrix A = new SimpleMatrix(a);

				SimpleMatrix tmp2 = SimpleOperators.multiplyMatrixProd(pMats[i].computeP(), A);
				pMats[i].setPMatrixSerialization(tmp2.toString());
			}
			Configuration.getGlobalConfiguration().getGeometry().setProjectionMatrices(pMats);
			System.out.println("Done");

		} catch (Exception e) {
			System.out.println("catching fire...");
			e.printStackTrace();
		}
	}

	public TrajectoryEditor() {
		initGUI();
	}

}
