package edu.stanford.rsl.apps.gui;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.ButtonGroup;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.ConfigFileBasedTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.MultiSweepTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.GUIUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * This code was edited or generated using CloudGarden's Jigloo
 * SWT/Swing GUI Builder, which is free for non-commercial
 * use. If Jigloo is being used commercially (ie, by a corporation,
 * company or business for any purpose whatever) then you
 * should purchase a license for each developer using Jigloo.
 * Please visit www.cloudgarden.com for details.
 * Use of Jigloo implies acceptance of these licensing terms.
 * A COMMERCIAL LICENSE HAS NOT BEEN PURCHASED FOR
 * THIS MACHINE, SO JIGLOO OR THIS CODE CANNOT BE USED
 * LEGALLY FOR ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
public class ConfigurationFrame extends JFrame implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 979874928758716096L;
	private JTabbedPane jTabbedPane;
	private JButton jSaveButton;
	private boolean exited = false;
	Configuration config;
	private UpdateableGUI parentFrame;
	private RegistryEditor regEditor;
	private TrajectoryEditor trajEdit;
	JButton jCenterVolume = new JButton("Center volume");
	JButton jDoubleVolumeResolution = new JButton("Resolution *= 2");
	JButton jHalfVolumeResolution = new JButton("Resolution /= 2");
	JButton jDoubleDetectorResolution = new JButton("Resolution *= 2");
	JButton jHalfDetectorResolution = new JButton("Resolution /= 2");

	public ConfigurationFrame(){
		config = Configuration.getGlobalConfiguration();
		regEditor = new RegistryEditor(config);
		SwingUtilities.invokeLater(new Runnable() {
			public void run(){
				regEditor.initGUI();
			};
		});
		initGUI();
		updateValues();
	}


	public boolean isExited() {
		return exited;
	}

	public void exit(){
		exited = true;
		setVisible(false);
		if (parentFrame != null) parentFrame.updateGUI();
		System.out.println("exiting config frame");
	}

	String [] camAxes = new String [] {
			"points in the direction of detector motion",
			"points against the direction of detector motion",
			"points in rotation axis direction",
	"points against rotation axis direction"};

	private CameraAxisDirection getDirection(String str){
		CameraAxisDirection dir = CameraAxisDirection.DETECTORMOTION_PLUS;
		if (str != null){
			if (str.equals("points in the direction of detector motion")){ 
				dir = CameraAxisDirection.DETECTORMOTION_PLUS; 
			} else if (str.equals("points against the direction of detector motion")){ 
				dir = CameraAxisDirection.DETECTORMOTION_MINUS; 
			} else if (str.equals("points in rotation axis direction")){
				dir = CameraAxisDirection.ROTATIONAXIS_PLUS;
			} else if (str.equals("points against rotation axis direction")){
				dir = CameraAxisDirection.ROTATIONAXIS_MINUS;
			}
		}
		return dir;
	}

	private int getDirectionIndex(CameraAxisDirection dir){
		int index = -1;
		if (dir == CameraAxisDirection.DETECTORMOTION_PLUS) { 
			index = 0; 
		} else if (dir == CameraAxisDirection.DETECTORMOTION_MINUS){
			index = 1; 
		} else if (dir == CameraAxisDirection.ROTATIONAXIS_PLUS){
			index = 2;
		} else if (dir == CameraAxisDirection.ROTATIONAXIS_MINUS){
			index = 3;
		}
		return index;
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() != null){
			if (e.getSource().equals(jSaveButton)) {
				readValuesToConfig();
				if (config.getImportFromDicomAutomatically()) {
					try {
						Trajectory load = Configuration.loadGeometrySource(config);
						if (load != null){
							config.setGeometry(load);
						}
					} catch (Exception e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
				}
				Configuration.setGlobalConfiguration(config);
				Configuration.saveConfiguration();
				exited = true;
				setVisible(false);
				if (parentFrame != null) parentFrame.updateGUI();
			}
			if (e.getSource().equals(trajectoryEditorButton)) {
				startTrajectoryEditor();
			}
			if (e.getSource().equals(jParseGeometry)) {
				JTextField field = this.projectionTableFileField;
				Trajectory geom = ConfigFileBasedTrajectory.openAsGeometrySource(field.getText(), config.getGeometry());
				if (geom != null){
					Configuration config = Configuration.getGlobalConfiguration();
					config.setProjectionTableFileName(field.getText());
					config.setGeometry(geom);
				}
				updateValues();
			}
			if (e.getSource().equals(optionList)){
				//System.out.println("Event" + e);
				trajectoryPanel.remove(trajectoryFromParameters);
				trajectoryPanel.remove(trajectoryFromFile);
				if (optionList.getSelectedIndex() == 0){
					trajectoryPanel.add(trajectoryFromFile, createConstraints(0, 2, 5, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, 0, 0, 0, 0));		
				} else {
					trajectoryPanel.add(trajectoryFromParameters, createConstraints(0, 2, 5, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, 0, 0, 0, 0));
				}
				this.repaint();
			}
			if (e.getSource().equals(jGenerateDetector)) {
				XRayDetector detector;
				try {
					detector = (XRayDetector) UserUtil.queryObject("Select Detector:", "Detector Selection", XRayDetector.class);
					detector.configure();
					Configuration.getGlobalConfiguration().setDetector(detector);
					jDetectorLabel.setText("Current Detector: " + config.getDetector());
				} catch (Exception e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
			if (e.getSource().equals(jGenerateGeometry)) {
				readValuesToConfig();
				int numProjectionMatrices = config.getGeometry().getProjectionStackSize();
				double sourceToAxisDistance = Double.parseDouble(sourceToPatientField.getText());
				double averageAngularIncrement = config.getGeometry().getAverageAngularIncrement();
				double detectorOffsetU = config.getGeometry().getDetectorOffsetU();
				double detectorOffsetV = config.getGeometry().getDetectorOffsetV();
				CameraAxisDirection uDirection = config.getGeometry().getDetectorUDirection();
				CameraAxisDirection vDirection = config.getGeometry().getDetectorVDirection();
				SimpleVector rotationAxis = config.getGeometry().getRotationAxis();
				Trajectory geom = new CircularTrajectory(config.getGeometry());
				geom.setSecondaryAngleArray(null);
				((CircularTrajectory)geom).setTrajectory(numProjectionMatrices, sourceToAxisDistance, averageAngularIncrement, detectorOffsetU, detectorOffsetV, uDirection, vDirection, rotationAxis);
				if (geom != null){
					Configuration config = Configuration.getGlobalConfiguration();
					if (config.getNumSweeps() > 1){
						MultiSweepTrajectory multi = new MultiSweepTrajectory(geom);
						multi.extrapolateProjectionGeometry();
						geom = new Trajectory(multi);
					}
					config.setGeometry(geom);
				}
			}
			if (e.getSource().equals(jCenterVolume)) {
				centerVolume();
			}
			if (e.getSource().equals(jDoubleVolumeResolution)) {
				changeVolumeResolution(2.0);
			}
			if (e.getSource().equals(jHalfVolumeResolution)) {
				changeVolumeResolution(0.5);	
			}
			if (e.getSource().equals(jDoubleDetectorResolution)) {
				changeDetectorResolution(2.0);
			}
			if (e.getSource().equals(jHalfDetectorResolution)) {
				changeDetectorResolution(0.5);	
			}
		}
	}

	private void startTrajectoryEditor() {
		if(trajEdit == null) {
			trajEdit = new TrajectoryEditor();
			trajEdit.setVisible(true);
			trajEdit.setLocation(CONRAD.getWindowTopCorner());
		}
	}


	/**
	 * creates a 5 x 4 GridBagLayout in ConfigurationFrame SubPanel-Style
	 * @return the sub panel layout
	 */
	static LayoutManager createSubPaneLayout(){
		GridBagLayout thisLayout = new GridBagLayout();
		thisLayout.rowWeights = new double[] {0.0, 0.1, 0.1, 0.0};
		thisLayout.rowHeights = new int[] {15, 120, 120, 15};
		thisLayout.columnWeights = new double[] {0.1, 0.1, 0.1, 0.1, 0.1};
		thisLayout.columnWidths = new int[] {15, 225, 15, 225, 15};
		return thisLayout;	
	}


	private void positionNumericLabelTextFieldPair(JPanel panel, String labelText, JTextField field, int x, int y, int topOffset){
		JLabel label = new JLabel(labelText);
		panel.add(label, createConstraints(x, y, 1, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		panel.add(field, createConstraints(x, y, 1, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset, 210, 0, 0));
		field.setHorizontalAlignment(JTextField.RIGHT);
	}

	private void positionLabelTextFieldPair(JPanel panel, String labelText, JTextField field, int x, int y, int width, int topOffset){
		JLabel label = new JLabel(labelText);
		panel.add(label, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		panel.add(field, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset, 210, 0, 0));
	}

	private void positionLabelTextFieldButton(JPanel panel, String labelText, JTextField field, int x, int y, int width, int topOffset, JButton button){
		JLabel label = new JLabel(labelText);
		panel.add(label, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		panel.add(field, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset+30, 0, 0, 0));
		panel.add(button, createConstraints(x+width-1, y, 1, 1, GridBagConstraints.NORTHEAST, GridBagConstraints.HORIZONTAL, topOffset+60, 100, 0, 0));
	}

	JTextField xDimensionField = new JTextField();
	JTextField yDimensionField = new JTextField();
	JTextField zDimensionField = new JTextField();
	JTextField xOriginVoxel = new JTextField();
	JTextField yOriginVoxel = new JTextField();
	JTextField zOriginVoxel = new JTextField();

	JTextField xOriginWorld = new JTextField();
	JTextField yOriginWorld = new JTextField();
	JTextField zOriginWorld = new JTextField();

	JTextField xSpacingField = new JTextField();
	JTextField ySpacingField = new JTextField();
	JTextField zSpacingField = new JTextField();
	JTextField maxVOIPathField = new JTextField();

	private void updateValues(){
		// volume
		Trajectory geometry = config.getGeometry(); 
		if (geometry != null) {
			xDimensionField.setText("" + geometry.getReconDimensionX());
			yDimensionField.setText("" + geometry.getReconDimensionY());
			zDimensionField.setText("" + geometry.getReconDimensionZ());
			xSpacingField.setText("" + geometry.getVoxelSpacingX());
			ySpacingField.setText("" + geometry.getVoxelSpacingY());
			zSpacingField.setText("" + geometry.getVoxelSpacingZ());
			xOriginWorld.setText("" + geometry.getOriginX());
			yOriginWorld.setText("" + geometry.getOriginY());
			zOriginWorld.setText("" + geometry.getOriginZ());
			xOriginVoxel.setText("" + geometry.getOriginInPixelsX());
			yOriginVoxel.setText("" + geometry.getOriginInPixelsY());
			zOriginVoxel.setText("" + geometry.getOriginInPixelsZ());


			// geometry
			detectorWidthField.setText("" + geometry.getDetectorWidth());
			detectorHeightField.setText("" + geometry.getDetectorHeight());
			pixelDimensionXField.setText("" + geometry.getPixelDimensionX());
			pixelDimensionYField.setText("" + geometry.getPixelDimensionY());
			sourceToDetectorField.setText("" + geometry.getSourceToDetectorDistance());
			sourceToPatientField.setText("" + geometry.getSourceToAxisDistance());
			projectionStackSizeField.setText("" + geometry.getProjectionStackSize());

			// new parameters for trajectory definition
			averageAngularIncrement.setText("" + geometry.getAverageAngularIncrement());
			detectorOffsetU.setText("" + geometry.getDetectorOffsetU());
			detectorOffsetV.setText("" + geometry.getDetectorOffsetV());
			SimpleVector rotAxis = geometry.getRotationAxis();
			if (rotAxis==null) rotAxis = new SimpleVector(0,0,1);
			rotationAxis.setText("" + rotAxis.toString());
			DetectorUDirection.setSelectedIndex(getDirectionIndex(geometry.getDetectorUDirection()));
			DetectorVDirection.setSelectedIndex(getDirectionIndex(geometry.getDetectorVDirection()));

		}
		numSweepsField.setText("" + config.getNumSweeps());

		maxVOIPathField.setText(config.getVolumeOfInterestFileName());
		projectionTableFileField.setText(config.getProjectionTableFileName());
		// other
		recentFileOneField.setText(config.getRecentFileOne());
		recentFileTwoField.setText(config.getRecentFileTwo());
		importFromDicom.setSelected(config.getImportFromDicomAutomatically());
		extrapolateGeometry.setSelected(config.getUseExtrapolatedGeometry());
		houns.setSelected(config.getUseHounsfieldScaling());
		if (config.getCitationFormat() == Configuration.BIBTEX_CITATION_FORMAT){
			citationFormatGroup.setSelected(bibButton.getModel(), true);
		} else {
			citationFormatGroup.setSelected(medlineButton.getModel(), true);
		}
	}

	private String getTextSave(JTextField textField){
		String revan = null;
		if (! textField.getText().equals("")) {
			revan = textField.getText();
		}
		return revan;
	}

	public void setParentFrame(UpdateableGUI parentFrame) {
		this.parentFrame = parentFrame;
	}


	public UpdateableGUI getParentFrame() {
		return parentFrame;
	}


	private void readValuesToConfig() throws NumberFormatException {
		// volume
		config.getGeometry().setReconDimensionX((int) Double.parseDouble(xDimensionField.getText()));
		config.getGeometry().setReconDimensionY((int) Double.parseDouble(yDimensionField.getText()));
		config.getGeometry().setReconDimensionZ((int) Double.parseDouble(zDimensionField.getText()));
		config.getGeometry().setVoxelSpacingX(Double.parseDouble(xSpacingField.getText()));
		config.getGeometry().setVoxelSpacingY(Double.parseDouble(ySpacingField.getText()));
		config.getGeometry().setVoxelSpacingZ(Double.parseDouble(zSpacingField.getText()));
		config.getGeometry().setOriginInPixelsX(Double.parseDouble(xOriginVoxel.getText()));
		config.getGeometry().setOriginInPixelsY(Double.parseDouble(yOriginVoxel.getText()));
		config.getGeometry().setOriginInPixelsZ(Double.parseDouble(zOriginVoxel.getText()));
		config.setVolumeOfInterestFileName(getTextSave(maxVOIPathField));
		// geometry
		config.getGeometry().setDetectorWidth((int)Math.floor(Double.parseDouble(detectorWidthField.getText())));
		config.getGeometry().setDetectorHeight((int)Math.floor(Double.parseDouble(detectorHeightField.getText())));
		config.getGeometry().setPixelDimensionX(Double.parseDouble(pixelDimensionXField.getText()));
		config.getGeometry().setPixelDimensionY(Double.parseDouble(pixelDimensionYField.getText()));
		config.getGeometry().setSourceToDetectorDistance(Double.parseDouble(sourceToDetectorField.getText()));
		config.getGeometry().setSourceToAxisDistance(Double.parseDouble(sourceToPatientField.getText()));

		config.getGeometry().setProjectionStackSize(Integer.parseInt(projectionStackSizeField.getText()));
		config.setNumSweeps(Integer.parseInt(numSweepsField.getText()));

		config.setProjectionTableFileName(getTextSave(projectionTableFileField));
		// other
		config.setRecentFileOne(getTextSave(recentFileOneField));
		config.setRecentFileTwo(getTextSave(recentFileTwoField));
		config.setImportFromDicomAutomatically(importFromDicom.isSelected());
		config.setUseExtrapolatedGeometry(extrapolateGeometry.isSelected());
		config.setUseHounsfieldScaling(houns.isSelected());
		if (bibButton.isSelected()){
			config.setCitationFormat(Configuration.BIBTEX_CITATION_FORMAT);
		} else {
			config.setCitationFormat(Configuration.MEDLINE_CITATION_FORMAT);
		}

		// new parameters for trajectory definition
		config.getGeometry().setAverageAngularIncrement(Double.parseDouble(averageAngularIncrement.getText()));
		config.getGeometry().setDetectorOffsetU(Double.parseDouble(detectorOffsetU.getText()));
		config.getGeometry().setDetectorOffsetV(Double.parseDouble(detectorOffsetV.getText()));
		SimpleVector rotAxis = new SimpleVector();
		rotAxis.setVectorSerialization(rotationAxis.getText());
		config.getGeometry().setRotationAxis(rotAxis);
		config.getGeometry().setDetectorUDirection(getDirection((String)DetectorUDirection.getSelectedItem()));
		config.getGeometry().setDetectorVDirection(getDirection((String)DetectorVDirection.getSelectedItem()));

		regEditor.updateToConfiguration();
	}

	JLabel jDetectorLabel;
	
	private JPanel detectorPane(){
		JPanel volume = new JPanel();
		volume.setBackground(Color.white);
		volume.setLayout(createSubPaneLayout());
		positionNumericLabelTextFieldPair(volume, "Detector size x [pixels]", detectorWidthField, 1, 1, 0);
		positionNumericLabelTextFieldPair(volume, "Detector size y [pixels]", detectorHeightField, 1, 1, 30);
		positionNumericLabelTextFieldPair(volume, "Pixel spacing x [mm]", pixelDimensionXField, 3, 1, 0);
		positionNumericLabelTextFieldPair(volume, "Pixel spacing y [mm]", pixelDimensionYField, 3, 1, 30);
		volume.add(jDoubleDetectorResolution, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, -150, 0, 0));
		jDoubleDetectorResolution.addActionListener(this);
		volume.add(jHalfDetectorResolution, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, 150, 0, 0));
		jHalfDetectorResolution.addActionListener(this);
		volume.add(jGenerateDetector, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 90, 0, 0, 0));
		jDetectorLabel = new JLabel("Current Detector: " + config.getDetector());
		volume.add(jDetectorLabel, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 60, 0, 0, 0));
		jGenerateDetector.addActionListener(this);
		return volume;
	}
	
	private void changeDetectorResolution(double scale){
		testNumericFields(detectorWidthField, detectorHeightField, pixelDimensionXField, pixelDimensionYField);
		updateNumericFieldValue(pixelDimensionXField, 1.0/scale);
		updateNumericFieldValue(pixelDimensionYField, 1.0/scale);
		updateNumericFieldValue(detectorWidthField, scale);
		updateNumericFieldValue(detectorHeightField, scale);
	}

	private void updateNumericFieldValue(JTextField field, double scale){
		updateNumericFieldValue(field, scale, false);
	}
	
	private void updateNumericFieldValue(JTextField field, double scale, boolean roundToInteger){
		if (!roundToInteger)
			field.setText(new Double(Double.parseDouble(field.getText()) * scale).toString());
		else
			field.setText(new Integer((int)Math.floor(Double.parseDouble(field.getText()) * scale)).toString());
	}
	
	/**
	 * Will throw a parsing exception, if the fields contain invalid numbers.
	 * @param fields
	 */
	private void testNumericFields(JTextField ... fields){
		for (int i=0; i<fields.length; i++){
			Double.parseDouble(fields[i].getText());
		}
	}

	private void changeVolumeResolution(double scale){
		testNumericFields(xSpacingField, ySpacingField, zSpacingField, xDimensionField, yDimensionField, zDimensionField);
		updateNumericFieldValue(xSpacingField, 1.0/scale);
		updateNumericFieldValue(ySpacingField, 1.0/scale);
		updateNumericFieldValue(zSpacingField, 1.0/scale);
		updateNumericFieldValue(xDimensionField, scale);
		updateNumericFieldValue(yDimensionField, scale);
		updateNumericFieldValue(zDimensionField, scale);
	}

	private JPanel volumePane(){
		JPanel volume = new JPanel();
		volume.setBackground(Color.white);
		volume.setLayout(createSubPaneLayout());
		positionNumericLabelTextFieldPair(volume, "Size x [voxels]", xDimensionField, 1, 1, 0);
		positionNumericLabelTextFieldPair(volume, "Spacing x [mm]", xSpacingField, 3, 1, 0);
		positionNumericLabelTextFieldPair(volume, "Size y [voxels]", yDimensionField, 1, 1, 30);
		positionNumericLabelTextFieldPair(volume, "Spacing y [mm]", ySpacingField, 3, 1, 30);
		positionNumericLabelTextFieldPair(volume, "Size z [voxels]", zDimensionField, 1, 1, 60);
		positionNumericLabelTextFieldPair(volume, "Spacing z [mm]", zSpacingField, 3, 1, 60);
		positionNumericLabelTextFieldPair(volume, "Index of (0mm/0mm/0mm) x [voxels]", xOriginVoxel, 1, 1, 90);
		xOriginVoxel.setEnabled(false);
		positionNumericLabelTextFieldPair(volume, "Index of (0mm/0mm/0mm) y [voxels]", yOriginVoxel, 1, 1, 120);
		yOriginVoxel.setEnabled(false);
		positionNumericLabelTextFieldPair(volume, "Index of (0mm/0mm/0mm) z [voxels]", zOriginVoxel, 1, 1, 150);
		zOriginVoxel.setEnabled(false);
		positionNumericLabelTextFieldPair(volume, "World coord. of vol. origin (0/0/0) x [mm]", xOriginWorld, 3, 1, 90);
		positionNumericLabelTextFieldPair(volume, "World coord. of vol. origin (0/0/0) y [mm]", yOriginWorld, 3, 1, 120);
		positionNumericLabelTextFieldPair(volume, "World coord. of vol. origin (0/0/0) z [mm]", zOriginWorld, 3, 1, 150);
		xSpacingField.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(xOriginWorld, xOriginVoxel, xSpacingField));
		xOriginWorld.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(xOriginWorld, xOriginVoxel, xSpacingField));
		ySpacingField.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(yOriginWorld, yOriginVoxel, ySpacingField));
		yOriginWorld.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(yOriginWorld, yOriginVoxel, ySpacingField));
		zSpacingField.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(zOriginWorld, zOriginVoxel, zSpacingField));
		zOriginWorld.getDocument().addDocumentListener(new VoxelWorldFieldUpdater(zOriginWorld, zOriginVoxel, zSpacingField));
		volume.add(jCenterVolume, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, 0, 0, 0));
		jCenterVolume.addActionListener(this);
		volume.add(jDoubleVolumeResolution, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, -300, 0, 0));
		jDoubleVolumeResolution.addActionListener(this);
		volume.add(jHalfVolumeResolution, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, 300, 0, 0));
		jHalfVolumeResolution.addActionListener(this);
		positionLabelTextFieldPair(volume, "Maximum VOI file:", maxVOIPathField, 1, 2, 3, 0);
		GUIUtil.enableDragAndDrop(maxVOIPathField);
		return volume;
	}

	private void centerVolume() {
		double dimX = Double.parseDouble(xDimensionField.getText());
		double dimY = Double.parseDouble(yDimensionField.getText());
		double dimZ = Double.parseDouble(zDimensionField.getText());
		double spacingX = Double.parseDouble(xSpacingField.getText());
		double spacingY = Double.parseDouble(ySpacingField.getText());
		double spacingZ = Double.parseDouble(zSpacingField.getText());
		xOriginWorld.setText(new Double(-(dimX-1.0)/2.0 * spacingX).toString());
		yOriginWorld.setText(new Double(-(dimY-1.0)/2.0 * spacingY).toString());
		zOriginWorld.setText(new Double(-(dimZ-1.0)/2.0 * spacingZ).toString());
	}


	JTextField detectorWidthField = new JTextField();
	JTextField detectorHeightField = new JTextField();
	JTextField pixelDimensionXField = new JTextField();
	JTextField pixelDimensionYField = new JTextField();
	JButton jGenerateDetector = new JButton("Configure absorption behavior");
	
	
	JTextField projectionStackSizeField = new JTextField();
	JTextField numSweepsField = new JTextField();
	JTextField sourceToDetectorField = new JTextField();
	JTextField sourceToPatientField = new JTextField();
	JTextField projectionTableFileField = new JTextField();
	JButton jParseGeometry = new JButton("Import");
	JButton jGenerateGeometry = new JButton("Define a trajectory");


	// new trajectory parameters
	JTextField averageAngularIncrement = new JTextField();
	JTextField detectorOffsetU = new JTextField();
	JTextField detectorOffsetV = new JTextField();
	JTextField rotationAxis = new JTextField();
	JComboBox DetectorUDirection = new JComboBox(camAxes);
	JComboBox DetectorVDirection = new JComboBox(camAxes);

	JComboBox optionList;
	JPanel trajectoryFromFile;
	JPanel trajectoryFromParameters;
	JPanel trajectoryPanel;
	JButton trajectoryEditorButton;

	/**
	 * Trajectory Tab
	 * @return
	 */
	private JPanel trajectoryPane(){
		trajectoryPanel = new JPanel();
		trajectoryPanel.setBackground(Color.white);
		GridBagLayout thisLayout = new GridBagLayout();
		thisLayout.rowWeights = new double[] {0.0, 0.1, 0.1, 0.0};
		thisLayout.rowHeights = new int[] {15, 30, 120, 15};
		thisLayout.columnWeights = new double[] {0.0, 0.1, 0.0, 0.1, 0.0};
		thisLayout.columnWidths = new int[] {15, 225, 15, 225, 15};
		trajectoryPanel.setLayout(thisLayout);
		int x = 1;
		int y = 1;
		int topOffset = 0;
		int width = 3;
		String[] options = { "from file", "from parameters"};
		trajectoryEditorButton = new JButton("Edit Trajetory");
		trajectoryEditorButton.addActionListener(this);
		optionList = new JComboBox(options);
		optionList.addActionListener(this);
		JLabel label = new JLabel("Define trajectory:");
		trajectoryPanel.add(label, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		trajectoryPanel.add(optionList, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset, 170, 0, 150));
		trajectoryPanel.add(trajectoryEditorButton, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset-2, 470, 0, 30));
		trajectoryFromFile = trajectoryFromFile();
		trajectoryFromParameters = trajectoryFromParameters();
		if (Configuration.getGlobalConfiguration().getProjectionTableFileName() != null){
			optionList.setSelectedIndex(0);
			trajectoryPanel.add(trajectoryFromFile, createConstraints(0, 2, 5, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, 0, 0, 0, 0));
		} else {
			optionList.setSelectedIndex(1);
			trajectoryPanel.add(trajectoryFromParameters, createConstraints(0, 2, 5, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, 0, 0, 0, 0));
		}
		return trajectoryPanel;
	}


	private JPanel trajectoryFromParameters(){
		JPanel volume = new JPanel();
		volume.setBackground(Color.white);
		volume.setLayout(createSubPaneLayout());
		positionNumericLabelTextFieldPair(volume, "Source to detector distance [mm]", sourceToDetectorField, 1, 1, 00);
		positionNumericLabelTextFieldPair(volume, "Source to patient distance [mm]", sourceToPatientField, 3, 1, 00);

		positionNumericLabelTextFieldPair(volume, "Projection stack size", projectionStackSizeField, 1, 1, 30);
		positionNumericLabelTextFieldPair(volume, "Number of sweeps", numSweepsField, 3, 1, 30);

		positionNumericLabelTextFieldPair(volume, "Rotation axis", rotationAxis, 1, 1, 60);
		positionNumericLabelTextFieldPair(volume, "Average angular increment [deg]", averageAngularIncrement, 3, 1, 60);

		positionNumericLabelTextFieldPair(volume, "Detector offset u [px]", detectorOffsetU, 1, 1, 90);
		positionNumericLabelTextFieldPair(volume, "Detector offset v [px]", detectorOffsetV, 3, 1, 90);


		int x = 1;
		int y = 1;
		int topOffset = 120;
		int width = 3;
		JLabel label = new JLabel("Detector u direction:");
		volume.add(label, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		volume.add(DetectorUDirection, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset, 170, 0, 0));

		topOffset = 150;
		JLabel label2 = new JLabel("Detector v direction:");
		volume.add(label2, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, topOffset + 3, 0, 0, 0));
		volume.add(DetectorVDirection, createConstraints(x, y, width, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.HORIZONTAL, topOffset, 170, 0, 0));

		volume.add(jGenerateGeometry, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 200, 0, 0, 0));
		jGenerateGeometry.addActionListener(this);

		return volume;
	}

	private JPanel trajectoryFromFile(){
		JPanel volume = new JPanel();
		volume.setBackground(Color.white);
		volume.setLayout(createSubPaneLayout());
		positionLabelTextFieldButton(volume, "Projection geometry source file:", projectionTableFileField, 1, 1, 3, 0, jParseGeometry);
		jParseGeometry.addActionListener(this);
		GUIUtil.enableDragAndDrop(projectionTableFileField);
		return volume;
	}

	JTextField recentFileOneField = new JTextField();
	JTextField recentFileTwoField = new JTextField();
	ButtonGroup citationFormatGroup = new ButtonGroup();
	JRadioButton bibButton = new JRadioButton("BibTeX");
	JRadioButton medlineButton = new JRadioButton("Medline");
	JCheckBox importFromDicom = new JCheckBox("Import geometry automatically from DICOM header");
	JCheckBox extrapolateGeometry = new JCheckBox("Extraploate geometry if less than short-scan");
	JCheckBox houns = new JCheckBox("Apply Hounsfield scaling");

	private JPanel otherPane(){
		JPanel volume = new JPanel();
		volume.setBackground(Color.white);
		volume.setLayout(createSubPaneLayout());
		medlineButton.setBackground(Color.WHITE);
		bibButton.setBackground(Color.WHITE);
		JLabel citationLabel = new JLabel("Citation format:");
		volume.add(citationLabel, createConstraints(1, 1, 1, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, 10, 0, 0, 0));
		volume.add(bibButton, createConstraints(3, 1, 1, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, 3, 0, 0, 0));
		volume.add(medlineButton, createConstraints(3, 1, 1, 1, GridBagConstraints.NORTHWEST, GridBagConstraints.NONE, 3, 120, 0, 0));
		volume.add(importFromDicom, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 50, 0, 0, 0));
		volume.add(extrapolateGeometry, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 80, 0, 0, 0));
		volume.add(houns, createConstraints(1, 1, 3, 1, GridBagConstraints.NORTH, GridBagConstraints.NONE, 110, 0, 0, 0));
		importFromDicom.setBackground(Color.WHITE);
		extrapolateGeometry.setBackground(Color.WHITE);
		houns.setBackground(Color.WHITE);
		citationFormatGroup.add(bibButton);
		citationFormatGroup.add(medlineButton);
		positionLabelTextFieldPair(volume, "File for most preferred use:", recentFileOneField, 1, 1, 3, 160);
		GUIUtil.enableDragAndDrop(recentFileOneField);
		positionLabelTextFieldPair(volume, "Reference Volume:", recentFileTwoField, 1, 1, 3, 180);
		GUIUtil.enableDragAndDrop(recentFileTwoField);
		return volume;
	}

	/**
	 * Creates GridBagConstraints
	 * @param x location x
	 * @param y location y
	 * @param width width of the element
	 * @param height height of the element
	 * @param anchor anchor type
	 * @param fill fill type
	 * @param top top inset
	 * @param left left inset
	 * @param bottom bottom inset
	 * @param right right inset
	 * @return the GridBagConstraints
	 */
	static GridBagConstraints createConstraints(int x, int y, int width, int height, int anchor, int fill, int top, int left, int bottom, int right){
		return new GridBagConstraints(x, y, width, height, 0.0, 0.0, anchor, fill, new Insets(top, left, bottom, right), 0, 0);
	}

	private void initGUI() {
		try {
			GridBagLayout thisLayout = new GridBagLayout();
			thisLayout.rowWeights = new double[] {0.0, 0.1, 0.1, 0.0};
			thisLayout.rowHeights = new int[] {15, 150, 150, 15};
			thisLayout.columnWeights = new double[] {0.1, 0.1, 0.1, 0.1};
			thisLayout.columnWidths = new int[] {15, 300, 300, 15};
			setTitle("Configuration");
			getContentPane().setLayout(thisLayout);
			getContentPane().setBackground(Color.WHITE);
			{
				jTabbedPane = new JTabbedPane();
				getContentPane().add(jTabbedPane, new GridBagConstraints(1, 1, 2, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
			}
			{
				jSaveButton = new JButton();
				getContentPane().add(jSaveButton, new GridBagConstraints(1, 3, 2, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSaveButton.setText("save");
				jSaveButton.addActionListener(this);
			}
			jTabbedPane.add("Volume", volumePane());
			jTabbedPane.add("Detector", detectorPane());
			jTabbedPane.add("Trajectory", trajectoryPane());
			jTabbedPane.add("Other", otherPane());
			jTabbedPane.add("Registry", regEditor);
			pack();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	private class VoxelWorldFieldUpdater implements DocumentListener {

		private JTextField world;
		private JTextField voxel;
		private JTextField spacing;

		public VoxelWorldFieldUpdater (JTextField world, JTextField voxel, JTextField spacing) {
			this.world = world;
			this.voxel = voxel;
			this.spacing = spacing;
		}

		public void changedUpdate(DocumentEvent e) {
			updateDependentField();
		}

		public void insertUpdate(DocumentEvent e) {
			updateDependentField();
		}

		public void removeUpdate(DocumentEvent e) {
			updateDependentField();
		}
		private void updateDependentField(){
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					try{
						double spacingDouble = Double.parseDouble(spacing.getText());
						double worldDouble = Double.parseDouble(world.getText());
						double voxelDouble = General.worldToVoxel(0.0, spacingDouble, worldDouble);
						voxel.setText(new Double(voxelDouble).toString());
					} catch (Exception e) {
						System.out.println("Could not parse field. Will update as soon as valid number is available.");
					}
				}
			});
		}
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
