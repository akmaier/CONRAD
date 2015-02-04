package edu.stanford.rsl.conrad.calibration;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.EllipseRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.gui.StackWindow;
import ij.gui.TextRoi;

import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.NumberFormat;
import java.util.ArrayList;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.RootPaneContainer;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.DefaultEditorKit;

import edu.stanford.rsl.apps.gui.ConfigurePipelineFrame;
import edu.stanford.rsl.apps.gui.blobdetection.AutomaticMarkerDetectionWorker;
import edu.stanford.rsl.apps.gui.blobdetection.MarkerDetection;
import edu.stanford.rsl.apps.gui.blobdetection.MarkerDetectionWorker;
import edu.stanford.rsl.conrad.filtering.FastRadialSymmetryTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.NumericalDerivativeComputationTool;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.phantom.MathematicalPhantom;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.StringFileFilter;
import edu.stanford.rsl.conrad.utils.XmlUtils;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class GeometricCalibrationGUI extends JFrame {

	protected static String location;
	ArrayList<JButton> buttonList;
	ConfigurePipelineFrame pipelineFrame;
	ImagePlus currentImage;
	String currentConfigFile;
	NumericalDerivativeComputationTool derivativeTool;
	ArrayList<Double> errors;

	// attributes for line detection/definition
	Roi leftBorder;
	Roi rightBorder;
	Roi leftLineRoi;
	Roi rightLineRoi;

	boolean identify2D;

	Component visualizationComponent;
	int sliceNumber;

	public int getSliceNumber() {
		return sliceNumber;
	}

	public void setSliceNumber(int sliceNumber) {
		this.sliceNumber = sliceNumber;
	}

	JLabel currentStatus;

	GeometricCalibration tool;
	Factorization toolF;

	protected void initTool() {
		this.tool = new GeometricCalibration();
	}

	protected void label(Overlay overlay) {
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(2);
		nf.setMinimumFractionDigits(2);

		for (int n = 0; n < tool.cBeads.size(); n++) {

			CalibrationBead first = tool.cBeads.get(n);
			TextRoi text = new TextRoi(first.getU(), first.getV() - 20, "Bead "
			/*
			 * + tool.ids[n] + " (" + nf.format(first.getX()) + ", " +
			 * nf.format(first.getY()) + ", " + nf.format(first.getZ()) + " | "
			 * + nf.format(first.getU()) + ", " + nf.format(first.getV()) + ")"
			 */);
			overlay.add(text);

		}
	}

	protected void overlayHoughLines() {
		// Roi[] lines = tool.detectHoughLines(currentImage, derivativeTool,
		// rightBorder, leftBorder);
		int sx1 = 300;
		int sy1 = 0;
		int ex1 = 0;
		int ey1 = 960;

		int sx2 = 980;
		int sy2 = 0;
		int ex2 = 0;
		int ey2 = 960;

		try {
			sx1 = UserUtil.queryInt("xStart", sx1);
			sy1 = UserUtil.queryInt("yStart", sy1);
			ex1 = UserUtil.queryInt("xEnd", ex1);
			ey1 = UserUtil.queryInt("xEnd", ey1);

			sx2 = UserUtil.queryInt("xStart", sx2);
			sy2 = UserUtil.queryInt("yStart", sy2);
			ex2 = UserUtil.queryInt("xEnd", ex2);
			ey2 = UserUtil.queryInt("yEnd", ey2);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		leftLineRoi = new Roi(sx1, sy1, ex1, ey1);
		rightLineRoi = new Roi(sx2, sy2, ex2, ey2);
		// leftLineRoi = lines[0];
		// rightLineRoi = lines[1];
		overlayLines();
		this.revalidate();
		this.repaint();
	}

	public GeometricCalibrationGUI() {
		derivativeTool = new NumericalDerivativeComputationTool();
		derivativeTool.setConfigured(true);

		initGUI();
		initTool();

		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});
	}

	protected void setSlice(int n) {
		currentImage.setSlice(n + 1);
		// IJ.run("Enhance Contrast", "0.35");
	}

	protected void overlayMultipleRois(Roi... rois) {
		ShapeRoi accumulationRoi = new ShapeRoi(rois[0]);
		for (int i = 1; i < rois.length; i++) {
			ShapeRoi additionalRoi = new ShapeRoi(rois[i]);
			accumulationRoi = accumulationRoi.or(additionalRoi);
		}
		currentImage.setRoi(accumulationRoi);
	}

	protected void overlayBorders() {
		overlayMultipleRois(leftBorder, rightBorder);

	}

	protected void overlayLines() {
		if (leftLineRoi != null && rightLineRoi != null) {
			overlayMultipleRois(leftLineRoi, rightLineRoi);
		} else {
			overlayHoughLines();
			System.out
					.println("No vertical lines detectable: manual definition.");
		}
	}

	protected void overlayBeads(ArrayList<PointND> listOfBeads1, double radius1) {
		Overlay overlay = new Overlay();

		Roi[] beadRois1 = new Roi[listOfBeads1.size()];
		for (int i = 0; i < listOfBeads1.size(); i++) {
			double x = listOfBeads1.get(i).get(0);
			double y = listOfBeads1.get(i).get(1);
			beadRois1[i] = new EllipseRoi(x - radius1, y - radius1,
					x + radius1, y + radius1, 1.0);
			overlay.add(new EllipseRoi(x - radius1, y - radius1, x + radius1, y
					+ radius1, 1.0));
		}
		// overlayMultipleRois(beadRois1);
		// label(overlay);
		currentImage.setOverlay(overlay);
		// this happens in the GUI
		// currentImage.setOverlay(overlay);
	}

	protected void overlayBeadsF(ArrayList<PointND> listOfBeads1, double radius1) {
		Overlay overlay = new Overlay();
		Roi[] beadRois1 = new Roi[listOfBeads1.size()];
		for (int i = 0; i < listOfBeads1.size(); i++) {
			if (listOfBeads1.get(i) != null) {
				double x = listOfBeads1.get(i).get(0);
				double y = listOfBeads1.get(i).get(1);
				beadRois1[i] = new EllipseRoi(x - radius1, y - radius1, x
						+ radius1, y + radius1, 1.0);
				overlay.add(new EllipseRoi(x - radius1, y - radius1, x
						+ radius1, y + radius1, 1.0));
			}
		}

		// overlayMultipleRois(beadRois1);
		currentImage.setOverlay(overlay);
		// this happens in the GUI
		// currentImage.setOverlay(overlay);
	}

	protected void overlayBeadsCB(ArrayList<CalibrationBead> listOfBeads1,
			double radius1) {
		Overlay overlay = new Overlay();
		overlay.setLabelColor(Color.RED);
		overlay.setFillColor(Color.RED);
		Roi[] beadRois1 = new Roi[listOfBeads1.size()];
		for (int i = 0; i < listOfBeads1.size(); i++) {
			double x = listOfBeads1.get(i).getU();
			double y = listOfBeads1.get(i).getV();
			beadRois1[i] = new EllipseRoi(x - radius1, y - radius1,
					x + radius1, y + radius1, 1.0);
			overlay.add(new EllipseRoi(x - radius1, y - radius1, x + radius1, y
					+ radius1, 1.0));
		}
		// overlayMultipleRois(beadRois1);
		label(overlay);
		currentImage.setOverlay(overlay);
		// this happens in the GUI
		// currentImage.setOverlay(overlay);
	}

	public static String queryFunction(String message, String messageTitle,
			String[] functions) throws Exception {
		return (String) UserUtil.chooseObject(message, messageTitle, functions,
				functions[0]);
	}

	protected void loadImageData(String filename) {
		System.out.println("Loading Image Data: " + filename);
		IJ.open(filename);
		currentImage = IJ.getImage();

		sliceNumber = 0;

		leftBorder = new Roi((int) (0.05 * currentImage.getWidth()), 0,
				(int) (0.30 * currentImage.getWidth()),
				currentImage.getHeight());
		leftBorder.setName("leftBorder");
		rightBorder = new Roi((int) (0.65 * currentImage.getWidth()), 0,
				(int) (0.30 * currentImage.getWidth()),
				currentImage.getHeight());
		rightBorder.setName("rightBorder");

		// overlayBorders();

		updateVisualizationComponent(getCanvas(currentImage));
		setSlice(sliceNumber);
		tool.setSlice(sliceNumber);
		tool.errors = new ArrayList<Double>();

		try {
			if (queryFunction("Function", "Function",
					new String[] { "Naive", "Factorization" }).equals("Naive")) {
				addCalibrationButtons();
			} else {
				addFactorizationButtons();
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/*
		 * Configuration .getGlobalConfiguration() .getGeometry()
		 * .setProjectionMatrices( new Projection[currentImage.getStackSize()]);
		 * Configuration.getGlobalConfiguration().getGeometry()
		 * .setPrimaryAngleArray(new double[currentImage.getStackSize()]);
		 * Configuration .getGlobalConfiguration() .getGeometry()
		 * .setSecondaryAngleArray(new double[currentImage.getStackSize()]);
		 * Configuration.getGlobalConfiguration().getGeometry()
		 * .setNumProjectionMatrices(currentImage.getStackSize());
		 */

	}

	Component getCanvas(ImagePlus imp) {
		StackWindow window = new StackWindow(imp);
		window.setVisible(false);
		return window.getComponents()[0];
	}

	public interface Cursors {
		Cursor WAIT_CURSOR = Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR);
		Cursor DEFAULT_CURSOR = Cursor
				.getPredefinedCursor(Cursor.DEFAULT_CURSOR);
	}

	public static void startWaitCursor(JComponent component) {
		RootPaneContainer root = (RootPaneContainer) component
				.getTopLevelAncestor();
		root.getGlassPane().setCursor(
				Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
		root.getGlassPane().setVisible(true);
	}

	public static void stopWaitCursor(JComponent component) {
		RootPaneContainer root = (RootPaneContainer) component
				.getTopLevelAncestor();
		root.getGlassPane().setCursor(
				Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		root.getGlassPane().setVisible(false);
	}

	private void updateVisualizationComponent(Component cmp) {
		if (visualizationComponent != null)
			getContentPane().remove(visualizationComponent);
		visualizationComponent = cmp;
		getContentPane().add(
				visualizationComponent,
				new GridBagConstraints(1, 1, 1, 3, 0.0, 0.0,
						GridBagConstraints.CENTER, GridBagConstraints.NONE,
						new Insets(10, 10, 10, 10), 0, 0));
	}

	private void loadConfiguration() throws Exception {
		String filename = FileUtil.myFileChoose(".xml", false);
		if (filename != null) {
			tool.config = Configuration.loadConfiguration(filename);
		}
	}

	private void loadImageData() throws Exception {
		String filename = FileUtil.myFileChoose(".zip", false);
		if (filename != null) {
			loadImageData(filename);
			Configuration.getGlobalConfiguration().setRecentFileOne(filename);
		}
	}

	final static String[] tools = { "Naive", "Factorization" };

	public static String queryTool(String message, String messageTitle)
			throws Exception {

		return (String) UserUtil.chooseObject(message, messageTitle, tools,
				tools[0]);
	}

	public void initPane() {

		getContentPane().removeAll();

		GridBagLayout thisLayout = new GridBagLayout();
		thisLayout.rowWeights = new double[] { 0.1, 0.1, 0.8 };
		thisLayout.rowHeights = new int[] { 50, 140, 618 };
		thisLayout.columnWeights = new double[] { 0.1, 0.9 };
		thisLayout.columnWidths = new int[] { 200, 800 };
		getContentPane().setLayout(thisLayout);
		JLabel jLabel1 = new JLabel();
		getContentPane().add(
				jLabel1,
				new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
						GridBagConstraints.CENTER, GridBagConstraints.NONE,
						new Insets(0, 0, 0, 0), 0, 0));
		jLabel1.setText("Visualization");

		jLabel1 = new JLabel();
		getContentPane().add(
				jLabel1,
				new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
						GridBagConstraints.CENTER, GridBagConstraints.NONE,
						new Insets(0, 0, 0, 0), 0, 0));
		jLabel1.setText("Tools");

		int insetY = 20;
		int insetX = 10;

		JButton pipeline = new JButton();
		pipeline.setText("configure Pipeline");
		pipeline.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				pipelineFrame = new ConfigurePipelineFrame();
				pipelineFrame.setVisible(true);
				pipelineFrame.setSaveToDisk(false);
			}
		});

		getContentPane().add(
				pipeline,
				new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));

		insetY += 40;

		JButton selectConfig = new JButton();
		selectConfig.setText("select Configuration");
		selectConfig.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				try {
					loadConfiguration();
					tool.reConfigure();
					doLayout();
					revalidate();
					repaint();
				} catch (Exception e1) {
					System.out.println("Action not completed: " + e1.toString());
				}
			}
		});

		getContentPane().add(
				selectConfig,
				new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));

		insetY += 40;

		JButton selectData = new JButton();
		selectData.setText("select File");
		selectData.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				try {
					loadImageData();
					doLayout();
					revalidate();
					repaint();
				} catch (Exception e1) {
					System.out.println("Action not completed: " + e1.toString());
				}
			}
		});

		getContentPane().add(
				selectData,
				new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		// insetY += 20;

		// if (Configuration.getGlobalConfiguration().getRecentFileOne() !=
		// null) {
		// loadImageData(Configuration.getGlobalConfiguration()
		// .getRecentFileOne());
		// }
		this.revalidate();
		this.repaint();

	}

	private String saveAs() {
		try {
			String filename = FileUtil.myFileChoose(".xml", true);
			if (!filename.endsWith(".xml"))
				filename += ".xml";
			save(filename);
			return filename;
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		return null;
	}

	private void save(String filename) {
		Configuration.saveConfiguration(Configuration.getGlobalConfiguration(),
				filename);
		Configuration.loadConfiguration(filename);
	}

	/**
	 * Create an Edit menu to support cut/copy/paste.
	 */
	@SuppressWarnings("serial")
	public JMenuBar createMenuBar() {
		JMenuItem menuItem = null;
		JMenuBar menuBar = new JMenuBar();

		JMenu mainMenu = new JMenu("Configuration");
		mainMenu.setMnemonic(KeyEvent.VK_C);

		menuItem = new JMenuItem(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				try {
					if (pipelineFrame != null)
						pipelineFrame.exit();
					Configuration configuration = new Configuration();
					configuration
							.setFilterPipeline(new ImageFilteringTool[] { new FastRadialSymmetryTool() });
					configuration.setCurrentPath(location);
					Configuration.setGlobalConfiguration(tool.config);

					Thread thread = new Thread(new Runnable() {
						@Override
						public void run() {
							initPane();
						}
					});
					thread.start();
				} catch (Exception e1) {
					System.out.println(e1.getLocalizedMessage());
				}
			}
		});
		menuItem.setText("New");
		menuItem.setMnemonic(KeyEvent.VK_N);
		mainMenu.add(menuItem);

		menuItem = new JMenuItem(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				try {
					JFileChooser FC = new JFileChooser();
					File dir = new File(location);
					FC.setCurrentDirectory(dir);
					FC.setVisible(true);
					FC.setFileFilter(new StringFileFilter(".xml"));
					FC.showOpenDialog(null);
					String filename = FC.getSelectedFile().getAbsolutePath();
					Configuration.setGlobalConfiguration(Configuration
							.loadConfiguration(filename));
					currentConfigFile = filename;
					if (pipelineFrame != null)
						pipelineFrame.exit();
					initPane();
				} catch (Exception e1) {
					e1.printStackTrace();
				}
			}
		});
		menuItem.setText("Load");
		menuItem.setMnemonic(KeyEvent.VK_L);
		mainMenu.add(menuItem);

		menuItem = new JMenuItem(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (currentConfigFile == null)
					currentConfigFile = saveAs();
				else
					save(currentConfigFile);
			}
		});
		menuItem.setText("Save");
		menuItem.setMnemonic(KeyEvent.VK_S);
		mainMenu.add(menuItem);

		menuItem = new JMenuItem(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				currentConfigFile = saveAs();
			}
		});
		menuItem.setText("Save As");
		menuItem.setMnemonic(KeyEvent.VK_A);
		mainMenu.add(menuItem);

		menuBar.add(mainMenu);

		mainMenu = new JMenu("Edit");
		mainMenu.setMnemonic(KeyEvent.VK_E);

		menuItem = new JMenuItem(new DefaultEditorKit.CutAction());
		menuItem.setText("Cut");
		menuItem.setMnemonic(KeyEvent.VK_T);
		mainMenu.add(menuItem);

		menuItem = new JMenuItem(new DefaultEditorKit.CopyAction());
		menuItem.setText("Copy");
		menuItem.setMnemonic(KeyEvent.VK_C);
		mainMenu.add(menuItem);

		menuItem = new JMenuItem(new DefaultEditorKit.PasteAction());
		menuItem.setText("Paste");
		menuItem.setMnemonic(KeyEvent.VK_P);
		mainMenu.add(menuItem);

		menuBar.add(mainMenu);

		mainMenu = new JMenu("Help");
		mainMenu.setMnemonic(KeyEvent.VK_H);

		menuItem = new JMenuItem(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				JOptionPane
						.showMessageDialog(
								null,
								"Geometric Calibration GUI\n\nCreated by Andreas Maier\nandreas.maier@fau.de",
								"About", JOptionPane.PLAIN_MESSAGE);
			}
		});
		menuItem.setText("About");
		menuItem.setMnemonic(KeyEvent.VK_A);
		mainMenu.add(menuItem);

		menuBar.add(mainMenu);

		return menuBar;
	}

	JTextField beadRadius = new JTextField();
	JTextField smallBeadPercentile = new JTextField();
	JTextField threshold = new JTextField();
	JTextField thresh2Percentile = new JTextField();

	public void initGUI() {
		this.setBackground(Color.WHITE);
		this.setTitle("Geometric Calibration GUI");
		this.setSize(1400, 980);
		this.setJMenuBar(createMenuBar());
		this.getContentPane().setBackground(Color.WHITE);
	}

	boolean intensity = false;

	protected void addCalibrationButtons() {

		int insetY = 0;
		int insetX = 10;

		buttonList = new ArrayList<JButton>();

		final String[] phantoms = { "Random Distribution Phantom",
				"Randomized Helix Phantom", "Mathematical Phantom" };

		final JComboBox<String> phantomBox = new JComboBox<String>(phantoms);
		phantomBox.addActionListener(new AbstractAction() {

			@Override
			public void actionPerformed(ActionEvent e) {
				tool.setPhantom(phantoms[phantomBox.getSelectedIndex()]);
				if (tool.phantom instanceof MathematicalPhantom) {

				}
				System.out.println(tool.phantomName + " chosen...");
			}

		});

		getContentPane().add(
				phantomBox,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));

		insetY += 40;

		JButton export = new JButton();
		buttonList.add(export);
		export.setText("export to .scad");
		export.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				tool.phantom.writeToOpenSCAD();
			}
		});

		getContentPane().add(
				export,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		JButton newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("next Projection");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				sliceNumber++;
				if (sliceNumber >= currentImage.getStackSize())
					sliceNumber = currentImage.getStackSize() - 1;
				setSlice(sliceNumber);
				tool.setSlice(sliceNumber);
				revalidate();
				repaint();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("previous Projection");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				sliceNumber--;
				if (sliceNumber < 0)
					sliceNumber = 0;
				setSlice(sliceNumber);
				tool.setSlice(sliceNumber);
				revalidate();
				repaint();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		int offsetLeft = 40;
		int offsetMid = 180;
		int offsetRight = 80;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("detect Boundary");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				Roi[] lines = tool.detectHoughLines(currentImage,
						derivativeTool, rightBorder, leftBorder);
				leftLineRoi = lines[0];
				rightLineRoi = lines[1];
				overlayLines();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("l");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (leftBorder.getName()
						.equals(currentImage.getRoi().getName())) {
					leftBorder = currentImage.getRoi();
				}
				currentImage.setRoi(leftBorder);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("r");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (rightBorder.getName().equals(
						currentImage.getRoi().getName())) {
					rightBorder = currentImage.getRoi();
				}
				currentImage.setRoi(rightBorder);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200 + 40, 0, insetX), 0,
						0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("estimate rotation");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				tool.idealBeads();
				tool.detectBeads(currentImage, leftLineRoi, rightLineRoi);
				tool.estimateRotation();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("ideal Beads 2D");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {

				tool.idealBeads();
				overlayBeads(tool.beads2DIdeal, tool.beadRadius);

			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("detect/identify Beads 2D");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {

				tool.setDetection(Integer.parseInt(beadRadius.getText()),
						Double.parseDouble(threshold.getText()));
				tool.detectBeads(currentImage, leftLineRoi, rightLineRoi);
				tool.idealBeads();
				tool.computeBeadIDsForward();
				identify2D = true;
				// overlayBeads(tool.beads2DIdeal, tool.beadRadius);
				System.out.println("correspondences detected...");
				overlayBeadsCB(tool.cBeads, tool.beadRadius);

			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("l");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (leftLineRoi.getName().equals(
						currentImage.getRoi().getName())) {
					leftLineRoi = currentImage.getRoi();
				}
				currentImage.setRoi(leftLineRoi);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("r");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (rightLineRoi.getName().equals(
						currentImage.getRoi().getName())) {
					rightLineRoi = currentImage.getRoi();
				}
				currentImage.setRoi(rightLineRoi);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200 + 40, 0, insetX), 0,
						0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("find Points");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {

				tool.setDetection(Integer.parseInt(beadRadius.getText()),
						Double.parseDouble(threshold.getText()));
				tool.detectBeads(currentImage, leftLineRoi, rightLineRoi);
				tool.findPointsNaive(currentImage, leftLineRoi, rightLineRoi);
				overlayBeadsCB(tool.cBeads, tool.beadRadius);

			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		JLabel beadLabel1 = new JLabel("Bead Radius");
		// JLabel beadLabel2 = new JLabel("Small Bead Percentile");
		JLabel threshLabel1 = new JLabel("Threshold Percentile");
		// JLabel threshLabel2 = new JLabel("High Threshold Percentile");
		beadRadius.setText("" + tool.beadRadius);
		// smallBeadPercentile.setText("" + tool.smallBeadPercentileD);
		threshold.setText("" + tool.threshold);
		// thresh2Percentile.setText("" + tool.thresh2PercentileD);

		getContentPane().add(
				beadLabel1,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		getContentPane().add(
				beadRadius,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		insetY += 40;

		getContentPane().add(
				threshLabel1,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		getContentPane().add(
				threshold,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("compute Matrix");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				NumberFormat nf = NumberFormat.getInstance();
				nf.setMaximumFractionDigits(2);
				nf.setMinimumFractionDigits(2);
				Overlay overlay = new Overlay();
				double error = tool.calibrate(sliceNumber, overlay);
				// double error = tool.testP(sliceNumber, overlay);
				currentImage.setOverlay(overlay);
				System.out.println("Projection: " + sliceNumber + " error: "
						+ nf.format(error) + " [px/bead].");
				currentStatus.setText("Projection: " + sliceNumber + " error: "
						+ nf.format(error) + " [px/bead].");

			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;
		offsetLeft = 0;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("continue");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				Thread thread = new Thread(new Runnable() {
					public void run() {
						NumberFormat nf = NumberFormat.getInstance();
						nf.setMaximumFractionDigits(2);
						nf.setMinimumFractionDigits(2);
						ArrayList<CalibrationBead> reference = tool.cBeads;

						startWaitCursor(getRootPane());
						double error = 0;
						while (/*
								 * (error < 1) &&
								 */sliceNumber != currentImage.getStackSize() - 1) {
							sliceNumber++;
							if (sliceNumber >= currentImage.getStackSize())
								sliceNumber = currentImage.getStackSize() - 1;
							setSlice(sliceNumber);
							tool.setSlice(sliceNumber);
							tool.detectBeads(currentImage, leftLineRoi,
									rightLineRoi);
							tool.findPointsNaive(currentImage, leftBorder,
									leftBorder);
							/*
							 * tool.detectBeads(currentImage, leftLineRoi,
							 * rightLineRoi); tool.idealBeads(); if (identify2D)
							 * { tool.computeBeadIDsForward(); } else {
							 * tool.computeBeadIDsBack(); }
							 */
							Overlay overlay = new Overlay();
							error = tool.calibrate(sliceNumber, overlay);
							if (error > 0.15) {
								tool.detectBeads(currentImage, leftLineRoi,
										rightLineRoi);
								tool.computeBeadIDsForward();
								Overlay newOverlay = new Overlay();
								double newError = tool.calibrate(sliceNumber,
										newOverlay);
								if (newError < error) {
									error = newError;
									overlay = newOverlay;
								}
							}

							currentImage.setOverlay(overlay);
							System.out.println("Projection: " + sliceNumber
									+ " error: " + nf.format(error)
									+ " [px/bead].");
							currentStatus.setText("Projection: " + sliceNumber
									+ " error: " + nf.format(error)
									+ " [px/bead].");
							Thread.yield();
							try {
								Thread.sleep(10);
							} catch (InterruptedException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
						stopWaitCursor(getRootPane());
					};
				});
				thread.start();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("save Results");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				try {
					System.out.println(tool.errors);
					String name = tool.phantom.getName().concat(
							"Calibration.txt");
					BufferedWriter out = new BufferedWriter(
							new FileWriter(name));
					out.write("Projection Matrices\n");
					for (SimpleMatrix p : tool.projectionMatrices) {
						out.write("Slice "
								+ tool.projectionMatrices.indexOf(p)
								+ ": "
								+ p
								+ ", Backprojection Error: "
								+ tool.errors.get(tool.projectionMatrices
										.indexOf(p)) + "\n");
					}

					out.close();
				} catch (Exception ex) {

				}
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		if (currentStatus != null) {
			getContentPane().remove(currentStatus);
		}
		currentStatus = new JLabel("Loaded.");
		getContentPane().add(
				currentStatus,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

	}

	public void addFactorizationButtons() {
		int insetY = 0;
		int insetX = 10;

		buttonList = new ArrayList<JButton>();

		final String[] phantoms = { "Random Distribution Phantom",
				"Randomized Helix Phantom", "Mathematical Phantom" };

		final JComboBox<String> phantomBox = new JComboBox<String>(phantoms);
		phantomBox.addActionListener(new AbstractAction() {

			@Override
			public void actionPerformed(ActionEvent e) {
				tool.setPhantom(phantoms[phantomBox.getSelectedIndex()]);
				if (tool.phantom instanceof MathematicalPhantom) {

				}
				System.out.println(tool.phantomName + " chosen...");
			}

		});

		getContentPane().add(
				phantomBox,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));

		insetY += 40;

		JButton export = new JButton();
		buttonList.add(export);
		export.setText("export to .scad");
		export.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				tool.phantom.writeToOpenSCAD();
			}
		});

		getContentPane().add(
				export,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		JButton newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("Intesity/Integral");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (intensity) {
					IJ.run(currentImage, "Log", "");
					IJ.run(currentImage, "Multiply...", "value=-1.0");
					intensity = false;
				} else {
					IJ.run(currentImage, "Multiply...", "value=-1.0");
					IJ.run(currentImage, "Exp", "");
					intensity = true;
				}
				revalidate();
				repaint();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("next Projection");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				sliceNumber++;
				if (sliceNumber >= currentImage.getStackSize())
					sliceNumber = currentImage.getStackSize() - 1;
				setSlice(sliceNumber);
				tool.setSlice(sliceNumber);
				revalidate();
				repaint();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("previous Projection");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				sliceNumber--;
				if (sliceNumber < 0)
					sliceNumber = 0;
				setSlice(sliceNumber);
				revalidate();
				repaint();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX, 0, insetX), 0, 0));
		insetY += 40;

		int offsetLeft = 40;
		int offsetMid = 180;
		int offsetRight = 80;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("detect Boundary");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				Roi[] lines = tool.detectHoughLines(currentImage,
						derivativeTool, rightBorder, leftBorder);
				leftLineRoi = lines[0];
				rightLineRoi = lines[1];
				overlayLines();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("l");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (leftBorder.getName()
						.equals(currentImage.getRoi().getName())) {
					leftBorder = currentImage.getRoi();
				}
				currentImage.setRoi(leftBorder);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("r");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (rightBorder.getName().equals(
						currentImage.getRoi().getName())) {
					rightBorder = currentImage.getRoi();
				}
				currentImage.setRoi(rightBorder);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200 + 40, 0, insetX), 0,
						0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("find initial points");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {

				tool.setDetection(Integer.parseInt(beadRadius.getText()),
						Double.parseDouble(threshold.getText()));
				tool.getInitialPoints(false, currentImage, leftLineRoi,
						rightLineRoi);
				overlayBeadsF(tool.referencePoints, tool.beadRadius);

			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("l");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (leftLineRoi.getName().equals(
						currentImage.getRoi().getName())) {
					leftLineRoi = currentImage.getRoi();
				}
				currentImage.setRoi(leftLineRoi);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("r");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				if (rightLineRoi.getName().equals(
						currentImage.getRoi().getName())) {
					rightLineRoi = currentImage.getRoi();
				}
				currentImage.setRoi(rightLineRoi);
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200 + 40, 0, insetX), 0,
						0));

		insetY += 40;

		JLabel beadLabel1 = new JLabel("Bead Radius");
		// JLabel beadLabel2 = new JLabel("Small Bead Percentile");
		JLabel threshLabel1 = new JLabel("Threshold Percentile");
		// JLabel threshLabel2 = new JLabel("High Threshold Percentile");
		beadRadius.setText("" + tool.beadRadius);
		// smallBeadPercentile.setText("" + tool.smallBeadPercentileD);
		threshold.setText("" + tool.threshold);
		// thresh2Percentile.setText("" + tool.thresh2PercentileD);

		getContentPane().add(
				beadLabel1,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		getContentPane().add(
				beadRadius,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		insetY += 40;

		getContentPane().add(
				threshLabel1,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		getContentPane().add(
				threshold,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft + 200, 0, insetX
										+ offsetRight / 2), 0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("find all points");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				Thread thread = new Thread(new Runnable() {

					public void run() {
						startWaitCursor(getRootPane());
						while (sliceNumber != currentImage.getStackSize() - 1) {
							sliceNumber++;
							if (sliceNumber >= currentImage.getStackSize())
								sliceNumber = currentImage.getStackSize() - 1;
							setSlice(sliceNumber);
							tool.setSlice(sliceNumber);
							tool.findPoints(currentImage, leftLineRoi,
									rightLineRoi);
							overlayBeadsF(tool.imagePoints.get(sliceNumber),
									tool.beadRadius);
							currentStatus.setText("Projection: "
									+ sliceNumber
									+ "Beads estimated: "
									+ (tool.phantom.getNumberOfBeads() - tool.imagePoints
											.get(sliceNumber).size()));
							Thread.yield();
							try {
								Thread.sleep(50);
							} catch (InterruptedException ex) {
								// TODO Auto-generated catch block
								ex.printStackTrace();
							}
						}
						// tool.removePoints();
						stopWaitCursor(getRootPane());
					}
				});

				thread.start();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("compute Matrices");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				tool.factorize();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;
		offsetLeft = 0;

		newButton = new JButton();
		buttonList.add(newButton);
		newButton.setText("calibrate");
		newButton.addActionListener(new AbstractAction() {
			public void actionPerformed(ActionEvent e) {
				Thread thread = new Thread(new Runnable() {
					public void run() {
						NumberFormat nf = NumberFormat.getInstance();
						nf.setMaximumFractionDigits(2);
						nf.setMinimumFractionDigits(2);

						startWaitCursor(getRootPane());
						double error = 0;
						sliceNumber = 0;
						while (sliceNumber != currentImage.getStackSize() - 1) {

							if (sliceNumber >= currentImage.getStackSize())
								sliceNumber = currentImage.getStackSize() - 1;

							setSlice(sliceNumber);
							tool.setSlice(sliceNumber);
							Overlay overlay = new Overlay();
							tool.detectBeads(currentImage, leftBorder,
									leftBorder);
							error = tool.calibrateF(sliceNumber, overlay);
							if (error > 1)
								System.out.println("Error in slice"
										+ sliceNumber + ": " + error);
							currentImage.setOverlay(overlay);
							currentStatus.setText("Projection: " + sliceNumber
									+ " error: " + nf.format(error)
									+ " [px/bead].");
							sliceNumber++;
							Thread.yield();
							try {
								Thread.sleep(10);
							} catch (InterruptedException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
						stopWaitCursor(getRootPane());
					};
				});
				thread.start();
			}
		});

		getContentPane().add(
				newButton,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

		insetY += 40;

		if (currentStatus != null) {
			getContentPane().remove(currentStatus);
		}
		currentStatus = new JLabel("Loaded.");
		getContentPane().add(
				currentStatus,
				new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
						GridBagConstraints.NORTH,
						GridBagConstraints.HORIZONTAL, new Insets(insetY,
								insetX + offsetLeft, 0, insetX + offsetRight),
						0, 0));

	}

	public static void main(String[] args) {
		if (args.length > 0)
			location = args[0];
		// CONRAD.setup();
		new ImageJ().setVisible(true);
		GeometricCalibrationGUI gui = new GeometricCalibrationGUI();
		gui.setVisible(true);

	}

}
