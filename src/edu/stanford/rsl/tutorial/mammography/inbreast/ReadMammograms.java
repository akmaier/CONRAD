package edu.stanford.rsl.tutorial.mammography.inbreast;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.xml.xpath.XPathExpressionException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.mammography.Mammogram;
import edu.stanford.rsl.tutorial.mammography.Mammogram.BreastDensityACR;
import edu.stanford.rsl.tutorial.mammography.Mammogram.Findings;
import edu.stanford.rsl.tutorial.mammography.Mammogram.MammographyView;
import edu.stanford.rsl.tutorial.mammography.Mammogram.SideOfBody;

public class ReadMammograms {

	private SideOfBody readOnlySides = SideOfBody.Both;
	private MammographyView readOnlyViews = MammographyView.Both;
	private Findings readOnlyKind = Findings.Mass;
	private ArrayList<Integer> queriedPatiendIds = new ArrayList<Integer>();
	private ArrayList<Integer> patientID = new ArrayList<Integer>();
	private ArrayList<SideOfBody> side = new ArrayList<SideOfBody>();
	private ArrayList<MammographyView> view = new ArrayList<MammographyView>();
	private ArrayList<BreastDensityACR> density = new ArrayList<BreastDensityACR>();
	private ArrayList<Integer> birads = new ArrayList<Integer>();
	private ArrayList<ArrayList<Findings>> findings = new ArrayList<ArrayList<Findings>>();
	private ArrayList<ArrayList<Roi>> roi = new ArrayList<ArrayList<Roi>>();

	private File[] listOfFiles = null;
	private ArrayList<Mammogram> mammograms = new ArrayList<Mammogram>();
	private int mammoCounter = 0;

	//set path to Dicoms, XML and CSV file
	private String imgDir = null;
	private String xmlDir = null;
	private String csvFile = null;

	public static void main(String[] args) {

		String imgDir =  "D:/Data/INbreast/AllDICOMs/";
		String xmlDir =  "D:/Data/INbreast/ALLXML/";
		String csvFile = "D:/Data/INbreast//INbreast.csv";
		
		ReadMammograms reader = new ReadMammograms(imgDir, xmlDir, csvFile);
		reader.setOptions(SideOfBody.Both, MammographyView.Both, Findings.All);
		reader.readImagesSuccessively();
		
		new ImageJ();
		
		Mammogram m = reader.readNextMammogram();
		while (m != null) {
			// here you can do cool stuff with the images
			RoiManager manager = new RoiManager();
			m = reader.readNextMammogram();
			ImagePlus imp = ImageUtil.wrapGrid(m.getImage(), "Mammogram");
			ArrayList<Roi> rois = m.getRois();
			for(int i = 0; i < rois.size(); i++){
				manager.add(imp, rois.get(i), i+1);
			}
			imp.show();
			manager.runCommand("Show All");
			
			manager.close();
			imp.close();			
		}
		System.out.println("Done.");
	}
	
	/**
	 * setOptions sets witch parameters will be accepted 
	 * 
	 * @param sob SideOfBody
	 * @param vws MammographyView
	 * @param finding Findings
	 */
	public void setOptions(SideOfBody sob, MammographyView vws, Findings finding) {
		this.readOnlySides = sob;
		this.readOnlyViews = vws;
		this.readOnlyKind = finding;
	}
	
	/**
	 * all CSVdata will be read
	 * all mammograms who satisfy setOptions will be read 
	 * a new mammogram constructor will be set for every suitable mammogram
	 */
	public void readImages() {
		readCSVdata();
		readMammograms();
	}
	
	/**
	 * all CSVdata will be read
	 * all mammograms who satisfy setOptions will be read, to start the constructor  
	 * readNextMammogram has to be used
	 */
	public void readImagesSuccessively() {
		readCSVdata();
		prepReadMammograms();
	}

	public ArrayList<Mammogram> getMammograms() {
		return mammograms;
	}

	public void setMammograms(ArrayList<Mammogram> mammograms) {
		this.mammograms = mammograms;
	}
	
	private void readMammograms() {

		//the path of imageFolder is given to listOfFiles, all DICOM images are now listed
		File folder = new File(imgDir);
		listOfFiles = folder.listFiles();

		//pid = PatiendtID
		ArrayList<Integer> pid = new ArrayList<Integer>();
		for (int i = 0; i < listOfFiles.length; i++) {
			//fn = filename, consist of 6 Elements, separated by a "_"
			String fn = listOfFiles[i].getName();
			if (fn.contains(".dcm")) {
				String[] identifiers = fn.split("_");
				pid.add(Integer.parseInt(identifiers[0]));
			}
		}

		//calls method getQueriedPatientIds= only patients/images
		//satisfying wanted SideOfBody,MammographyView and Findings are returned
		queriedPatiendIds = getQueriedPatientIds(pid);

		System.out.println("Reading " + String.valueOf(queriedPatiendIds.size()) + " mammograms.");

		//all filenames will be read and checked if DICOM
		for (int i = 0; i < listOfFiles.length; i++) {
			String fn = listOfFiles[i].getName();
			if (fn.contains(".dcm")) {
				//separate patientID from filename, by splitting
				String[] identifiers = listOfFiles[i].getName().split("_");
				int patientId = Integer.parseInt(identifiers[0]);
				// check if the patientID from the i-th file matches one of the queriedPatiendIds
				// returns positiv integer if found, -1 if not found
				int inList = getPatiendIdx(patientId, queriedPatiendIds);
				if (inList >= 0) {
					// get actual index from the list of  all collected patientID
					int refIdx = getPatiendIdx(patientId, this.patientID);
					//load the actual image as Grid2D
					ImagePlus imgAsImp = IJ.openImage(listOfFiles[i].getAbsolutePath());
					Grid2D img = ImageUtil.wrapImageProcessor(imgAsImp.getProcessor());
					//collect all information of the Object of refIdx and start a constructor  
					Mammogram mammo = new Mammogram(img, side.get(refIdx), view.get(refIdx), density.get(refIdx),
							findings.get(refIdx), roi.get(refIdx), birads.get(refIdx));
					//adding this mammogram to ArrayList
					this.mammograms.add(mammo);
				}
			}
		}
		System.out.println("ReadMammograms - done.");
	}

	
	
	private void prepReadMammograms() {
		this.mammoCounter = 0;
		//the path of imageFolder is given to listOfFiles, all DICOM images are now listed
		File folder = new File(imgDir);
		listOfFiles = folder.listFiles();
		//pid = PatiendtID
		ArrayList<Integer> pid = new ArrayList<Integer>();
		for (int i = 0; i < listOfFiles.length; i++) {
			//fn = filename, consist of 6 Elements, separated by a "_"
			String fn = listOfFiles[i].getName();
			if (fn.contains(".dcm")) {
				String[] identifiers = fn.split("_");
				pid.add(Integer.parseInt(identifiers[0]));
			}
		}
		//calls method getQueriedPatientIds= only patients/images
		//satisfying wanted SideOfBody,MammographyView and Findings are returned
		queriedPatiendIds = getQueriedPatientIds(pid);
	}

	public Mammogram readNextMammogram() {
		// with every checked mammogram, mammoCounter increases
		while (mammoCounter < listOfFiles.length) {
			String fn = listOfFiles[mammoCounter].getName();
			// check if file is a DICOM, if not, mammoCounter++ 
			if (fn.contains(".dcm")) {
				// get the name of the file in identifiers[0] , rest is also stored but will not be used
				String[] identifiers = listOfFiles[mammoCounter].getName().split("_");
				int patientId = Integer.parseInt(identifiers[0]);
				// check if this patientID of matches one of the queriedPatiendIds
				int inList = getPatiendIdx(patientId, queriedPatiendIds);
				// if patientId matches one of the queriedIds an positive integer will be returned
				// else -1 will be returned and mammoCounter++ will be executed
				if (inList >= 0) {
					// get actual index of all stored patients
					int refIdx = getPatiendIdx(patientId, this.patientID);
					ImagePlus imgAsImp = IJ.openImage(listOfFiles[mammoCounter].getAbsolutePath());
					Grid2D img = ImageUtil.wrapImageProcessor(imgAsImp.getProcessor());
					//start the constructor
					Mammogram mammo = new Mammogram(img, side.get(refIdx), view.get(refIdx), density.get(refIdx),
							findings.get(refIdx), roi.get(refIdx), birads.get(refIdx));
					mammoCounter++;
					return mammo;
				} else {
					mammoCounter++;
				}
			} else {
				mammoCounter++;
			}
		}
		return null;
	}
/**
 * 
 * @param listOfPatients = all collected patients
 * 
 * @return ArrayList<Integer> containing patientIDs of images satisfying wanted SideOfBody, MammographyView and Findings
 * 
 */
	private ArrayList<Integer> getQueriedPatientIds(ArrayList<Integer> listOfPatients) {
		//list is the return value, will be filled later
		ArrayList<Integer> list = new ArrayList<Integer>();
		// the i-th element of patientID (all 410 patients) will be compared to
		// the id-th element of listOfPatients () 
		for (int i = 0; i < patientID.size(); i++) {
			boolean isInList = false;
			for (int id = 0; id < listOfPatients.size(); id++) {
				if (patientID.get(i).equals(listOfPatients.get(id))) {
					isInList = true;
					continue;
				}
			}
			if (!isInList) {
				continue;
			}
			// read out, witch SideOfBody do we currently want to look up
			String[] sides = readOnlySides.getValue();
			boolean isCorrectSide = false;
			for (int s = 0; s < sides.length; s++) {
				// compare element i with the elements of SideOfBody
				if (sides[s].equals(side.get(i).getValue()[0])) {
					isCorrectSide = true;
				}
			}
			if (!isCorrectSide) {
				continue;
			}
			// read out, witch MammographyViews do we currently want to look up
			String[] views = readOnlyViews.getValue();
			boolean isCorrectView = false;
			for (int s = 0; s < views.length; s++) {
				// compare element i with the elements of the MammographyView
				if (views[s].equals(view.get(i).getValue()[0])) {
					isCorrectView = true;
				}
			}
			if (!isCorrectView) {
				continue;
			}
			// read out, witch Finding do we currently want to look up
			String[] kind = readOnlyKind.getValue();
			ArrayList<Findings> fndgInMammoI = findings.get(i);
			boolean isCorrectFinding = false;
			for (int s = 0; s < kind.length; s++) {
				//compare element i with the elements of Findings
				for (Findings f : fndgInMammoI) {
					if (kind[s].equals(f.getValue()[0])) {
						isCorrectFinding = true;
					}
				}
			}
			if (!isCorrectFinding) {
				continue;
			}
			// if SideOfBody, MammographyView and Findings were matching out specification
			// the matching patientID will now be stored in "list"
			list.add(patientID.get(i));
		}
		return list;
	}
	
	/**
	 * 
	 * @param id = patientID which could be in the idList
	 * @param idList = all patientIDs that have been previously collected
	 * @return index i from idList if found , returns -1 if not found
	 */
	private int getPatiendIdx(int id, ArrayList<Integer> idList) {
		for (int i = 0; i < idList.size(); i++) {
			if (idList.get(i).equals(id))
				return i;
		}
		return -1;
	}
	
	/**
	 * the provided CSVdate will be read 
	 */
	private void readCSVdata() {
		System.out.println("Parsing CSV file.");

		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ";";

		try {
			//read the csvFile
			br = new BufferedReader(new FileReader(csvFile));
			int count = -1;
			while ((line = br.readLine()) != null) {
				count++;
				//first line of csvFile: Patient; Patient age; Laterality; View; Acquisition date; File Name; ACR; Bi-Rads
				//ignore first line, it has only labels no, patient information
				if (count == 0)
					continue;
				char lastCh = line.charAt(line.length() - 1);
				while (lastCh == ' ') {
					line = line.substring(0, line.length() - 1);
					lastCh = line.charAt(line.length() - 1);
				}
				if (line.charAt(line.length() - 1) == ';') {
					line = line.substring(0, line.length() - 1);
				}

				// use separator to split line
				// entry containes all information of removed, removed, SideOfBody, MammographyView, Date, FileName, ACR , Bi-Rads
				String[] entry = line.split(cvsSplitBy);
				// parse Side Of Body
				if (entry[2].equals(SideOfBody.Left.getValue()[0])) {
					this.side.add(SideOfBody.Left);
				} else if (entry[2].equals(SideOfBody.Right.getValue()[0])) {
					this.side.add(SideOfBody.Right);
				} else {
					this.side.add(SideOfBody.Unknown);
				}
				// parse View
				if (entry[3].equals(MammographyView.ML.getValue()[0])) {
					this.view.add(MammographyView.ML);
				} else if (entry[3].equals(MammographyView.CC.getValue()[0])) {
					this.view.add(MammographyView.CC);
				} else {
					this.view.add(MammographyView.Unknown);
				}
				// parse Patient ID
				int patient = Integer.parseInt(entry[5]);
				this.patientID.add(patient);

				// parse ACR
				int acr = -1;
				if (entry.length > 6 && !entry[6].contains(" ")) {
					//extract acr score and compare to the values denoted in Mammogram.java
					acr = Integer.parseInt(entry[6]);
				}
				if (acr == BreastDensityACR.Fatty.getACRValue()) {
					this.density.add(BreastDensityACR.Fatty);
				} else if (acr == BreastDensityACR.FattyGlandular1.getACRValue()) {
					this.density.add(BreastDensityACR.FattyGlandular1);
				} else if (acr == BreastDensityACR.FattyGlandular2.getACRValue()) {
					this.density.add(BreastDensityACR.FattyGlandular2);
				} else if (acr == BreastDensityACR.Glandular.getACRValue()) {
					this.density.add(BreastDensityACR.Glandular);
				} else {
					this.density.add(BreastDensityACR.Unknown);
				}
				// parse Birads
				int biradsScore = 0;
				if (entry.length > 7) {
					biradsScore = Integer.parseInt(entry[7].substring(0, 1));
				}
				this.birads.add(biradsScore);

				//finding with their corresponding rois are stored
				ArrayList<Findings> fndgs = new ArrayList<Findings>();
				ArrayList<Roi> rois = new ArrayList<Roi>();
				String[] rawOutput = null;
				// rawOutput contains:
				// [0]total number of findings
				// [1]class of nth finding
				// [2]number of XY coordinates
				// [3...] XY coordinates pairs as an array
				// [1]to[3..]repeat themself for every finding

				// health breast have a biradsScore of 1
				if (biradsScore != 1) {
					XMLparser parser = new XMLparser();
					parser.readXml(xmlDir + entry[5] + ".xml");
					// entry[5] contains the name of the .xml file
					// value is of ArrayList<String>
					fndgs.addAll(parser.getFindings());
					// Roi.Polygon => X/Y coordinates, width & height denotes
					// the total XY variance
					rois.addAll(parser.getRois());
					this.findings.add(fndgs);
					this.roi.add(rois);
				} else {
					//findings is ArrayList of ArrayList of String
					ArrayList<Findings> healthy = new ArrayList<Findings>();
					healthy.add(Findings.Healthy);
					this.findings.add(healthy);
					this.roi.add(null);
				}
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (XPathExpressionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		System.out.println("ReadCSVdata - Done.");
	}
	
	/**
	 * /path to scvFile and imgFolder are setted
	 * @param imgFolder
	 * @param csvF
	 */
	public ReadMammograms(String imgFolder, String xmlDir, String csvF) {
		this.imgDir = imgFolder;
		this.xmlDir = xmlDir;
		this.csvFile = csvF;
	}
}