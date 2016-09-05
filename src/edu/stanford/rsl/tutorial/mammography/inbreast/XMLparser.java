package edu.stanford.rsl.tutorial.mammography.inbreast;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.awt.Polygon;

import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import edu.stanford.rsl.tutorial.mammography.Mammogram;
import edu.stanford.rsl.tutorial.mammography.Mammogram.Findings;
import ij.gui.PolygonRoi;
import ij.gui.Roi;

public class XMLparser {

	String[] rawData = null;
	ArrayList<Roi> rois = null;
	ArrayList<Findings> findings = null;

	public static void main(String[] args) throws XPathExpressionException,
			FileNotFoundException {

		String xmldat = "D:/Data/INbreast/ALLXML/53586960.xml";
		XMLparser parse = new XMLparser();
		parse.readXml(xmldat);
		
		System.out.println("debug point");
	}

	public void readXml(String file) throws XPathExpressionException,
			FileNotFoundException {
		this.rawData = parseXmlFile(file);
		decodeRawData();
	}

	private void decodeRawData() {
		ArrayList<Roi> rois = new ArrayList<Roi>();
		ArrayList<Findings> findings = new ArrayList<Findings>();
		
		int numComponents = Integer.parseInt(rawData[0]);
		
		int pos = 1;
		for(int i = 1; i < numComponents; i++){
			String fndg = rawData[pos];
			for(Findings f : Findings.values()){
				String[] vals = f.getValue();
				// make sure its not the "all" value
				if(vals.length > 1){
					break;
				}
				if(fndg.equals(f.getValue()[0])){
					findings.add(f);
					break;
				}
			}
			pos++;
			int numPolygonPoints = Integer.parseInt(rawData[pos]);
			pos++;
			int[] x = new int[numPolygonPoints];
			int[] y = new int[numPolygonPoints];
			for(int j = 0; j < numPolygonPoints; j++){
				String coordPair = rawData[pos];
				String[] split = coordPair.split(",");
				x[j] = (int)Double.parseDouble(split[0].substring(1));
				y[j] = (int)Double.parseDouble(split[1].substring(0,split[1].length()-1));
				pos++;
			}
			Polygon poly = new Polygon(x,y,numPolygonPoints);
			PolygonRoi roi = new PolygonRoi(poly, Roi.POLYGON);
			rois.add(roi);
		}
		this.rois = rois;
		this.findings = findings;		
	}
	
	public ArrayList<Roi> getRois(){
		return this.rois;
	}

	public ArrayList<Findings> getFindings(){
		return this.findings;
	}
	private String[] parseXmlFile(String pfad) throws XPathExpressionException,
			FileNotFoundException {

		try {
			InputSource xml = new InputSource(new FileInputStream(pfad));
			// Hier passiert Magie! Mit anderen Worten ein kleines Wunder!
			String expression = "/plist/dict/array/dict/array/dict/key[text()=\"Name\"]/following-sibling::*[1]|" // Name
																													// des
																													// Finding
					+ "/plist/dict/array/dict/array/dict/key[text()=\"Point_px\"]/following-sibling::*[1]/child::*|" // Die
																														// XY
																														// Koordinaten
					+ "/plist/dict/array/dict/integer[2]|" // Anzahl der ROIs
					+ "/plist/dict/array/dict/array/dict/key[text()=\"NumberOfPoints\"]/following-sibling::*[1]"; // Anzahl
																													// der
																													// XY
																													// Koordinaten

			NodeList ding = (NodeList) XPathFactory.newInstance().newXPath()
					.evaluate(expression, xml, XPathConstants.NODESET);

			String[] array = new String[ding.getLength()];

			for (int i = 0; i < ding.getLength(); i++) {
				array[i] = ding.item(i).getTextContent();
			}

			return array;

		} catch (XPathExpressionException e) {
			// TODO Auto-generated catch block
			throw e;
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			throw e;
		}
	}
}