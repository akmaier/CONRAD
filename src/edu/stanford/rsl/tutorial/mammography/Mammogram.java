package edu.stanford.rsl.tutorial.mammography;

import ij.gui.Roi;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class Mammogram {

	private Grid2D image = null;

	private ArrayList<Findings> findings = null;
	private ArrayList<Roi> rois = null;
	private int biradsScore = 0;
	private BreastDensityACR density = BreastDensityACR.Unknown;
	private SideOfBody side = SideOfBody.Unknown;
	private MammographyView view = MammographyView.Unknown;

	// private Findings finding = Findings.Unnamed;

	public Mammogram(Grid2D img, SideOfBody sob, MammographyView vw, BreastDensityACR dnsty, ArrayList<Findings> fndgs,
			ArrayList<Roi> rs, int brdsScr) {
		this.image = img;
		this.findings = fndgs;
		this.rois = rs;
		this.density = dnsty;
		this.side = sob;
		this.view = vw;
		this.biradsScore = brdsScr;
	}

	public static void main(String[] args) {

	}

	public BreastDensityACR getDensity() {
		return density;
	}

	public void setDensity(BreastDensityACR density) {
		this.density = density;
	}

	public MammographyView getView() {
		return view;
	}

	public void setView(MammographyView view) {
		this.view = view;
	}

	public SideOfBody getSideOfBody() {
		return side;
	}

	public void setSideOfBody(SideOfBody side) {
		this.side = side;
	}

	public int getBiradsScore() {
		return biradsScore;
	}

	public void setBiradsScore(int biradsScore) {
		this.biradsScore = biradsScore;
	}

	public ArrayList<Findings> getFindings() {
		return findings;
	}

	public void setFindings(ArrayList<Findings> findings) {
		this.findings = findings;
	}

	public ArrayList<Roi> getRois() {
		return rois;
	}

	public void setRois(ArrayList<Roi> rois) {
		this.rois = rois;
	}

	public Grid2D getImage() {
		return image;
	}

	public void setImage(Grid2D image) {
		this.image = image;
	}

	public enum MammographyView {
		ML(new String[] 	{ "MLO"}), 
		CC(new String[] 	{ "CC" }), 
		Both(new String[] 	{ "MLO", "CC" }), 
		Unknown(new String[]{ "?" });

		private String[] keys = null;

		MammographyView(String[] k) {
			this.keys = k;
		}

		public String[] getValue() {
			return this.keys;
		}
	}

	public enum Findings {
		Asymmetry	(new String[] { "Asymmetry" }), 
		Calcification(new String[] { "Calcification" }), 
		Cluster		(new String[] { "Cluster" }), 
		Mass		(new String[] { "Mass" }), 
		Distortion	(new String[] { "Distortion" }), 
		Spiculated	(new String[] { "Spiculated region" }), 
		Pectoral	(new String[] { "Pectoral muscle" }), 
		Unnamed		(new String[] { "Unnamed" }), 
		Healthy		(new String[] { "healthy" }), 
		All			(new String[] { "Asymmetry",	"Calcification", "Cluster",
				"Mass", "Distortion", "Spiculated region", 
				"Pectoral muscle", "Unnamed","healthy" }), 
		WithFindingsOnly(new String[] { "Asymmetry", "Calcification", 
				"Cluster", "Mass","Distortion", "Spiculated region", 
				"Pectoral muscle", "Unnamed", });

		private String[] keys = null;

		Findings(String[] k) {
			this.keys = k;
		}

		public String[] getValue() {
			return this.keys;
		}

	}

	public enum SideOfBody {
		Left (new String[] { "L" }), 
		Right(new String[] { "R" }), 
		Both (new String[] { "L", "R" }), 
		Unknown(new String[] { "?" });

		private String[] keys = null;

		SideOfBody(String[] k) {
			this.keys = k;
		}

		public String[] getValue() {
			return this.keys;
		}
	}

	public enum BreastDensityACR {
		Fatty(1), 
		FattyGlandular1(2), 
		FattyGlandular2(3), 
		Glandular(4), 
		Unknown(-1);

		private int flag = -1;

		BreastDensityACR(int f) {
			this.flag = f;
		}

		public int getACRValue() {
			return this.flag;
		}
	}

}
