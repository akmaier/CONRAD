package edu.stanford.rsl.tutorial.fan;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;

/**
 * Example that mirrors {@link FanBeamReconstructionExample} but uses the analytic
 * projector {@link FanBeamAnalyticProjector2D}. It builds a small analytic scene (a
 * water body Cylinder + one bone insert Cylinder), projects it into per-material
 * path-length sinograms, forms a polychromatic EID line-integral sinogram using
 * CONRAD's {@link PolychromaticXRaySpectrum} and {@link Material#getAttenuation},
 * and reconstructs through CosineFilter -&gt; RamLakKernel -&gt; FanBeamBackprojector2D.
 * <p>
 * Headless-friendly: display is only performed when {@code -Dshow=true} is passed
 * (or a display is otherwise wanted), so it can run without a screen.
 *
 * @author Andreas Maier
 */
public class FanBeamAnalyticProjectionExample {

	public static void main(String[] args) {
		boolean show = Boolean.getBoolean("show");

		// fan geometry (matches the SPION conrad_ct.fan_geometry defaults)
		double focalLength = 750.0;
		double maxT = 256.0;
		double deltaT = 1.0;
		double maxBeta = 2.0 * Math.PI;
		double deltaBeta = maxBeta / 360.0;

		// image params for reconstruction
		int imgSzXMM = 256, imgSzYMM = 256;

		// ---- build an analytic scene: water body + bone insert ----
		double bodyRadius = 80.0;   // mm
		double insertRadius = 12.5; // mm
		double insertCx = 40.0, insertCy = 0.0;
		double cylHeight = 200.0;

		PrioritizableScene scene = new PrioritizableScene();
		Material water = MaterialsDB.getMaterialWithName("water");
		Material bone = MaterialsDB.getMaterialWithName("bone");

		// body first (lowest priority), insert added afterwards (higher priority -> overrides)
		scene.add(makeCylinder(bodyRadius, cylHeight, 0.0, 0.0, water));
		scene.add(makeCylinder(insertRadius, cylHeight, insertCx, insertCy, bone));

		if (show) {
			new ImageJ();
		}

		// ---- analytic projection: per-material path lengths [mm] ----
		FanBeamAnalyticProjector2D projector = new FanBeamAnalyticProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
		MultiChannelGrid2D materialSino = projector.projectRayDrivenMaterials(scene);

		System.out.println("Materials (channel order):");
		for (int c = 0; c < projector.getMaterials().size(); c++) {
			System.out.println("  channel " + c + ": " + projector.getMaterials().get(c).getName());
		}

		if (show) {
			for (int c = 0; c < materialSino.getNumberOfChannels(); c++) {
				materialSino.getChannel(c).clone().show("path length [mm]: " + materialSino.getChannelNames()[c]);
			}
		}

		// ---- EID line-integral sinogram via CONRAD polychromatic spectrum ----
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum();
		double[] energies = spectrum.getPhotonEnergies();
		double[] weights = spectrum.getPhotonFlux();
		Grid2D eidSino = projector.combineEID(materialSino, energies, weights);
		if (show) {
			eidSino.clone().show("EID sinogram");
		}

		// ---- reconstruct: CosineFilter -> RamLakKernel -> FanBeamBackprojector2D ----
		CosineFilter cKern = new CosineFilter(focalLength, maxT, deltaT);
		RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
		for (int theta = 0; theta < eidSino.getSize()[1]; ++theta) {
			cKern.applyToGrid(eidSino.getSubGrid(theta));
		}
		for (int theta = 0; theta < eidSino.getSize()[1]; ++theta) {
			ramLak.applyToGrid(eidSino.getSubGrid(theta));
		}

		FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSzXMM, imgSzYMM);
		Grid2D reco = fbp.backprojectPixelDriven(eidSino);
		if (show) {
			reco.show("Analytic-projector FBP reconstruction");
		}

		// ---- quick central-ray sanity print ----
		int centerT = (int) (maxT / deltaT) / 2;
		double waterL = 0.0, boneL = 0.0, total = 0.0;
		for (int c = 0; c < materialSino.getNumberOfChannels(); c++) {
			double v = materialSino.getPixelValue(centerT, 0, c);
			total += v;
			if ("water".equals(materialSino.getChannelNames()[c])) {
				waterL = v;
			}
			if ("bone".equals(materialSino.getChannelNames()[c])) {
				boneL = v;
			}
		}
		System.out.println(String.format(
				"Central ray (beta=0, t=%d): water=%.3f mm  bone=%.3f mm  total=%.3f mm (body chord=%.3f mm)",
				centerT, waterL, boneL, total, 2.0 * bodyRadius));
		System.out.println("Done.");
	}

	private static PhysicalObject makeCylinder(double radius, double height, double cx, double cy, Material material) {
		Cylinder cyl = new Cylinder(radius, radius, height);
		if (cx != 0.0 || cy != 0.0) {
			PointND pt = new PointND(cx, cy, 0.0);
			cyl.applyTransform(new Translation(pt.getAbstractVector()));
		}
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(material);
		po.setShape(cyl);
		return po;
	}
}
/*
 * Copyright (C) 2010-2024 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
