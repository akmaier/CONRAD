package edu.stanford.rsl.tutorial.iterative;

import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;

/**
 * SART reconstruction
 * 
 * @author Mario Amrehn
 * 
 */
public class SartCPU implements Sart {

	protected final Grid3D vol;
	protected final Grid3D normSino;
	protected Grid3D oProj = null;
	protected final float beta;
	protected float normFactor;

	protected NumericGridOperator gop = NumericGridOperator.getInstance();

	// -----------------------------------------
	protected boolean verbose = false;
	protected boolean debug = false;
	protected final static boolean USE_CL_FP = true;	// GPU acceleration
	protected final static boolean USE_CL_BP = true;	// GPU acceleration
	// -----------------------------------------

	public SartCPU(int[] volDims, double[] spacing, double[] origin, Grid3D oProj,
			float beta) throws Exception {
		if (null == oProj) {
			throw new Exception("SART: No projection data given");
		}
		if (1 > volDims[0] || 1 > volDims[1] || 1 > volDims[2]) {
			throw new Exception(
					"SART: Span of each dimension in the volume has to be a natural number");
		}

		// create initial volume filled with zeros
		vol = new Grid3D(volDims[0], volDims[1], volDims[2]);
		// vol.setBoundary(boundarySize);
		vol.setOrigin(origin);
		vol.setSpacing(spacing);

		this.normSino = createNormProj();
		this.oProj = oProj;
		/* calculated once for speedup */
		this.normFactor = (float) (gop.normL1(oProj) / gop.normL1(normSino));
		this.beta = beta;
	}

	public SartCPU(Grid3D initialVol, Grid3D sino, float beta) throws Exception {
		if (null == initialVol) {
			throw new Exception("SART: No initial volume given");
		}
		if (null == sino) {
			throw new Exception("SART: No projection data given");
		}
		vol = initialVol;
		normSino = createNormProj();
		this.oProj = sino;
		
		/* calculated once for speedup*/
		normFactor = (float) (gop.normL1(oProj) / gop.normL1(normSino));
		this.beta = beta;
	}

	/**
	 * @return Normalized projection data
	 */
	protected Grid3D createNormProj() {
		if (verbose)
			System.out.println("Create normalized projections");
		final Grid3D onesVol = new Grid3D(vol);
		gop.fill(onesVol, 1.0f);
		ConeBeamProjector cbp = new ConeBeamProjector();
		Grid3D sino = USE_CL_FP ? cbp.projectRayDrivenCL(onesVol) : cbp
				.projectPixelDriven(onesVol);
		
		if(verbose)
			sino.show("sinoOfNormProjs");
		// prevent div by zero
		float min = gop.min(sino);
		//sino.show("Sino of norm projection");
		if (0 >= min)
			gop.addBy(sino, -min + 0.1f);
		return sino;
	}

	public void iterate() throws Exception {
		this.iterate(1);
	}

	public void iterate(final int iter) throws Exception {
		ConeBeamProjector cbp = new ConeBeamProjector();
		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		int numProjs = cbp.getMaxProjections();
		for (int i = 0; i < iter; ++i) {

			boolean[] projIsUsed = new boolean[numProjs]; // default: false
			int p = 0; // current projection index

			for (int n = 0; n < numProjs; ++n) {
				/* edit/init data structures */
				projIsUsed[p] = true;
				Grid2D sino = USE_CL_FP ? cbp.projectRayDrivenCL(vol, p) : cbp
						.projectPixelDriven(vol, p);
				sino = gop.transpose(sino);
				if(debug && 0 < gop.normL1(vol)){
					Grid3D sinoTest = USE_CL_FP ? cbp.projectRayDrivenCL(vol) : cbp
							.projectPixelDriven(vol);
					Grid2D sinoTestP = sinoTest.getSubGrid(p);
					sinoTest.show("sinoCL-Test");
					sinoTestP.show("sinoCL-Test-Proj:" + p);
					Grid2D s = new Grid2D(sinoTestP);
					gop.subtractBySave(s, sino);
					s.show("sinoCL-Test-Proj-Diff");
					System.out.println("Diff L1: " + gop.normL1(s));
				}
				/*
				if(verbose && 0 != gop.normL1(sino)){
					System.out.println("Some actual data found in sino!");
					if(0==p%5)
						sino.show("sino of it:" + i + " proj:" + p);
				}
				*/
				gop.fillInvalidValues(sino, 0);
				
				if (verbose) System.out.println(gop.min(sino) + ":" + gop.max(sino)); // TEST
				
				//oProj.show("oProj");
				
				Grid2D oProjP = new Grid2D(oProj.getSubGrid(p));
				Grid2D normSinoP = new Grid2D(normSino.getSubGrid(p)); // used read-only, cloning not necessary but save
				gop.multiplyBy(normSinoP, normFactor);

				/* update step */
				// NOTE: upd = (oProj - sino) ./ normSino
				
				if (verbose) reportInvalidValues(oProjP, "oProjP"); // Just in case.. 
				if (verbose) reportInvalidValues(sino, "sino"); // Just in case.. should not happen after fillInvalidValues()
				
				gop.subtractBy(oProjP, sino);
				
				if (verbose) reportInvalidValues(oProjP, "oProjP"); // Just in case.. 
				if (verbose) reportInvalidValues(normSinoP, "normSinoP"); // Just in case.. 
				
				gop.divideBySave(oProjP, normSinoP);
				Grid2D upd = oProjP;

				if (verbose) reportInvalidValues(upd, "for projection " + p);

				// NOTE: vol = vol + updBP * beta
				// upd.setOrigin(oProj.getOrigin()[0], oProj.getOrigin()[1]); //
				// needed after update?
				// upd.setSpacing(oProj.getSpacing()[0], oProj.getSpacing()[1]);
				// // needed after update?
				Grid3D updBP = USE_CL_BP ? cbbp.backprojectPixelDrivenCL(upd, p)
						: cbbp.backprojectPixelDriven(upd, p);

				if (verbose) reportInvalidValues(updBP, "updBP");
				
				if(debug){
					Grid3D updBPTest = (!USE_CL_BP) ? cbbp.backprojectPixelDrivenCL(upd, p)
							: cbbp.backprojectPixelDriven(upd, p);
					if (verbose) reportInvalidValues(updBPTest, "updBPTest");
					gop.multiplyBySave(updBPTest, 1);
					updBPTest.show("updBPTest");
					
					System.out.println("vol: " + Arrays.toString(vol.getOrigin()));
					System.out.println("updBP: " + Arrays.toString(updBP.getOrigin()));
					System.out.println("updBPTest: " + Arrays.toString(updBPTest.getOrigin()));
				}				
				
				// GridOp.addInPlace(vol, GridOp.mulInPlace(updBP, beta));
				gop.multiplyBySave(updBP, beta);
				if (verbose) reportInvalidValues(updBP, "updBP after mult");
				
				gop.addBy(vol, updBP);
				if (verbose) reportInvalidValues(vol, "vol after " + i + " SART iterations");

				/*
				 * Don't use projections with a small angle to each other
				 * subsequently
				 */
				p = (p + numProjs / 3) % numProjs;
				for (int ii = 1; projIsUsed[p] && ii < numProjs; ++ii)
					p = (p + 1) % numProjs;
			}
		}
	}

	protected void reportInvalidValues(NumericGrid upd, String msg) {
		if (null != msg && msg.length()>0)
			msg = " " + msg;
		int invalCount = gop.countInvalidElements(upd);
		if (0 < invalCount)
			System.out.println("Invalid values" + msg + ": " + invalCount);
		
		float min = gop.min(upd);
		float max = gop.max(upd);
		float normL1 = (float) gop.normL1(upd);
		System.out.println("Info\t" + msg + "\tmin=" + min + "\tmax=" + max + "\tnormL1=" + normL1);
	}

	public Grid3D getVol() {
		gop.fillInvalidValues(vol, 0);
		return new Grid3D(vol);
	}
}
