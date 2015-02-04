package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.BilateralFilter3DBlock;

public class BilateralFilter3D extends BlockWiseMultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2430924392909104202L;

	@Override
	public ImageFilteringTool clone() {
		BilateralFilter3D clone = new BilateralFilter3D();
		clone.modelBlock = modelBlock;
		clone.numBlocks = numBlocks;
		clone.blockOverlap = blockOverlap;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Bilateral Filter 3D";
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Tomasi98-BFF,\n" +
		"  author = {Tomasi, C. and Manduchi, R.},\n" +
		"  title = {Bilateral Filtering for Gray and Color Images},\n" +
		"  booktitle = {ICCV '98: Proceedings of the Sixth International Conference on Computer Vision},\n" +
		"  year = {1998},\n" +
		"  isbn = {81-7319-221-9},\n" +
		"  pages = {839-846},\n" +
		"  publisher = {IEEE Computer Society},\n" +
		"  address = {Washington, DC, USA},\n" +
		"}\n";
		return bibtex;
	}
	
	@Override
	public void configure() throws Exception{
		super.configure();
		if (modelBlock == null) {
			modelBlock = new BilateralFilter3DBlock();
		}
		modelBlock.configure();
		configured = true;
	}
	

	@Override
	public String getMedlineCitation() {
		return "Tomasi C, Maduchi R, Bilateral Filtering for Gray and Color Images. In: ICCV '98: Proceedings of the Sixth International Conference on Computer Vision, pp. 839-846, IEEE Computer Society, Washington, DC, United States 1998.";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/