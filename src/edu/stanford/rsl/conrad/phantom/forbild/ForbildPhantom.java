package edu.stanford.rsl.conrad.phantom.forbild;

import java.io.File;

import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.AbstractScene;
import edu.stanford.rsl.conrad.utils.FileUtil;


/**
 * <p>This class creates <a href = "http://www.imp.uni-erlangen.de/forbild/english/forbild/index.htm">forbild</a> phantoms from configuration files</p>
 * 
 * 
 * @author Rotimi Ojo
 *
 */
public class ForbildPhantom extends AnalyticPhantom {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2384108143541335664L;
	private boolean isConfigured;


	public ForbildPhantom(){
		
	}
	
	public ForbildPhantom(AbstractScene scene){
		addAll(scene);
		isConfigured = true;
	}
	
	@Override
	public void configure() throws Exception {
		super.configure();
		if(!isConfigured){
			File file = new File(FileUtil.myFileChoose(".pha", false));	
			
			//File file = new File(System.getProperty("user.dir")+"/data/configfiles/forbild/ThoraxPhantom.pha");
			addAll(new ForbildParser(file).getScene());
		}

	}

	
	@Override
	public String getName() {
		return "Forbild Phantom";
	}

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/