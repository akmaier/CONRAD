package edu.stanford.rsl.conrad.phantom.forbild;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.AbstractScene;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.parsers.SceneFileParser;

/**
 * <p>This class adds objects defined in <a href = "http://www.imp.uni-erlangen.de/forbild/">forbild</a> format into a scene that can be used to create a phantom. </p>
 *
 * @author Rotimi X Ojo
 */
public class ForbildParser implements SceneFileParser {

	private String directoryPath;
	private ArrayList<String> objectDefs = new ArrayList<String>();
	private HashMap<String,ArrayList<String>> templatePlaceholders = new HashMap<String, ArrayList<String>>();
	private AbstractScene scene = new PrioritizableScene();

	public ForbildParser(File file) {
		this(file, null);
	}

	public ForbildParser(File file, HashMap<String,String> placeHoldersMap) {
		try {
			directoryPath = file.getParent() + "\\";
			preProcessing(new Scanner(file), placeHoldersMap);
		} catch (Exception e) {
			e.printStackTrace();
		}		
		createObjects();
	}

	private void preProcessing(Scanner scanner, HashMap<String,String> placeHoldersMap)
			throws FileNotFoundException {
		while (scanner.hasNextLine()) {
			parseLine(scanner.nextLine().trim(), placeHoldersMap);
		}
	}
	private String lastObjectName = "";
	private void parseLine(String currLine, HashMap<String,String> varMap) throws FileNotFoundException {
		if(currLine == null || currLine.isEmpty()|| (currLine.indexOf("//")==0)){
			return;
		}
		currLine=currLine.trim();
		if (currLine.substring(0, currLine.indexOf(" ")).equals("#Prototype")) {
			String name = currLine.substring(currLine.indexOf(" "),	currLine.indexOf('(')).trim();
			templatePlaceholders.put(name, getTemplatePlaceHolders(currLine));
		} else if (currLine.substring(0, currLine.indexOf(" ")).equals(	"Phantom")) {
			scene.setName(currLine.substring(currLine.indexOf('"') + 1, currLine.length()-1));
		} else if(currLine.contains("Object ")){
			lastObjectName = currLine.substring(currLine.indexOf(" ") + 1).trim();
		}else if (currLine.charAt(0) == '{') {	
            currLine = lastObjectName + ";" + currLine.substring(1,currLine.length()-1);
			objectDefs.add(replacePlaceHolders(currLine,varMap));
		} else if (currLine.substring(0, currLine.indexOf(" ")).equals("#include")
				&& currLine.contains(".pha")) {
			String filename = currLine.substring(currLine.indexOf('"')+1,	currLine.length() - 1);
			scene.addAll((new ForbildParser(new File(directoryPath + filename))	.getScene()));
		} else if (currLine.substring(0, currLine.indexOf(" ")).equals("func")) {
			String name = currLine.substring(currLine.indexOf(" "),	currLine.indexOf('(')).trim();
			String fileName = name + ".pha";
			HashMap<String, String> buff = new HashMap<String, String>();
			ArrayList<String> placeHolders = templatePlaceholders.get(name);
			ArrayList<String> parameters = getFunctionParameters(currLine);
			for(int i =0; i < placeHolders.size();i++){
				buff.put(placeHolders.get(i), parameters.get(i));
			}
			scene.addAll(new ForbildParser(new File(directoryPath + fileName.trim()),buff).getScene());
		}
	}

	private ArrayList<String> getTemplatePlaceHolders(String currLine) {
		return getFunctionParameters(currLine);
	}

	private void createObjects() {
		Iterator<String> it = objectDefs.iterator();
		Scanner sc;
		int textLine = 0;
		while (it.hasNext()) {
			String line = it.next();
			textLine++;
			PhysicalObject basicobj = new PhysicalObject();
			if(line.contains("[") && line.contains("]")){
				try{
					AbstractShape shape = ForbildShapeFactory.getShape(line.substring(line.indexOf("["),line.indexOf("]")+ 1));
					if (shape== null){
						throw new RuntimeException("Error in Parsing");
					}
					basicobj.setShape(shape);
				}catch (Exception e) {
					//e.printStackTrace();
					throw new RuntimeException("Parsing Error at line: " + textLine + "\n" + line);
				}
				basicobj.setNameString(line.substring(line.indexOf("[")+1, line.indexOf(":")));
			}
			String nline = line.substring(line.indexOf("]") + 1);
			sc = new Scanner(nline.substring(1).trim());
			sc.useDelimiter(";");		
			
			
			while(sc.hasNext()){
				String property = sc.next();
				if(property.toLowerCase().contains("rho")){
					try{
						basicobj.setMaterial(MaterialsDB.getMaterial(property.substring(property.indexOf("=")+1)));
					}catch (Exception e) {
						throw new RuntimeException("Parsing Error at line: " + textLine + "\n" + line);
					}
				}
			}
			scene.add(basicobj);			
		}
	}

	private String replacePlaceHolders(String currLine, HashMap<String, String> varMap) {
		if (varMap == null || varMap.size() == 0) {
			return currLine;
		}
		Iterator<String> pit = varMap.keySet().iterator();
		while (pit.hasNext()) {
			String placeHolder = pit.next();
			String geometryDef = currLine.substring(0,currLine.indexOf(']') + 1);
			currLine = geometryDef.replace(placeHolder, varMap.get(placeHolder))+ currLine.substring(currLine.indexOf(']') + 1);
		}
		return currLine;
	}

	private ArrayList<String> getFunctionParameters(String functionCall) {
		ArrayList<String> param = new ArrayList<String>(4);
		String parameters = functionCall.substring(functionCall.indexOf('(') +  1, functionCall.indexOf(')'));
		Scanner temp = new Scanner(parameters);
		temp.useDelimiter(",");
		while (temp.hasNext()) {
			param.add(temp.next().trim());
		}
		return param;
	}

	@Override
	public AbstractScene getScene() {
		return scene;
	}


}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/