package edu.stanford.rsl.BA_Niklas;

import java.io.*; 
import java.util.*;

public class ListInFile {
	
    public static void export(List<Float> darklist, String string) { 
    	final long timeStart = System.currentTimeMillis();
    	
        PrintWriter printWriter = null; 
        try { 
            printWriter = new PrintWriter(new FileWriter(string)); 
            Iterator iter = darklist.iterator(); 
            while(iter.hasNext() ) { 
                Object o = iter.next(); 
                printWriter.println(o); 
            } 
        } catch (IOException e) { 
            e.printStackTrace(); 
        } finally { 
            if(printWriter != null) printWriter.close(); 
        }
        final long timeEnd = System.currentTimeMillis(); 
	     System.out.println("Export done in: " + (timeEnd - timeStart) + " Millisek.");
    } 
}
