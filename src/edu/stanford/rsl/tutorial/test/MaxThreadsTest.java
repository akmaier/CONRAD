package edu.stanford.rsl.tutorial.test;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class MaxThreadsTest {

	public static void main(String[] args){
		
		// this call should throw an exception
		int maxThreads = CONRAD.getNumberOfThreads();
				
		// this call should retun the number of threads defined
        CONRAD.setup();
        // let us check if the number is defined
        // if it isn't we'll set it manually
        if(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS) == null){
        	Configuration.getGlobalConfiguration().setRegistryEntry(RegKeys.MAX_THREADS, String.valueOf(1));
        }
        
        maxThreads = CONRAD.getNumberOfThreads();
        System.out.println("Number of Threads is: "+maxThreads);
	}
	
}
