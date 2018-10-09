package edu.stanford.rsl.conrad.pyconrad;

import java.lang.Object;
import java.lang.RuntimeException;
import edu.stanford.rsl.conrad.pyconrad.PythonCallback;


public class PyConrad {
	
	private boolean autoConvertConradGrids = true;
	private boolean throwIfPythonIsNotAvailable = true;
	private static PythonCallback pythonCallback; // this is set by pyconrad

	
	public PyConrad()
	{
		
	}
	
	public void importModule(String module)
	{
		pythonCallback.importModule(this, module);
	}
	
	public Object eval(String code, Object... args)
	{
		if(pythonCallback != null) {
			return pythonCallback.eval(this, code, args);
		} else if(throwIfPythonIsNotAvailable) {
			throw new RuntimeException("JVM must be launched by pyconrad!\n" +
					"Use \"pyconrad_run\" in your shell to launch Java by pyconrad.");
		} else {
			return null;
		}
	}
	
	public void exec(String code, Object... args)
	{
		if(pythonCallback != null) {
			pythonCallback.exec(this, code, args);
		} else if(throwIfPythonIsNotAvailable) {
			throw new RuntimeException("JVM must be launched by pyconrad!\n" +
					"Use \"pyconrad_run\" in your shell to launch Java by pyconrad.");
		}
	}

	public boolean getAutoConvertConardGrids() {
		return autoConvertConradGrids;
	}

	public void setAutoConvertConardGrids(boolean autoConvertConardGrids) {
		this.autoConvertConradGrids = autoConvertConardGrids;
	}
	
	public boolean isPyConradAvailable() 
	{
		return (pythonCallback != null);
	}

	public boolean getThrowIfPythonIsNotAvailable() {
		return throwIfPythonIsNotAvailable;
	}

	public void setThrowIfPythonIsNotAvailable(boolean throwIfPythonIsNotAvailable) {
		this.throwIfPythonIsNotAvailable = throwIfPythonIsNotAvailable;
	}
}
