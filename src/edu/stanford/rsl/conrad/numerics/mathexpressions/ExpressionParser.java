package edu.stanford.rsl.conrad.numerics.mathexpressions;


/**
 * Class to parse arbitrary well formed mathematical expression
 * This class also contains a custom scanner class
 * @author Rotimi X Ojo
 */

public class ExpressionParser {
	private String expression;
	
	public ExpressionParser(String expression){
		this.expression =  preProcessExpression(expression).trim();
		if(!isWellFormed()){
			throw new RuntimeException("Expression is not well formed");
		}
	}
	
	private String preProcessExpression(String input) {
		input = input.replace("+", " + ");
		input = input.replace("(", " ( ");
		input = input.replace(")", " ) ");
		input = input.replace("-", " - ");
		input = input.replace("*", " * ");
		input = input.replace("/", " / ");
		return input;
	}
	
	/**
	 * Returns a parse tree equivalent for supplied mathematical expression
	 * @return the expression
	 */
	public AbstractMathExpression getMathExpression(){
		ExpressionScanner scanner = new ExpressionScanner(expression);
		String token = scanner.nextToken().trim();
		AbstractMathExpression leftexp  = getExp(token);;		
		
		while (scanner.hasNextToken()) {
			String operator  = scanner.nextToken();
			if (operator.equals("*") || operator.equals("/")) {
				AbstractMathExpression rightexp = getExp(scanner.nextToken());
				leftexp = new CompoundExpression(leftexp,rightexp,operator.charAt(0));				
			} else if (operator.equals("+") || operator.equals("-")) {
				AbstractMathExpression rightexp = new ExpressionParser(scanner.nextLine()).getMathExpression();
				leftexp = new CompoundExpression(leftexp,rightexp,operator.charAt(0));
			}
		}
		return leftexp;
	}

	/**'
	 * Determines if token contains a single mathematical operation
	 * @param token expression to be parsed
	 * @return true if expression requires a single operation.
	 */
	private boolean isSingleOperation(String token) {	
		int numoperators = numOperators(token);
		
		if(numoperators > 1 ){
			return false;
		}
		
		if(numoperators == 1 && token.charAt(0) != '-'&& token.charAt(0) != '+'){
			return false;
		}
		
		return true;
	}
	
	/**
	 * Counts the number of operators in a given token
	 * @param token expression to be parsed
	 * @return number of supported operators in a given token.
	 */
	private int numOperators(String token) {
		int count = 0;
		for(int i =0;i < token.length();i++){
			char curr = token.charAt(i);
			if(curr == '-' || curr == '+' || curr == '/'|| curr == '*'){
				if(i < 1 || curr == '/'|| curr == '*' || (i > 1 && token.charAt(i-2)!= 'E')){
					count++;
				}				
			}
		}		
		
		return count;
	}
	
	/**
	 * Converts the given token to an expression tree.
	 * @param currToken: expression to be parsed
	 * @return An expression tree representing the input expression.
	 */
	private AbstractMathExpression getExp(String currToken) {				
		currToken = currToken.trim();
		if(!isSingleOperation(currToken) && !isFormula(currToken)){
			 return new MathExpression(currToken);
		}
		String firstChar = currToken.substring(0,1);
		if(firstChar.equals("-")){
			AbstractMathExpression leftExp = new RealExpression(-1);
			AbstractMathExpression rightExp = getExp(currToken.substring(currToken.indexOf("-")+1));
			return new CompoundExpression(leftExp, rightExp, '*');
		}else if(firstChar.equals("+")){
			return getExp(currToken.substring(currToken.indexOf("+")+1));
		}else if(firstChar.matches("\\p{Digit}+")){	
			return new RealExpression(Double.parseDouble(currToken.replace(" ", "")));		
		}else if(firstChar.matches("\\p{Alpha}+") && !currToken.contains("(")){
			return new IdentifierExpression(currToken);
		}else if(isFormula(currToken)){
			AbstractMathExpression input = new MathExpression(currToken.substring(currToken.indexOf("(")+1,currToken.length()-1));
			return new FunctionExpression(currToken.substring(0,currToken.indexOf("(")),input);
		}else if(currToken.charAt(0)=='('  && currToken.charAt(currToken.length()-1) == ')'){			
			return new MathExpression(currToken);
		}
		throw new UnsupportedOperationException("invalid token " + currToken);
	}
	
	
	/**
	 * Determines if input token is an expression expression
	 * @param currToken
	 * @return: true if token is an expression.
	 */
	private boolean isFormula(String currToken) {
		String firstChar = currToken.substring(0,1);
		return firstChar.matches("\\p{Alpha}+") && currToken.contains("(");
	}
	
	/**
	 * Determines if an expression is well-formed by counting braces.
	 * @return
	 */
	private boolean isWellFormed() {
		runBracketCheck();	
		return true;
	}
	
	/**
	 * Determines if the braces delineating expressions are balanced and well formed.
	 */
	public void  runBracketCheck(){
		int count = 0;
		for(int i = 0; i < expression.length(); i++){
			if(expression.charAt(i)== '('){
				count++;
			}else if(expression.charAt(i)==')'){
				count--;
			}
			if(count < 0){
				throw new RuntimeException("unbalanced Bracketing at index: " + i + " of " + expression);
			}
		}
		
		if(count != 0){
			throw new RuntimeException("unbalanced Brackets");
		}		
	}
	
	/**
	 * Class to retrieve standalone mathematical expressions.
	 * @author Rotimi X Ojo
	 */
	private class ExpressionScanner{
		private String expression;
		private String lastToken = "*";
		
		
		public ExpressionScanner(String expression) {
			this.expression = expression.trim();
			this.expression = removeOutterBrackets(this.expression);
		}

		private String removeOutterBrackets(String exp) {	
			int expLength = exp.length();
            if(isSingleGroup(exp)){
				String buff = exp.substring(1,expLength -1);
				if(isSingleGroup(buff)){
					return removeOutterBrackets(buff);
				}else{
					if(buff.indexOf("(") > buff.indexOf(")")){
						return exp;
					}
					return buff.trim();
				}            	
			}
            return exp.trim();
		}

		private boolean isSingleGroup(String exp) {
			int expLength = exp.length();
			if(exp.charAt(0)=='(' &&  exp.charAt(expLength-1) ==')'){
				int count = 0;
				for(int i = 0; i < expLength; i++){
					if(exp.charAt(i)== '('){
						count++;
					}else if(exp.charAt(i)==')'){
						count--;
					}
					if(count == 0 && i!=expLength-1){
						return false;
					}
				}
				return true;
			}
			return false;
		}

		public String nextToken() {
			
			expression = expression.trim();
		
			String firstChar = expression.substring(0,1);
			if(expression.charAt(0) == '('){
				lastToken = getEnclosedToken();				
				return lastToken;
			}else if(isOperator(firstChar)){
				lastToken = getOperatorOrNegation(firstChar);				
				return lastToken;
			}else if(firstChar.matches("\\p{Alpha}+")){
				lastToken = getIdentifierOrFormula();				
				return lastToken;
			}else{
				if(expression.contains("E - ") || expression.contains("E + ")){
				    lastToken = expression;
				    expression = null;
				}else{
					lastToken = getRealExpression();
				}
				return lastToken;
			}
			
		}

		private String getRealExpression() {
			String token = "";
			if(expression.contains(" ")){
				token = expression.substring(0,expression.indexOf(" "));
				expression = expression.substring(expression.indexOf(" ")).trim();
			}else{
				token = expression;
				expression = null;
			}
			return token;
		}

		private String getEnclosedToken() {
			String token = "";
			int bracketcloseIndex = getBracketCloseIndex();
			if (bracketcloseIndex == expression.length() - 1) {
				token = expression.substring(1, bracketcloseIndex).trim();
				expression = null;
			} else {
				token  = expression.substring(1, bracketcloseIndex).trim();
				expression = expression.substring(bracketcloseIndex+1).trim();
			}
			return token;
		}

		private String getOperatorOrNegation(String firstChar) {
			String token = expression.substring(0, 1);
			expression = expression.substring(1).trim();
			if (isNegation(firstChar)) {
				lastToken = token;
				token = token + nextToken();
			}
			return token;
		}

		private String getIdentifierOrFormula() {
			String token = "";
			int openbracIndex = expression.indexOf('(');
			if(openbracIndex != -1){
				if(!isIdentifier(expression.substring(0,openbracIndex))){
					token = getFormula();						
				}
			}else{
				token = expression.substring(0,expression.indexOf(" "));
				expression = expression.substring(expression.indexOf(" ")).trim();
			}
			return token;
		}

		private String getFormula() {
			String token = "";
			int closebrackindex = getBracketCloseIndex();						
			if(closebrackindex == expression.length()-1){
				token = expression;
				expression = null;
			}else{
				token = expression.substring(0,closebrackindex + 1);
				expression = expression.substring(closebrackindex + 1).trim();
			}
			return token;
		}	
		
		
		

		private boolean isIdentifier(String buff) {
			return buff.contains("-")|| buff.contains("+")||buff.contains("*")||buff.contains("/");
		}

		private boolean isNegation(String token) {
			return (token.equals("-") && isOperator(lastToken)) || (token.equals("+") && isOperator(lastToken));
		}

		private boolean isOperator(String token) {
			return token.equals("-")|| token.equals("+")||token.equals("*")||token.equals("/");
		}

		private int getBracketCloseIndex() {
			int count = 0;
			int index = 0;
			boolean nohit = false;
			do{
				nohit = false;
				if(expression.charAt(index)== '('){
					count++;
				}else if(expression.charAt(index)==')'){
					count--;
				}else{
					nohit =true;
				}
				index++;
			}while(count > 0 || nohit);			
			return index - 1;
		}
		
		/**
		 * Checks if scanner has more tokens
		 * @return true if scanner has more tokens.
		 */
		public boolean hasNextToken() {
			if(expression != null ){
				return true;
			}
			return false;
		}
		
		/**
		 * Returns the next line of expressions
		 * @return
		 */
		public String nextLine() {
			String buff = expression;
			expression = null;
			return buff;
		}
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/