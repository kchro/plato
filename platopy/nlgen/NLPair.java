package openproof.plato.nlgen;

import java.util.ArrayList;

public class NLPair implements NLRendering {
	private ArrayList<String> rules;
	private String string;
	public static final String RULES = "rules";
	public static final String TRANSLATION = "sent";
	
	public NLPair() {
		rules = new ArrayList<String>();
	}
	
	public void addRule(String rule) {
		rules.add(rule);
	}
	
	public void addTranslation(String t) {
		string = t;
	}
	
	public ArrayList<String> getRules() {
		return rules;
	}
	public String getString() {
		return string;
	}
	
	public ArrayList<String> getMeta() {
		return rules;
	}
	
	public String toString() {
		return string+" - "+rules;
	}
}
