package openproof.plato.model;

import java.io.UnsupportedEncodingException;

import org.apache.log4j.Logger;

import openproof.fol.representation.OPFormula;
import openproof.fol.representation.parser.ParseException;
import openproof.fol.representation.parser.StringToFormulaParser;

public class Answer {
	private String text;
	private OPFormula formula;

	public Answer(String text) {
		this.text = text;
		try {
			this.setFormula(StringToFormulaParser.getFormula(fromUnicode(text)));
		} catch (UnsupportedEncodingException e) {
			Logger.getLogger(Answer.class).warn("", e);
		} catch (ParseException e) {
		}
	}
	
	private String fromUnicode(String unicodeFormulaString) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < unicodeFormulaString.length(); i++) {
			switch(unicodeFormulaString.charAt(i)) {
			 case '\u2260': sb.append('#'); break;	// not equal --> #
			 case '\u2192': sb.append('$'); break;	// implication --> $
			 case '\u2194': sb.append('%'); break;	// biconditional --> %
			 case '\u2227': sb.append('&'); break;	// conjunction --> &
			 case '\u2203': sb.append('/'); break;	// existential quantifier --> /
			 case '\u2200': sb.append('@'); break;	// universal quantifier --> @ 
			 case '\u22A5': sb.append('^'); break;	// bottom --> ^
			 case '\u2228': sb.append('|'); break;	// disjunction --> |
			 case '\u00AC': sb.append('~'); break;	// negation --> ~
			 default: sb.append(unicodeFormulaString.charAt(i));
			}
		}
		return sb.toString();
	}

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}

	public OPFormula getFormula() {
		return formula;
	}

	public void setFormula(OPFormula formula) {
		this.formula = formula;
	}

	public String toString() { return this.text + " parsed as " + this.formula; }

}
