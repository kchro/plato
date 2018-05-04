package openproof.plato.model;

import java.io.UnsupportedEncodingException;
import java.net.ConnectException;
import java.util.List;

import openproof.fol.representation.OPBiconditional;
import openproof.fol.representation.OPFormula;
import openproof.fol.representation.parser.ParseException;
import openproof.gentzen.Prover;
import openproof.plato.folgen.FOLGenerationPolicyFace;
import openproof.plato.folgen.FOLGeneratorFace;
import openproof.plato.nlgen.NLGeneratorFace;
import openproof.plato.nlgen.NLRendering;
import openproof.plato.nlgen.NLRenderingException;
import openproof.plato.nlgen.NLRenderingPolicyFace;

import org.apache.log4j.Logger;

public class Problem {
	
	private OPFormula problemFormula;
	private String words;
	private FOLGenerationPolicyFace folgenPolicy;
	private Long timeToGenerate;
	private Integer problemId;
	private List<NLRendering> alternatives;
	private int promptindex = 0;

	private Integer attempts = 0;
	private Integer reprompts = 0;


	public Problem(FOLGeneratorFace folgen, NLGeneratorFace nlgen, FOLGenerationPolicyFace folGenerationPolicy, NLRenderingPolicyFace nlRenderingPolicy) throws NLRenderingException {
		this.setProblemId(randomInRange(0, 10000000));
		generate(folgen, folGenerationPolicy, nlgen, nlRenderingPolicy);
	}

	private void generate(FOLGeneratorFace folgen, FOLGenerationPolicyFace folgenPolicy, NLGeneratorFace nlgen, NLRenderingPolicyFace nlgenPolicy) throws NLRenderingException {
		try {
			
			this.folgenPolicy = folgenPolicy;
			this.problemFormula = folgen.generateFOL(this.folgenPolicy);
			
			long start = System.currentTimeMillis();
			this.alternatives = nlgen.renderAll(this.problemFormula, nlgenPolicy);
			this.timeToGenerate = new Long(System.currentTimeMillis()-start);
			
			if (null == this.alternatives || 0 == this.alternatives.size() ) {
				throw new NLRenderingException("No NL sentences generated for FOL: "+this.problemFormula);
			}
			
			this.words = this.alternatives.get(promptindex++).getString();
			
		} catch (NLRenderingException e) {
			if (e.getCause() instanceof ConnectException) {
				throw e;
			}
			Logger.getLogger(Problem.class).error(e);
			//try again.
			generate(folgen, folgenPolicy, nlgen, nlgenPolicy);
		}
	}
	
	public Long getTimeToGenerate() { return timeToGenerate; }

	public OPFormula getFormula() {
		return problemFormula;
	}

	public void setFormula(OPFormula formula) {
		this.problemFormula = formula;
	}

	public String getWords() {
		return words;
	}

	public void setWords(String words) {
		this.words = words;
	}

	public ResultType isCorrectAnswer(Answer answer)  {
		OPFormula answerFormula = answer.getFormula();
		if (null == answerFormula) return ResultType.ILL_FORMED;
		if (!answerFormula.isInVocabulary(this.folgenPolicy.getVocabulary())) return ResultType.OUT_OF_VOCABULARY;
		try {
			OPFormula toProve = new OPBiconditional(this.problemFormula, answerFormula, this.problemFormula.getSymbolTable());

			Prover p = new Prover(toProve, Prover.FO);
			Object currentResult = p.prove();

			return currentResult instanceof Integer && Prover.VALID == ((Integer) currentResult).intValue() ? ResultType.CORRECT : ResultType.INCORRECT;
		} catch (UnsupportedEncodingException e) {
			return null; // can;t happen
		} catch (ParseException e) {
			return null;
		} catch (Exception e) {
			Logger.getLogger(Problem.class).warn("isCorrectAnswer", e);
			return null;
		}
	}
	
	public String toString() { return this.problemFormula + " presented as \"" + this.words + "\""; }

	public Integer getProblemId() {
		return problemId;
	}

	private void setProblemId(Integer problemId) {
		this.problemId = problemId;
	}

	public void reprompt() {
		setWords(this.alternatives.get(promptindex++).getString());
		incrementReprompts();
	}

	public boolean canReprompt() {
		return promptindex < alternatives.size();
	}
	
	public List<NLRendering> getAlternatives() {
		return this.alternatives;
	}
	
	public NLRendering getRendering() {
		return alternatives.get(promptindex);
	}

	public static int randomInRange(int min, int max) {
		return min + (int)(Math.random() * ((max - min) + 1));
	}

	public boolean isCurrentPrompt(Integer index) {
		return index.intValue()+1 == promptindex;
	}


	
	public boolean attempted() {
		return this.attempts != 0;
	}

	public void incrementAttempts() {
		this.attempts++;		
	}
	
	public boolean reprompted() {
		return this.reprompts != 0;
	}
	
	public void incrementReprompts() {
		this.reprompts ++;
	}
	
	public boolean clean() {
		return this.attempts == 0 && this.reprompts == 0;
	}
	

}
