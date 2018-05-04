package openproof.plato.nlgen;

import java.util.List;

import openproof.fol.representation.OPFormula;

public interface NLGeneratorFace {
	
	public abstract NLRendering render(OPFormula f, NLRenderingPolicyFace face) throws NLRenderingException;
	public abstract List<NLRendering> renderAll(OPFormula f, NLRenderingPolicyFace face) throws NLRenderingException;

}
