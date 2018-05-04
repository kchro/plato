package openproof.plato.nlgen;


public interface NLRenderingPolicyFace {

	public abstract void setAllowAmbiguity(Boolean unambiguous);
	public abstract Boolean getAllowAmbiguity();
		
	public abstract void setRuleSet(RuleSet rules);
	public abstract RuleSet getRuleSet();


}
