package openproof.plato.nlgen;

import openproof.plato.model.LinguisticPhenom;
import openproof.plato.model.Policy;

public class NLRenderingPolicy extends Policy implements NLRenderingPolicyFace {

	private static final String ALLOW_AMBIGUITY_KEY = "ALLOW_AMBIGUITY";
	private static final String RULESET_KEY = "RULE SET";


	public NLRenderingPolicy() {
		setAllowAmbiguity(true);
	}

	public NLRenderingPolicy(LinguisticPhenom phenom) {
		setRuleSet(ruleSetFromPhenom(phenom));
	}

	private RuleSet ruleSetFromPhenom(LinguisticPhenom phenom) {
		switch (phenom) {
		case NONE: return RuleSet.NONE;
		case AGGREGATION: return RuleSet.AGGREGATION;
		case CONNECTIVES: return RuleSet.CONNECTIVES;
		case ELLIPSES: return RuleSet.ELLIPSES;
		case MODIFICATION: return RuleSet.MODIFICATION;
		case PARTITIVES: return RuleSet.PARTITIVES;
		case PRONOUNS: return RuleSet.PRONOUNS;
		default: return RuleSet.ALL;
		}
	}


	public NLRenderingPolicy(Boolean b, RuleSet rs) {
		setAllowAmbiguity(b);
		setRuleSet(rs);
	}


	@Override
	public void setAllowAmbiguity(Boolean allowAmbiguity) {
		put(ALLOW_AMBIGUITY_KEY, allowAmbiguity);
	}

	@Override
	public Boolean getAllowAmbiguity() {
		return (Boolean) get(ALLOW_AMBIGUITY_KEY);
	}

	@Override
	public void setRuleSet(RuleSet rules) {
		put(RULESET_KEY, rules);
	}

	@Override
	public RuleSet getRuleSet() {
		return (RuleSet) get(RULESET_KEY);
	}


}
