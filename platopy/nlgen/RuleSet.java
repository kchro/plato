package openproof.plato.nlgen;

public enum RuleSet {
	
	// Enum for ruleset types, each recording its corresponding Lingo filename argument.
	
	NONE("None", "rules.one"),
	ALL("All", "rules.all"),
	PRONOUNS("Pronouns", "rules.pro"),
	ELLIPSES("Ellipses", "rules.ellip"),
	AGGREGATION("Aggregation", "rules.agg"),
	MODIFICATION("Modification", "rules.mod"),
	PARTITIVES("Partitives", "rules.part"),
	CONNECTIVES("Connectives", "rules.conn");
	
	private String displayName;
	private String filename;

	RuleSet(String s, String filename) {
		this.displayName = s;
		this.setFilename(filename);
	}
	
	public String toString() { return this.displayName; }

	public String getFilename() {
		return filename;
	}

	public void setFilename(String filename) {
		this.filename = filename;
	}

}
