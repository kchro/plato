package openproof.plato.model;

public enum LinguisticPhenom {
	
	NONE("NONE"),
	CONNECTIVES("CONNECTIVES"),
	MODIFICATION("MODIFICATION"),
	AGGREGATION("AGGREGATION"),
	PRONOUNS("PRONOUNS"),
	ELLIPSES("ELLIPSES"),
	PARTITIVES("PARTITIVES");
	
	private String dbVal;
	
	private LinguisticPhenom(String dbVal) {
		this.dbVal = dbVal;
	}
	
	public String toDB() {
		return dbVal;
	}
	
	public static LinguisticPhenom fromDB(String value) {
		if (value.equals(NONE.toDB())) { return NONE; }
		if (value.equals(CONNECTIVES.toDB())) { return CONNECTIVES; }
		if (value.equals(MODIFICATION.toDB())) { return MODIFICATION; }
		if (value.equals(AGGREGATION.toDB())) { return AGGREGATION; }
		if (value.equals(PRONOUNS.toDB())) { return PRONOUNS; }
		if (value.equals(ELLIPSES.toDB())) { return ELLIPSES; }
		if (value.equals(PARTITIVES.toDB())) { return PARTITIVES; }
		return null;
	}

}
