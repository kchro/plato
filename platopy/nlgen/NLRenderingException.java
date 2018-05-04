package openproof.plato.nlgen;

import java.io.IOException;

public class NLRenderingException extends Exception {
	
	public NLRenderingException(String message) {
		super(message);
	}

	public NLRenderingException(IOException e) {
		this(e.getMessage());
		this.initCause(e);
	}

}
