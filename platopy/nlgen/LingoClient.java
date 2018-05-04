package openproof.plato.nlgen;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import openproof.fol.representation.OPFormula;
import openproof.fol.representation.parser.ParseException;
import openproof.fol.representation.parser.StringToFormulaParser;

import org.apache.log4j.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/**
 * NL client to interface with the LinGo sentence generator by HTTP Post connection.
 * @author dbp
 *
 */
public class LingoClient implements NLGeneratorFace {

	private static String PROTOCOL = "http";
	private static final String POST = "POST";

	// cypriot is only available from CSLI.  
	// Use a tunnel, and switch this to localhost to make it work outside.
//		private static String STANFORD_HOST = "localhost";
	private static String STANFORD_HOST = "cypriot.stanford.edu";
	private static String CLOUD_HOST = "172.31.2.196";
	
	private static String DEFAULT_HOST = STANFORD_HOST;

	private String hostname = DEFAULT_HOST;
	private int port = DEFAULT_PORT;
	private static int DEFAULT_PORT = 5000;

	private static String DEFAULT_URL = "/ace/";
	

	private List<NLRendering> responseLines;


	public LingoClient() {
		this(true);
	}
	
	public LingoClient(boolean testing) {
		this.hostname = testing ? STANFORD_HOST : CLOUD_HOST;
		Logger.getLogger(LingoClient.class).info("Lingo host is: "+this.hostname);
		
	}

	private URL getServerURL(NLRenderingPolicyFace policy) throws MalformedURLException {
		String baseurl = DEFAULT_URL;

		URL result = new URL(PROTOCOL, this.hostname, this.port, baseurl);
		Logger.getLogger(LingoClient.class).debug("URL: " + result);
		
		return result;
	}

	private void generateSentence(OPFormula formula, NLRenderingPolicyFace policy) throws NLRenderingException {
		try {
			URL url = getServerURL(policy);

			Logger.getLogger(getClass()).debug(getRulesSetName(policy));

			this.responseLines = null;
			
			String urlParameters = "sig="+formulaURL(formula)+"&rules="+getRulesSetName(policy);
			byte[] postData       = urlParameters.getBytes( Charset.forName( "UTF-8" ));
			int    postDataLength = postData.length;

			Logger.getLogger(getClass()).debug(urlParameters);

			HttpURLConnection connection = (HttpURLConnection) url.openConnection();

			connection.setUseCaches(false);
			HttpURLConnection.setFollowRedirects(false);

			connection.setRequestMethod(POST);
			
			connection.setDoOutput(true);			
			connection.setDoInput(true);
			
			connection.setRequestProperty( "Content-Type", "application/x-www-form-urlencoded"); 
			connection.setRequestProperty( "charset", "utf-8");
			connection.setRequestProperty( "Content-Length", Integer.toString( postDataLength ));
			connection.setUseCaches( false );
			
			try (DataOutputStream wr = new DataOutputStream(connection.getOutputStream())) {
				wr.write(postData);
			}

			int responseCode = connection.getResponseCode();
			Logger.getLogger(getClass()).debug("Response: "+responseCode);
			for (Entry<String, List<String>> k : connection.getHeaderFields().entrySet()) {
				Logger.getLogger(getClass()).debug(k.toString());
			}

			responseLines = new ArrayList<NLRendering>();
			
			ArrayList<NLPair> x = parseResults(connection.getInputStream());
			for (int i = 0; i < x.size(); i++) responseLines.add(x.get(i));
			

		} catch (MalformedURLException e) {
			Logger.getLogger(LingoClient.class).warn("", e);
		} catch (ConnectException e) {
			throw new NLRenderingException(e);
		} catch (IOException e) {
			Logger.getLogger(LingoClient.class).warn("", e);
		}
	}

	private String getRulesSetName(NLRenderingPolicyFace policy) {
		return policy.getRuleSet().getFilename();
	}

	private ArrayList<NLPair> parseResults(InputStream stream) {
		ArrayList<NLPair> result = new ArrayList<NLPair>();
		
		DocumentBuilderFactory factory = null;
		DocumentBuilder builder = null;
		Document doc = null;

		try {
			factory = DocumentBuilderFactory.newInstance();
			builder = factory.newDocumentBuilder();
			doc = builder.parse(new InputSource(stream));
			
			NodeList nodeList = doc.getDocumentElement().getChildNodes();
			for (int i = 0; i < nodeList.getLength(); i++) {
				Node node = nodeList.item(i);
				if (node.getNodeType() == Node.ELEMENT_NODE) {
					Element elem = (Element) node;
					NLPair p = new NLPair();
					List<String> rules = new ArrayList<String>();
					p.addTranslation(elem.getElementsByTagName(NLPair.TRANSLATION).item(0).getChildNodes().item(0).getNodeValue());
					
					Element rulesnode = (Element) elem.getElementsByTagName(NLPair.RULES).item(0);
					
					NodeList ruleslist = rulesnode.getElementsByTagName("rule");
					for (int j = 0; j < ruleslist.getLength(); j++) {
						Element thisnode = (Element) ruleslist.item(j);
						p.addRule(thisnode.getChildNodes().item(0).getNodeValue());
					}
					result.add(p);
					
					Logger.getLogger(LingoClient.class).debug(p);
				}
			}
		} catch (ParserConfigurationException e) {
			Logger.getLogger(LingoClient.class).error(e);
		} catch (SAXException e) {
			Logger.getLogger(LingoClient.class).error(e);
		} catch (IOException e) {
			Logger.getLogger(LingoClient.class).error(e);
		}

		return result;
	}

	private String formulaURL(OPFormula formula) throws UnsupportedEncodingException {
		Logger.getLogger(LingoClient.class).debug("Formula: "+formula);
		String result = URLEncoder.encode(formula.toInternal().replaceAll("\\s*", ""), "UTF-8");
		Logger.getLogger(LingoClient.class).debug("Returned as: "+result);
		return result;
	}

	@Override
	public synchronized NLRendering render(OPFormula f, NLRenderingPolicyFace policy) throws NLRenderingException {

		generateSentence(f, policy);
				
		return responseLines.get(randomInRange(0, responseLines.size()-1));
	}

	private int randomInRange(int min, int max) {
		return min + (int)(Math.random() * ((max - min) + 1));
	}




	@Override
	public synchronized List<NLRendering> renderAll(OPFormula f, NLRenderingPolicyFace face) throws NLRenderingException {
		generateSentence(f, face);
		return responseLines;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			new LingoClient().render(StringToFormulaParser.getFormula("Tet(a)&Tet(b)"), new NLRenderingPolicy(false, RuleSet.NONE));
		} catch (UnsupportedEncodingException e) {
			Logger.getLogger(LingoClient.class).warn("Issue rendering", e);
		} catch (ParseException e) {
			Logger.getLogger(LingoClient.class).warn("Issue rendering", e);
		} catch (NLRenderingException e) {
			Logger.getLogger(LingoClient.class).warn("Issue rendering", e);
		}
	}

}
