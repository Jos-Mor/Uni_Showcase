package main.java.tukano.impl.rest;

import java.util.HashSet;
import java.util.Set;
import java.util.logging.Logger;

import main.java.tukano.impl.Token;
import main.java.tukano.impl.rest.utils.Props;
import main.java.utils.IP;
import jakarta.ws.rs.core.Application;


public class TukanoRestServer extends Application {
	public static final Logger Log = Logger.getLogger(TukanoRestServer.class.getName());

	static final String INETADDR_ANY = "0.0.0.0";
	static String SERVER_BASE_URI = "http://%s:%s/rest";

	public static final int PORT = 8080;

	public static String serverURI;
	private Set<Class<?>> resources = new HashSet<>();
	private Set<Object> singletons = new HashSet<>();


	static {
		System.setProperty("java.util.logging.SimpleFormatter.format", "%4$s: %5$s");
	}

	public TukanoRestServer() {
		serverURI = String.format(SERVER_BASE_URI, IP.hostname(), PORT);

		Log.info("Initializing Tukano REST Server at " + serverURI);
		
		// Register REST resource classes
		resources.add(RestBlobsResource.class);
		resources.add(RestUsersResource.class);
		resources.add(RestShortsResource.class);
		resources.add(ControlResource.class);

		// Load Azure configuration
		Props.load("azurekeys-northeurope.props");

		// Initialize token secret from environment or use default for development
		String tokenSecret = System.getenv("TOKEN_SECRET");
		if (tokenSecret == null || tokenSecret.isEmpty()) {
			tokenSecret = "default-dev-secret-change-in-production";
			Log.warning("Using default token secret. Set TOKEN_SECRET environment variable in production.");
		}
		Token.setSecret(tokenSecret);

		Log.info("Tukano REST Server initialization completed");
	}

	@Override
	public Set<Class<?>> getClasses() {
		return resources;
	}

	@Override
	public Set<Object> getSingletons() {
		return singletons;
	}


}
