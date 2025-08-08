package main.java.tukano.impl.rest;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

import static main.java.tukano.impl.rest.TukanoRestServer.Log;

@Path("/ctrl")
public class ControlResource
{

    /**
     * Health check endpoint to verify the service is running.
     * Useful for monitoring and deployment verification.
     */
    @Path("/version")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String version() {
        Log.info("Version check requested");
        return "Tukano REST API v1.0 - Service Running";
    }
    
    /**
     * Simple health check endpoint.
     */
    @Path("/health")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String health() {
        return "OK";
    }

}
