package main.java.tukano.impl.storage.database.azure;

import com.azure.cosmos.*;

import static main.java.tukano.impl.rest.TukanoRestServer.Log;

public class CosmosDBSource {


    private static final String CONNECTION_URL = System.getenv("COSMOSDB_URL");
    private static final String DB_KEY = System.getenv("COSMOSDB_KEY");
    private static final String DB_NAME = System.getenv("COSMOSDB_DATABASE");

    protected static synchronized void init() {
        if (short_container != null) return;
        
        // Validate required environment variables
        if (CONNECTION_URL == null || DB_KEY == null || DB_NAME == null) {
            Log.severe("Missing required Cosmos DB environment variables: COSMOSDB_URL, COSMOSDB_KEY, COSMOSDB_DATABASE");
            throw new RuntimeException("Cosmos DB configuration incomplete");
        }
        
        Log.info("Initializing Cosmos DB connection - URL: " + CONNECTION_URL + ", Database: " + DB_NAME);

        try {
            client = new CosmosClientBuilder()
                    .endpoint(CONNECTION_URL)
                    .key(DB_KEY)
                    .gatewayMode()
                    // Use gateway mode for better compatibility, switch to .directMode() for better performance
                    .consistencyLevel(ConsistencyLevel.SESSION)
                    .multipleWriteRegionsEnabled(true)
                    .connectionSharingAcrossClientsEnabled(true)
                    .contentResponseOnWriteEnabled(true)
                    .buildClient();
            Log.info("Cosmos DB client created successfully");

            db = client.getDatabase(DB_NAME);
            Log.info("Connected to database: " + DB_NAME);
            
            user_container = db.getContainer(USERS_CONTAINER);
            Log.info("Connected to users container");
            
            short_container = db.getContainer(SHORTS_CONTAINER);
            Log.info("Connected to shorts container");
            
        } catch (Exception e) {
            Log.severe("Failed to initialize Cosmos DB: " + e.getMessage());
            throw new RuntimeException("Cosmos DB initialization failed", e);
        }
    }

    protected static final String USERS_CONTAINER = "users";

    protected static final String SHORTS_CONTAINER = "shorts";

    protected static CosmosClient client;
    private static CosmosDatabase db;
    protected static CosmosContainer user_container;

    protected static CosmosContainer short_container;


}
