package main.java.tukano.impl.storage.blobs;

import java.util.function.Consumer;

import com.azure.core.util.BinaryData;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobContainerClientBuilder;

import main.java.tukano.api.Result;

import static main.java.tukano.api.Result.ErrorCode.*;
import static main.java.tukano.api.Result.error;
import static main.java.tukano.api.Result.ok;

public class AzureBlobStorage implements BlobStorage {
    
    private static final String CONTAINER_NAME = "shorts";
    private final BlobContainerClient containerClient;
    
    public AzureBlobStorage() {
        String storageConnectionString = System.getenv("BlobStoreConnection");
        if (storageConnectionString == null) {
            throw new RuntimeException("BlobStoreConnection environment variable not set");
        }
        
        this.containerClient = new BlobContainerClientBuilder()
                .connectionString(storageConnectionString)
                .containerName(CONTAINER_NAME)
                .buildClient();
    }


    @Override
    public Result<Void> write(String path, byte[] bytes) {
        if (path == null || bytes == null) {
            return error(BAD_REQUEST);
        }
        
        try {
            var blob = containerClient.getBlobClient(path);
            if (blob.exists()) {
                return error(CONFLICT);
            }
            var data = BinaryData.fromBytes(bytes);
            blob.upload(data);
            return ok();
        } catch (Exception e) {
            return error(INTERNAL_ERROR);
        }
    }

    @Override
    public Result<Void> delete(String path) {
        if (path == null) {
            return error(BAD_REQUEST);
        }
        
        try {
            var blob = containerClient.getBlobClient(path);
            if (!blob.exists()) {
                return error(NOT_FOUND);
            }
            blob.delete();
            return ok();
        } catch (Exception e) {
            return error(INTERNAL_ERROR);
        }
    }

    @Override
    public Result<byte[]> read(String path) {
        if (path == null) {
            return error(BAD_REQUEST);
        }

        try {
            var blob = containerClient.getBlobClient(path);
            if (!blob.exists()) {
                return error(NOT_FOUND);
            }

            byte[] data = blob.downloadContent().toBytes();
            return data != null ? ok(data) : error(INTERNAL_ERROR);
        } catch (Exception e) {
            return error(INTERNAL_ERROR);
        }
    }

    @Override
    public Result<Void> read(String path, Consumer<byte[]> sink) {
        if (path == null || sink == null) {
            return error(BAD_REQUEST);
        }

        try {
            var blob = containerClient.getBlobClient(path);
            if (!blob.exists()) {
                return error(NOT_FOUND);
            }

            byte[] data = blob.downloadContent().toBytes();
            if (data != null) {
                sink.accept(data);
                return ok();
            } else {
                return error(INTERNAL_ERROR);
            }
        } catch (Exception e) {
            return error(INTERNAL_ERROR);
        }
    }
}
