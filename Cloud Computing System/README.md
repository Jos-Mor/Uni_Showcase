# Tukano - Social Media Backend

A Java-based REST API for a social media platform with Azure hosting, built as coursework for Cloud Computing Systems.

## What This Project Is

This is a backend server for a TikTok-like app where users can upload short videos, follow each other, and like content. Instead of running everything locally, we learned to use Microsoft Azure cloud services to store data and files.

**Important**: This was a learning project - I was figuring out cloud computing concepts as I went along, so the code isn't perfect but it works!

## What Was Built

### Core Features
- **User accounts**: Sign up, login, update profiles
- **Video uploads**: Users can post short videos 
- **Social features**: Follow users, like videos, see a personalized feed
- **Search**: Find users by username

### Cloud Integration
- **Azure Cosmos DB**: Stores user info and video metadata (NoSQL database)
- **Azure Blob Storage**: Stores the actual video files
- **Azure App Service**: Hosts the web application
- **Multi-region setup**: Configured for both North Europe and West US

## Technologies Used

- **Java 17** - Main programming language
- **JAX-RS** - For creating REST API endpoints
- **Maven** - Build tool and dependency management
- **Azure services** - Cloud database and storage
- **Artillery.js** - For load testing (first time using this)

## API Endpoints

The basic REST endpoints implemented:

```
👤 Users
Method	Path	Description
POST	/users	                Create a new user
GET	/users/{userId}?pwd=...	Get a user by ID (auth via password)
PUT	/users/{userId}?pwd=...	Update user
DELETE	/users/{userId}?pwd=...	Delete user
GET	/users?query=...	Search users by pattern

🎞 Shorts
Method	Path	Description
POST	/shorts/{userId}?pwd=...	                Create a short (post)
GET	/shorts/{shortId}	                        Get a short
GET	/shorts/{userId}/shorts	                        Get all shorts from user
POST	/shorts/{shortId}/{userId}/likes?pwd=...	Like/unlike a short
GET	/shorts/{shortId}/likes?pwd=...	                Get users who liked a short
POST	/shorts/{userId1}/{userId2}/followers?pwd=...	Follow/unfollow a user
GET	/shorts/{userId}/followers?pwd=...	        Get followers of a user
GET	/shorts/{userId}/feed?pwd=...	                Get feed for a user
DELETE	/shorts/{shortId}?pwd=...	                Delete a short
DELETE	/shorts/{userId}/shorts?pwd=...&token=...	Delete all shorts from user

📦 Blobs
Method	Path	Description
POST	/blobs/{blobId}?token=...	Upload binary data
GET	/blobs/{blobId}?token=...	Download binary data
DELETE	/blobs/{blobId}?token=...	Delete one blob
DELETE	/blobs/{userId}/blobs?token=...	Delete all blobs from user
```

## What I Learned

### Cloud Computing Concepts
- How to use cloud databases instead of local ones
- File storage in the cloud vs local file systems
- Deploying applications to cloud platforms
- Basic understanding of distributed systems

### Backend Development
- Building REST APIs from scratch
- Connecting different services together
- Basic authentication and security
- Database design for social media features

### Performance & Testing
- Load testing with Artillery (testing how many users the system can handle)
- Caching to make things faster
- Monitoring response times and error rates

## Challenges I Faced

1. **Azure Setup**: Getting all the cloud services configured and talking to each other
2. **Database Design**: Figuring out how to structure data for social media features
3. **Authentication**: Implementing a basic token system for user login
4. **Performance**: Making sure the system could handle multiple users at once

## Test Results

I ran performance tests to see how the system handled load:
- **Response times**: 49-136ms (mean around 70-85ms)
- **Test setup**: Warm up phase (1-5 users/sec for 10s), then main test (10 users/sec for 30s)
- **Success rate**: Mixed results - got successful 200s, some 409s (conflicts), and quite a few 500s (server errors - not sure why)
- **Peak request rate**: Up to 7 requests/second during testing
- **Total virtual users created**: 77 users across different test scenarios

*Note: The 500 errors suggest there were some issues with the server, possibly one of the endpoints, more debugging needed.*

## Running the Project

**Note**: You need Azure credentials to actually run this, which I can't share publicly.

```bash
# Build the project
mvn clean package

# Deploy to Azure (requires proper credentials)
mvn azure-webapp:deploy
```

## Project Structure

```
├── scc2425-tukano/          # Main application code
│   ├── src/main/java/       # Java source files
│   ├── Tests_Artillery/     # Performance testing scripts
│   └── pom.xml             # Maven configuration
├── scc2425-mgt-code/       # Azure setup scripts
└── azurekeys-*.template    # Configuration templates
```

## Honest Assessment

### What Went Well
- Got the basic social media features working
- Successfully integrated with Azure cloud services
- Learned a lot about backend development and cloud computing
- Performance testing showed the system could handle reasonable load

### What Could Be Better
- Security is pretty basic (not production-ready)
- Error handling could be more robust
- Code organization could be cleaner
- Some features were implemented just to meet requirements

### What I'd Do Differently
- Spend more time on code organization from the start
- Better error messages and logging
- More comprehensive testing
- Cleaner separation between different parts of the system

## Academic Context

This was built for my Cloud Computing Systems course. The main learning goals were:
- Understanding cloud platforms and services
- Building distributed applications
- Working with NoSQL databases
- Performance testing and optimization

**This is student work** - it demonstrates learning and experimentation rather than production-ready software.
