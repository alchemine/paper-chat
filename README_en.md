# Paper-Chat

**Paper-Chat** is an AI chatbot that enables conversations about academic papers using arXiv IDs. \
It is designed to assist researchers and students in quick learning, efficient practical application, and trend analysis, helping them easily understand and explore complex academic content.

![Paper-Chat Interface](assets/image-5.png)

# Prerequisites

1. Docker and Docker Compose must be installed.

# Installation and Execution

## Production

1. Clone the repository

   ```bash
   git clone https://github.com/alchemine/paper-chat.git
   cd paper-chat
   ```

2. Build the cluster (entrypoint, elasticsearch cluster) and run the Streamlit app using Docker Compose

   ```bash
   docker-compose -f docker-compose.prd.yml up
   ```

3. Access the Streamlit app in your browser

   ```bash
   http://localhost:8501
   ```

## Development

1. Clone the repository

   ```bash
   git clone https://github.com/alchemine/paper-chat.git
   cd paper-chat
   ```

2. Build the cluster (entrypoint, elasticsearch cluster)

   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

   - Otherwise, use dev container

3. If you want to use `AzureChatOpenAI`, required environment variables must be set in the `dev.env` file.

   ```bash
   OPENAI_API_KEY=...

   # Needed if using Azure LLM
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_KEY=...
   OPENAI_API_VERSION=...
   AZURE_OPENAI_LLM_DEPLOYMENT_NAME=...
   AZURE_OPENAI_LLM_MODEL=...
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=...
   AZURE_OPENAI_EMBEDDINGS_MODEL=...
   ```

   - If you write `OPENAI_API_KEY`, OpenAI API Key value is automatically filled with the value.
   - You can use `AzureChatOpenAI` with `azure` input in OpenAI API Key.

# How to Use

### 1. Enter the OpenAI API Key and the arXiv ID of the paper in the sidebar.

![alt text](assets/image-3.png)

- The paper ID is in the format like `2004.07606`.
- If you input a string that can identify the ID, such as `https://arxiv.org/pdf/2004.07606` or `https://arxiv.org/abs/2004.07606`, the ID will be automatically identified.

### 2. A summary of the paper is automatically generated and displayed.

![alt text](assets/image-1.png)

- When generating a summary for a paper for the first time, it takes about 30 seconds.
- Subsequent requests will load the previously generated summary.

### 3. You can ask questions about the paper through the chatbot interface and receive answers based on the paper's content.

![alt text](assets/image-4.png)

The queries used for exploration (Queries) and the referenced content (Contexts) are displayed along with the answer.
Check these to see if any hallucination has occurred.

### 4. If you want to ask questions about a different paper, enter a new arXiv ID in the sidebar.

![alt text](assets/image-5.png)

# Additional Features and Improvements

Please refer to the [Issues](https://github.com/alchemine/paper-chat/issues) page for major development progress and updates.
