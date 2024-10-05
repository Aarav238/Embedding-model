# Processing Large Files with OpenAI Embeddings

This document explains how the provided Node.js code processes a large file in chunks, generates embeddings using the OpenAI API, and aggregates these embeddings into a final result using different methods like mean, max, or attention-based aggregation.

## Table of Contents
1. [Overview](#overview)
2. [Main Components](#main-components)
3. [Detailed Breakdown](#detailed-breakdown)
   - [1. Setting Up OpenAI API](#1-setting-up-openai-api)
   - [2. The `EmbeddingAggregator` Class](#2-the-embeddingaggregator-class)
   - [3. Function: `getEmbeddingFromAPI`](#3-function-getembeddingfromapi)
   - [4. Reading a Large File in Chunks](#4-reading-a-large-file-in-chunks)
   - [5. Processing and Aggregating Embeddings](#5-processing-and-aggregating-embeddings)
   - [6. Example Usage in the `main` Function](#6-example-usage-in-the-main-function)
4. [Conclusion](#conclusion)

---

## Overview

The code processes large text files by reading them in chunks, generating embeddings for each chunk using OpenAI's embedding API, and finally aggregating these embeddings into a single vector. The aggregation can be performed using different methods such as mean, max, or attention.

This approach allows you to process very large files efficiently without overloading memory by using chunk-based reading and processing.

---

## Main Components

The code consists of several main components:

- **OpenAI API**: Used to generate text embeddings for chunks of the file.
- **`EmbeddingAggregator` Class**: Aggregates embeddings across chunks using mean, max, or attention-based methods.
- **File Chunk Reader**: Reads the large file in smaller parts to avoid memory overload.
- **TensorFlow.js**: Used for matrix operations to manipulate the embeddings (e.g., mean, max, and attention operations).

---

## Detailed Breakdown

### 1. Setting Up OpenAI API

```javascript
const OpenAI = require("openai");

const openai = new OpenAI({
    apiKey: "your-api-key-here"
});
```
- **OpenAI Setup**: The `OpenAI` library is initialized with an API key, which is used to authenticate API requests.
- **Embedding API**: We use OpenAI's embedding endpoint to generate text embeddings. In this example, we are using the `"text-embedding-3-small"` model.

### 2. The EmbeddingAggregator Class
```javascript
class EmbeddingAggregator {
    constructor(inputDim = 1536, method = 'mean') {
        this.inputDim = inputDim;
        this.method = method;
        
        if (method === 'attention') {
            this.attentionWeights = tf.layers.dense({
                units: 1,
                inputShape: [inputDim],
                kernelInitializer: 'glorotNormal'
            });
        }
    }

    async aggregate(chunkEmbeddings) {
        switch (this.method) {
            case 'mean':
                return tf.mean(chunkEmbeddings, 0);
            case 'max':
                return tf.max(chunkEmbeddings, 0);
            case 'attention':
                const attentionScores = this.attentionWeights.apply(chunkEmbeddings);
                const attentionWeights = tf.softmax(attentionScores, 0);
                const weightedEmbeddings = tf.mul(chunkEmbeddings, attentionWeights);
                return tf.sum(weightedEmbeddings, 0);
            default:
                throw new Error("Unsupported aggregation method");
        }
    }
}
```
- **Purpose**: : The `EmbeddingAggregator` class handles the aggregation of embeddings across different chunks.
- **Constructor**: Takes the input dimension (inputDim = 1536 by default) and the aggregation method ('mean', 'max', or 'attention').
- **Aggregation Methods**:
   - **Mean**: Averages the embeddings.
   - **Max**:: Takes the maximum value across the embeddings.
   - **Attention**: Uses an attention mechanism to weight embeddings before summing them.

### 3. Function: `getEmbeddingFromAPI`
```javascript
async function getEmbeddingFromAPI(text) {
    try {
        const response = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
            encoding_format: "float",
        });
        return response.data[0].embedding;
    } catch (error) {
        console.error("Error fetching embedding from OpenAI API:", error);
        throw error;
    }
}
```
- **Purpose**: Sends the text to OpenAI's API and retrieves the embedding for the input.
- **API Call**: The `openai.embeddings.create` function sends the text input to the OpenAI API and gets the embedding response.
- **Response Handling**: The API returns a list of embeddings, and we extract the first embedding from `response.data[0].embedding`.


### 4. Reading Large Files in chunks
```javascript
async function* readFileInChunks(filePath, chunkSize = 1000000) {
    const fileHandle = await fs.open(filePath, 'r');
    const readStream = fileHandle.createReadStream();
    const rl = readline.createInterface({
        input: readStream,
        crlfDelay: Infinity
    });

    let chunk = '';
    for await (const line of rl) {
        chunk += line + '\n';
        if (chunk.length >= chunkSize) {
            yield chunk;
            chunk = '';
        }
    }

    if (chunk) {
        yield chunk;
    }

    await fileHandle.close();
}
```
- **Purpose**: Reads a large file in chunks of a specified size (`chunkSize = 1MB `by default).
- **Stream-Based**: Uses a `readStream` and the `readline` module to handle large files efficiently without loading the entire file into memory.
- **Yielding Chunks**: After accumulating data up to `chunkSize`, it yields the chunk to be processed.

### 5. Processing and Aggregating Embeddings
```javascript
async function getLargeFileEmbedding(filePath) {
    const aggregator = new EmbeddingAggregator(1536, 'mean');
    let chunkEmbeddings = [];

    for await (const chunk of readFileInChunks(filePath)) {
        const embedding = await getEmbeddingFromAPI(chunk);
        chunkEmbeddings.push(embedding);

        if (chunkEmbeddings.length >= 10) {
            const embeddingsTensor = tf.tensor2d(chunkEmbeddings);
            const intermediateTensor = await aggregator.aggregate(embeddingsTensor);
            chunkEmbeddings = [intermediateTensor.arraySync()];
            tf.dispose(embeddingsTensor);
            tf.dispose(intermediateTensor);
        }
    }

    if (chunkEmbeddings.length > 0) {
        const embeddingsTensor = tf.tensor2d(chunkEmbeddings);
        const finalEmbedding = await aggregator.aggregate(embeddingsTensor);
        tf.dispose(embeddingsTensor);
        return finalEmbedding;
    }
}
```
- **Purpose**: Processes the file in chunks, generates embeddings for each chunk, and aggregates them.
- **Chunk Embedding**: For each chunk of the file, `getEmbeddingFromAPI` is called to retrieve the embedding. These embeddings are stored in `chunkEmbeddings`.
- **Batch Processing**: Every 10 chunks, the embeddings are aggregated to avoid memory overload.
- **Final Aggregation**: After processing all the chunks, the remaining embeddings are aggregated into a final vector.

### 6. Example Usage in the main Function
```javascript
async function main() {
    const filePath = './sample.yaml'; // Path to your large file
    console.log("Processing large file...");
    const embedding = await getLargeFileEmbedding(filePath);
    
    // Output the full embedding
    console.log("Full embedding:", embedding.arraySync());
    console.log("Embedding shape:", embedding.shape);
}

main().catch(console.error);
```
**Purpose**: This function demonstrates how to use the getLargeFileEmbedding function.
**File Path**: You provide the path to the large file you want to process.
**Final Embedding**: The final aggregated embedding is printed after processing the entire file.


## Conclusion

This code efficiently processes large text files in chunks and generates embeddings for each chunk using OpenAI's API. The embeddings are then aggregated into a single vector using methods like mean, max, or attention. The chunk-based approach helps avoid memory overload when dealing with large files, making the system scalable and robust.

