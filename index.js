const fs = require('fs').promises;
const readline = require('readline');
const tf = require('@tensorflow/tfjs-node');
const OpenAI = require("openai");

// Set up OpenAI API
const openai = new OpenAI({
    apiKey: "your api key"
});

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

// Use OpenAI's embedding API to get real embeddings
async function getEmbeddingFromAPI(text) {
    try {
        const response = await openai.embeddings.create({
            model: "text-embedding-3-small", // Updated model name
            input: text,
            encoding_format: "float",
        });
        // The API returns a list of embeddings, so we extract the first one
        return response.data[0].embedding;
    } catch (error) {
        console.error("Error fetching embedding from OpenAI API:", error);
        throw error;
    }
}

// Read large file in chunks
async function* readFileInChunks(filePath, chunkSize = 1000000) { // 1MB chunks
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

// Process large file to get final aggregated embedding
async function getLargeFileEmbedding(filePath) {
    const aggregator = new EmbeddingAggregator(1536, 'mean');
    let chunkEmbeddings = [];

    for await (const chunk of readFileInChunks(filePath)) {
        const embedding = await getEmbeddingFromAPI(chunk);
        chunkEmbeddings.push(embedding);

        // Process in batches to avoid memory issues
        if (chunkEmbeddings.length >= 10) {
            const embeddingsTensor = tf.tensor2d(chunkEmbeddings);
            const intermediateTensor = await aggregator.aggregate(embeddingsTensor);
            chunkEmbeddings = [intermediateTensor.arraySync()];
            tf.dispose(embeddingsTensor);
            tf.dispose(intermediateTensor);
        }
    }

    // Process any remaining chunks
    if (chunkEmbeddings.length > 0) {
        const embeddingsTensor = tf.tensor2d(chunkEmbeddings);
        const finalEmbedding = await aggregator.aggregate(embeddingsTensor);
        tf.dispose(embeddingsTensor);
        return finalEmbedding;
    }
}

// Example usage
async function main() {
    const filePath = './sample.yaml'; // Path to your large file
    console.log("Processing large file...");
    const embedding = await getLargeFileEmbedding(filePath);
  console.log("Full embedding:", embedding.arraySync());
    console.log("Embedding shape:", embedding.shape);
  const vectorArray = embedding.arraySync();
  console.log(vectorArray.length)
}

main().catch(console.error);
