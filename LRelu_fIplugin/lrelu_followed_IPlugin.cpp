#include <memory>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <string.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "lrelu";
static const int BATCH_SIZE = 1;
static const int INPUT_C = 128;
static const int INPUT_H = 52;
static const int INPUT_W = 52;

static const int TIMING_ITERATIONS = 1;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity == Severity::kINFO) return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }
} gLogger;

class CopyHalfPlugin: public IPlugin
{
public:
    CopyHalfPlugin(const Weights *weights, int nbWeights, int nbOutputChannels)
    {
    }

    CopyHalfPlugin(const void* data, size_t length)
    {
    }

    ~CopyHalfPlugin()
    {
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
    }

    int initialize() override
    {
        return 0;
    }

    virtual void terminate() override
    {
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        std::cout << "IPlugin   enqueue" << std::endl;
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W  * sizeof(float) / 2, cudaMemcpyDeviceToDevice, stream));
        std::cout << "IPlugin   enqueue  Done" << std::endl;
        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return 0;
    }

    virtual void serialize(void* buffer) override
    {
    }
private:
};

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
    public:
        bool isPlugin(const char* name)
        {
            return !strcmp(name, "cp1");
        }

        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
        {
            // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
            static const int NB_OUTPUT_CHANNELS = 64;
            assert(isPlugin(layerName));
            assert(mPlugin.get() == nullptr);
            mPlugin = std::unique_ptr<CopyHalfPlugin>(new CopyHalfPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));
            return mPlugin.get();
        }

        IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
        {
            assert(isPlugin(layerName));
            assert(mPlugin.get() == nullptr);
            mPlugin = std::unique_ptr<CopyHalfPlugin>(new CopyHalfPlugin(serialData, serialLength));
            return mPlugin.get();
        }

        void destroyPlugin()
        {
            mPlugin.release();
        }

        std::unique_ptr<CopyHalfPlugin> mPlugin{ nullptr };
};

ICudaEngine* buildEngine(unsigned int maxBatchSize,
        nvcaffeparser1::IPluginFactory* pluginFactory)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    IPlugin *pluginObj = pluginFactory->createPlugin("cp1", 0, 0);
    auto inputs = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{INPUT_C, INPUT_H, INPUT_W});
    auto ip = network->addPlugin(&inputs, 1, *pluginObj);
    auto lrelu = network->addActivation(*ip->getOutput(0), ActivationType::kLEAKY_RELU);
    lrelu->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*lrelu->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setFp16Mode(true);
    builder->setStrictTypeConstraints(true);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    network->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
}

void doInference(
        IExecutionContext& context,
        float* input,
        float* output,
        int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    DimsCHW inputDims =
        static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex));
    size_t inputSize =
        batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);

    int outputIndex= engine.getBindingIndex(OUTPUT_BLOB_NAME);
    DimsCHW outputDims =
        static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex));
    size_t outputSize =
        batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize,
                cudaMemcpyHostToDevice)); 

    for (int i = 0; i < TIMING_ITERATIONS;i++)
        context.execute(batchSize, buffers);

    CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

}

int main(int argc, char** argv)
{
    ICudaEngine* engine;

    PluginFactory pluginFactory;
    engine = buildEngine(BATCH_SIZE, &pluginFactory);

    pluginFactory.destroyPlugin();
    IExecutionContext *context = engine->createExecutionContext();

    int size = BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W;
    float* inputData = new float[size];
    float* outputData = new float[size];

    std::ifstream fin(argv[1], std::ios::binary);
    int i = 0;
    float f;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        inputData[i++] = f;
    }

    doInference(*context, inputData, outputData, BATCH_SIZE);

    std::ofstream myfile(argv[2]);
    for(int count = 0; count < size; count ++){
        myfile << outputData[count] << std::endl;
    }

    context->destroy();
    engine->destroy();
    pluginFactory.destroyPlugin();

    delete[] inputData;
    delete[] outputData;

    return 0;
}
