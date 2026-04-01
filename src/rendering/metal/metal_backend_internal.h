#pragma once

#include <memory>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <simd/simd.h>

#include "rendering/gpu_shared_types.h"
#include "rendering/metal/metal_postprocess.h"
#include "rendering/ray_tracing_backend.h"
#include "rendering/scene_packer.h"

namespace rt::metal_backend_internal {

class MetalPathTracerBackend final : public IRayTracingBackend {
public:
  explicit MetalPathTracerBackend(const std::string& shaderLibraryPath);

  std::string name() const override;
  void setScene(const RTScene& scene) override;
  void updateCamera(const RTCamera& camera) override;
  void resize(uint32_t width, uint32_t height) override;
  void resetAccumulation() override;
  void renderIteration(const RenderConfig& config) override;
  RenderBuffer downloadRenderBuffer() const override;

private:
  id<MTLTexture> createPrivateTexture(MTLPixelFormat format, uint32_t w, uint32_t h);
  void ensureMetalFXResources(uint32_t outputWidth, uint32_t outputHeight);
  void teardownMetalFXResources();

  void encodePathTracePass(id<MTLCommandBuffer> cmdBuf);

  void buildSceneBuffers();
  void uploadSceneBuffers(SceneGpuAccumulator& acc);
  bool sceneContainsTransmission() const;
  id<MTLAccelerationStructure> buildAndCompactBLAS(MTLPrimitiveAccelerationStructureDescriptor* descriptor);
  void buildAccelerationStructure(uint32_t triangleCount);

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> pathTracePipelineState_ = nil;

  id<MTLBuffer> positionBuffer_ = nil;
  id<MTLBuffer> normalVertexBuffer_ = nil;
  id<MTLBuffer> vertexColorBuffer_ = nil;
  id<MTLBuffer> texcoordBuffer_ = nil;
  id<MTLBuffer> isoScalarsBuffer_ = nil;
  id<MTLBuffer> accelIndexBuffer_ = nil;
  id<MTLBuffer> triangleBuffer_ = nil;
  id<MTLBuffer> materialBuffer_ = nil;
  id<MTLBuffer> textureMetadataBuffer_ = nil;
  id<MTLBuffer> texturePixelBuffer_ = nil;
  id<MTLBuffer> lightBuffer_ = nil;
  id<MTLBuffer> emissiveTriangleBuffer_ = nil;
  id<MTLBuffer> environmentSampleBuffer_ = nil;

  id<MTLAccelerationStructure> meshCurveAcceleration_ = nil;
  id<MTLAccelerationStructure> pointAcceleration_ = nil;
  id<MTLAccelerationStructure> triangleBLAS_ = nil;
  id<MTLAccelerationStructure> curveBLAS_ = nil;

  id<MTLBuffer> curvePrimitiveBuffer_ = nil;
  id<MTLBuffer> curveControlPointBuffer_ = nil;
  id<MTLBuffer> curveRadiusBuffer_ = nil;
  uint32_t curveSegmentCount_ = 0;
  std::vector<simd_float3> curveControlPoints_;
  std::vector<float> curveRadii_;

  id<MTLAccelerationStructure> pointBLAS_ = nil;
  id<MTLBuffer> pointPrimitiveBuffer_ = nil;
  id<MTLBuffer> pointBboxBuffer_ = nil;
  std::vector<GPUPointPrimitive> pointPrimitives_;
  std::vector<PackedBoundingBox> pointBboxData_;
  uint32_t emissiveTriangleCount_ = 0u;

  id<MTLIntersectionFunctionTable> intersectionFunctionTable_ = nil;

  id<MTLBuffer> cameraBuffer_ = nil;
  id<MTLBuffer> frameBuffer_ = nil;
  id<MTLBuffer> lightingBuffer_ = nil;
  id<MTLBuffer> toonBuffer_ = nil;

  id<MTLBuffer> accumulationBuffer_ = nil;
  id<MTLBuffer> rawColorBuffer_ = nil;
  id<MTLBuffer> outputBuffer_ = nil;
  id<MTLBuffer> depthBuffer_ = nil;
  id<MTLBuffer> linearDepthBuffer_ = nil;
  id<MTLBuffer> normalBuffer_ = nil;
  id<MTLBuffer> objectIdBuffer_ = nil;

  mutable id<MTLCommandBuffer> lastCommandBuffer_ = nil;
  bool lastUseFxaa_ = false;
  bool lastEnableMetalFX_ = false;
  bool lastUseToon_ = false;

  id<MTLComputePipelineState> bufferToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> textureToBufferPipelineState_ = nil;
  id<MTLComputePipelineState> depthToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> roughnessToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> motionToTexturePipelineState_ = nil;

  id<MTLBuffer> diffuseAlbedoBuffer_ = nil;
  id<MTLBuffer> specularAlbedoBuffer_ = nil;
  id<MTLBuffer> roughnessAuxBuffer_ = nil;
  id<MTLBuffer> motionVectorBuffer_ = nil;

  id<MTLTexture> metalFXInputTexture_ = nil;
  id<MTLTexture> metalFXOutputTexture_ = nil;
  id<MTLTexture> metalFXDepthTexture_ = nil;
  id<MTLTexture> metalFXMotionTexture_ = nil;
  id<MTLTexture> metalFXNormalTexture_ = nil;
  id<MTLTexture> metalFXDiffuseAlbedoTexture_ = nil;
  id<MTLTexture> metalFXSpecularAlbedoTexture_ = nil;
  id<MTLTexture> metalFXRoughnessTexture_ = nil;
  id<MTLBuffer> metalFXOutputBuffer_ = nil;
  id<MTLBuffer> metalFXTonemappedBuffer_ = nil;
  id<MTLBuffer> metalFXToonBuffer_ = nil;
  id<MTLFXTemporalDenoisedScaler> metalFXDenoisedScaler_ = nil;
  uint32_t metalFXOutputWidth_ = 0;
  uint32_t metalFXOutputHeight_ = 0;
  simd_float4x4 prevViewProj_;
  bool hasPrevViewProj_ = false;

  std::unique_ptr<metal_rt::IMetalPostProcessPreset> standardPreset_;
  std::unique_ptr<metal_rt::IMetalPostProcessPreset> toonPreset_;

  RTScene scene_;
  RTCamera camera_;
  mutable RenderBuffer latestBuffer_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t frameIndex_ = 0;
};

} // namespace rt::metal_backend_internal
