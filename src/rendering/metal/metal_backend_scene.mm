#include "rendering/metal/metal_backend_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "rendering/metal/metal_device.h"
#include "rendering/metal/metal_scene_builder.h"

namespace {

void normalizeEmissiveTriangleDistribution(std::vector<GPUEmissiveTriangle>& emissiveTriangles) {
  if (emissiveTriangles.empty()) return;

  double totalWeight = 0.0;
  for (const GPUEmissiveTriangle& tri : emissiveTriangles) {
    totalWeight += static_cast<double>(tri.params.y);
  }

  if (totalWeight <= 1e-8) {
    for (size_t i = 0; i < emissiveTriangles.size(); ++i) {
      emissiveTriangles[i].params.y = 0.0f;
      emissiveTriangles[i].params.z = (i == emissiveTriangles.size() - 1u) ? 1.0f : 0.0f;
    }
    return;
  }

  float cumulative = 0.0f;
  for (GPUEmissiveTriangle& tri : emissiveTriangles) {
    const float selectionPdf = static_cast<float>(static_cast<double>(tri.params.y) / totalWeight);
    cumulative += selectionPdf;
    tri.params.y = selectionPdf;
    tri.params.z = cumulative;
  }
  emissiveTriangles.back().params.z = 1.0f;
}

} // namespace

namespace rt::metal_backend_internal {

void MetalPathTracerBackend::buildSceneBuffers() {
  PackedSceneData packed = packScene(scene_);
  SceneGpuAccumulator& acc = packed.acc;
  curveControlPoints_ = std::move(packed.curveControlPoints);
  curveRadii_ = std::move(packed.curveRadii);
  pointPrimitives_ = std::move(packed.pointPrimitives);
  pointBboxData_ = std::move(packed.pointBoundingBoxes);

  if (acc.positions.empty()) {
    for (int j = 0; j < 3; ++j) {
      acc.positions.push_back(simd_make_float4(1e6f, 1e6f, 1e6f, 1.0f));
      acc.normals.push_back(simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f));
      acc.vertexColors.push_back(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f));
      acc.texcoords.push_back(simd_make_float2(0.0f, 0.0f));
      acc.isoScalars.push_back(0.0f);
    }
    acc.accelIndices.push_back({0, 1, 2});
    GPUMaterial dummyMat{};
    acc.materials.push_back(dummyMat);
    GPUTriangle dummyTri{};
    dummyTri.indicesMaterial = simd_make_uint4(0, 1, 2, static_cast<uint32_t>(acc.materials.size() - 1));
    acc.shaderTriangles.push_back(dummyTri);
  }

  uploadSceneBuffers(acc);
  buildAccelerationStructure(static_cast<uint32_t>(acc.accelIndices.size()));
}

void MetalPathTracerBackend::uploadSceneBuffers(SceneGpuAccumulator& acc) {
  positionBuffer_ = [device_ newBufferWithBytes:acc.positions.data()
                                         length:acc.positions.size() * sizeof(simd_float4)
                                        options:MTLResourceStorageModeShared];
  normalVertexBuffer_ = [device_ newBufferWithBytes:acc.normals.data()
                                             length:acc.normals.size() * sizeof(simd_float4)
                                            options:MTLResourceStorageModeShared];
  vertexColorBuffer_ = [device_ newBufferWithBytes:acc.vertexColors.data()
                                            length:acc.vertexColors.size() * sizeof(simd_float4)
                                           options:MTLResourceStorageModeShared];
  texcoordBuffer_ = [device_ newBufferWithBytes:acc.texcoords.data()
                                         length:acc.texcoords.size() * sizeof(simd_float2)
                                        options:MTLResourceStorageModeShared];

  while (acc.isoScalars.size() < acc.positions.size()) acc.isoScalars.push_back(0.0f);
  isoScalarsBuffer_ = [device_ newBufferWithBytes:acc.isoScalars.data()
                                           length:acc.isoScalars.size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
  accelIndexBuffer_ = [device_ newBufferWithBytes:acc.accelIndices.data()
                                           length:acc.accelIndices.size() * sizeof(PackedTriangleIndices)
                                          options:MTLResourceStorageModeShared];
  triangleBuffer_ = [device_ newBufferWithBytes:acc.shaderTriangles.data()
                                         length:acc.shaderTriangles.size() * sizeof(GPUTriangle)
                                        options:MTLResourceStorageModeShared];
  materialBuffer_ = [device_ newBufferWithBytes:acc.materials.data()
                                         length:acc.materials.size() * sizeof(GPUMaterial)
                                        options:MTLResourceStorageModeShared];

  if (acc.textures.empty()) {
    GPUTexture defaultTexture;
    defaultTexture.data = simd_make_uint4(0u, 0u, 0u, 0u);
    acc.textures.push_back(defaultTexture);
  }
  if (acc.texturePixels.empty()) {
    acc.texturePixels.push_back(simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f));
  }
  textureMetadataBuffer_ = [device_ newBufferWithBytes:acc.textures.data()
                                                length:acc.textures.size() * sizeof(GPUTexture)
                                               options:MTLResourceStorageModeShared];
  texturePixelBuffer_ = [device_ newBufferWithBytes:acc.texturePixels.data()
                                             length:acc.texturePixels.size() * sizeof(simd_float4)
                                            options:MTLResourceStorageModeShared];

  if (acc.lights.empty()) {
    GPUPunctualLight defaultLight{};
    acc.lights.push_back(defaultLight);
  }
  lightBuffer_ = [device_ newBufferWithBytes:acc.lights.data()
                                      length:acc.lights.size() * sizeof(GPUPunctualLight)
                                     options:MTLResourceStorageModeShared];

  if (acc.emissiveTriangles.empty()) {
    GPUEmissiveTriangle dummy{};
    dummy.params = simd_make_float4(1.0f, 0.0f, 1.0f, 0.0f);
    acc.emissiveTriangles.push_back(dummy);
    emissiveTriangleCount_ = 0u;
  } else {
    normalizeEmissiveTriangleDistribution(acc.emissiveTriangles);
    emissiveTriangleCount_ = static_cast<uint32_t>(acc.emissiveTriangles.size());
  }
  emissiveTriangleBuffer_ = [device_ newBufferWithBytes:acc.emissiveTriangles.data()
                                                 length:acc.emissiveTriangles.size() * sizeof(GPUEmissiveTriangle)
                                                options:MTLResourceStorageModeShared];

  if (!acc.curvePrimitives.empty()) {
    curvePrimitiveBuffer_ = [device_ newBufferWithBytes:acc.curvePrimitives.data()
                                                 length:acc.curvePrimitives.size() * sizeof(GPUCurvePrimitive)
                                                options:MTLResourceStorageModeShared];
    curveControlPointBuffer_ = [device_ newBufferWithBytes:curveControlPoints_.data()
                                                    length:curveControlPoints_.size() * sizeof(simd_float3)
                                                   options:MTLResourceStorageModeShared];
    curveRadiusBuffer_ = [device_ newBufferWithBytes:curveRadii_.data()
                                              length:curveRadii_.size() * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    curveSegmentCount_ = static_cast<uint32_t>(acc.curvePrimitives.size());
  } else {
    GPUCurvePrimitive dummy{};
    curvePrimitiveBuffer_ = [device_ newBufferWithBytes:&dummy
                                                 length:sizeof(GPUCurvePrimitive)
                                                options:MTLResourceStorageModeShared];
    curveControlPointBuffer_ = nil;
    curveRadiusBuffer_ = nil;
    curveSegmentCount_ = 0;
  }

  if (!pointPrimitives_.empty()) {
    const std::vector<MTLAxisAlignedBoundingBox> metalPointBboxes =
        metal_rt::makeMetalBoundingBoxes(pointBboxData_);
    pointPrimitiveBuffer_ = [device_ newBufferWithBytes:pointPrimitives_.data()
                                                 length:pointPrimitives_.size() * sizeof(GPUPointPrimitive)
                                                options:MTLResourceStorageModeShared];
    pointBboxBuffer_ = [device_ newBufferWithBytes:metalPointBboxes.data()
                                            length:metalPointBboxes.size() * sizeof(MTLAxisAlignedBoundingBox)
                                           options:MTLResourceStorageModeShared];
  } else {
    GPUPointPrimitive dummy{};
    pointPrimitiveBuffer_ = [device_ newBufferWithBytes:&dummy
                                                 length:sizeof(GPUPointPrimitive)
                                                options:MTLResourceStorageModeShared];
    pointBboxBuffer_ = nil;
  }

  if (intersectionFunctionTable_ != nil && pointPrimitiveBuffer_ != nil) {
    [intersectionFunctionTable_ setBuffer:pointPrimitiveBuffer_ offset:0 atIndex:25];
  }
}

bool MetalPathTracerBackend::sceneContainsTransmission() const {
  for (const RTMesh& mesh : scene_.meshes) {
    if (mesh.transmissionFactor > 1e-4f) return true;
  }
  return false;
}

id<MTLAccelerationStructure>
MetalPathTracerBackend::buildAndCompactBLAS(MTLPrimitiveAccelerationStructureDescriptor* descriptor) {
  MTLAccelerationStructureSizes sizes = [device_ accelerationStructureSizesWithDescriptor:descriptor];
  id<MTLAccelerationStructure> uncompacted = [device_ newAccelerationStructureWithSize:sizes.accelerationStructureSize];
  id<MTLBuffer> scratch = [device_ newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];
  id<MTLBuffer> compactedSizeBuf = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
  id<MTLAccelerationStructureCommandEncoder> enc = [cmdBuf accelerationStructureCommandEncoder];
  [enc buildAccelerationStructure:uncompacted descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
  [enc writeCompactedAccelerationStructureSize:uncompacted toBuffer:compactedSizeBuf offset:0];
  [enc endEncoding];
  [cmdBuf commit];
  [cmdBuf waitUntilCompleted];

  uint32_t compactedSize = 0;
  std::memcpy(&compactedSize, compactedSizeBuf.contents, sizeof(uint32_t));

  id<MTLAccelerationStructure> compacted = [device_ newAccelerationStructureWithSize:compactedSize];

  id<MTLCommandBuffer> compactCmd = [commandQueue_ commandBuffer];
  id<MTLAccelerationStructureCommandEncoder> compactEnc = [compactCmd accelerationStructureCommandEncoder];
  [compactEnc copyAndCompactAccelerationStructure:uncompacted toAccelerationStructure:compacted];
  [compactEnc endEncoding];
  [compactCmd commit];
  [compactCmd waitUntilCompleted];

  return compacted;
}

void MetalPathTracerBackend::buildAccelerationStructure(uint32_t triangleCount) {
  @autoreleasepool {
    auto* triGeom = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    triGeom.vertexBuffer = positionBuffer_;
    triGeom.vertexStride = sizeof(simd_float4);
    triGeom.vertexFormat = MTLAttributeFormatFloat3;
    triGeom.indexBuffer = accelIndexBuffer_;
    triGeom.indexType = MTLIndexTypeUInt32;
    triGeom.triangleCount = triangleCount;
    triGeom.opaque = YES;

    auto* triDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    triDesc.geometryDescriptors = @[ triGeom ];
    triDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

    triangleBLAS_ = buildAndCompactBLAS(triDesc);

    bool hasCurves = curveSegmentCount_ > 0 && curveControlPointBuffer_ != nil && curveRadiusBuffer_ != nil;

    if (hasCurves) {
      std::vector<uint32_t> curveIndices(curveSegmentCount_);
      for (uint32_t i = 0; i < curveSegmentCount_; ++i) curveIndices[i] = i * 4;
      id<MTLBuffer> curveIndexBuf = [device_ newBufferWithBytes:curveIndices.data()
                                                        length:curveIndices.size() * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];

      auto* curveGeom = [MTLAccelerationStructureCurveGeometryDescriptor descriptor];
      curveGeom.controlPointBuffer = curveControlPointBuffer_;
      curveGeom.controlPointCount = curveControlPoints_.size();
      curveGeom.controlPointStride = sizeof(simd_float3);
      curveGeom.controlPointFormat = MTLAttributeFormatFloat3;
      curveGeom.controlPointBufferOffset = 0;
      curveGeom.radiusBuffer = curveRadiusBuffer_;
      curveGeom.radiusBufferOffset = 0;
      curveGeom.indexBuffer = curveIndexBuf;
      curveGeom.indexType = MTLIndexTypeUInt32;
      curveGeom.segmentCount = curveSegmentCount_;
      curveGeom.segmentControlPointCount = 4;
      curveGeom.curveBasis = MTLCurveBasisCatmullRom;
      curveGeom.curveType = MTLCurveTypeRound;
      curveGeom.curveEndCaps = MTLCurveEndCapsNone;
      curveGeom.intersectionFunctionTableOffset = NSUIntegerMax;

      auto* curveDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      curveDesc.geometryDescriptors = @[ curveGeom ];
      curveDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

      curveBLAS_ = buildAndCompactBLAS(curveDesc);
    } else {
      curveBLAS_ = nil;
    }

    bool hasPoints = !pointPrimitives_.empty() && pointBboxBuffer_ != nil;

    auto buildPointIas = [&]() -> id<MTLAccelerationStructure> {
      id<MTLBuffer> bboxBuf = nil;
      NSUInteger bboxCount = 0;

      if (hasPoints) {
        bboxBuf = pointBboxBuffer_;
        bboxCount = static_cast<NSUInteger>(pointPrimitives_.size());
      } else {
        MTLAxisAlignedBoundingBox dummyBox = {{9.99e9f, 9.99e9f, 9.99e9f},
                                              {1.001e10f, 1.001e10f, 1.001e10f}};
        bboxBuf = [device_ newBufferWithBytes:&dummyBox
                                       length:sizeof(dummyBox)
                                      options:MTLResourceStorageModeShared];
        bboxCount = 1;
      }

      auto* bboxGeom = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
      bboxGeom.boundingBoxBuffer = bboxBuf;
      bboxGeom.boundingBoxCount = bboxCount;
      bboxGeom.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
      bboxGeom.intersectionFunctionTableOffset = 0;

      auto* ptDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      ptDesc.geometryDescriptors = @[ bboxGeom ];
      ptDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;
      pointBLAS_ = buildAndCompactBLAS(ptDesc);

      MTLAccelerationStructureInstanceDescriptor ptInst{};
      ptInst.transformationMatrix.columns[0] = {1.f, 0.f, 0.f};
      ptInst.transformationMatrix.columns[1] = {0.f, 1.f, 0.f};
      ptInst.transformationMatrix.columns[2] = {0.f, 0.f, 1.f};
      ptInst.transformationMatrix.columns[3] = {0.f, 0.f, 0.f};
      ptInst.mask = 0xFF;
      ptInst.options = MTLAccelerationStructureInstanceOptionNone;
      ptInst.intersectionFunctionTableOffset = 0;
      ptInst.accelerationStructureIndex = 0;

      id<MTLBuffer> ptInstBuf = [device_ newBufferWithBytes:&ptInst
                                                     length:sizeof(ptInst)
                                                    options:MTLResourceStorageModeShared];
      auto* ptIasDesc = [[MTLInstanceAccelerationStructureDescriptor alloc] init];
      ptIasDesc.instancedAccelerationStructures = @[ pointBLAS_ ];
      ptIasDesc.instanceCount = 1;
      ptIasDesc.instanceDescriptorBuffer = ptInstBuf;
      ptIasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
      ptIasDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

      MTLAccelerationStructureSizes ptSizes = [device_ accelerationStructureSizesWithDescriptor:ptIasDesc];
      id<MTLAccelerationStructure> ptIas = [device_ newAccelerationStructureWithSize:ptSizes.accelerationStructureSize];
      id<MTLBuffer> ptScratch = [device_ newBufferWithLength:ptSizes.buildScratchBufferSize
                                                     options:MTLResourceStorageModePrivate];
      id<MTLCommandBuffer> ptCmd = [commandQueue_ commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> ptEnc = [ptCmd accelerationStructureCommandEncoder];
      [ptEnc buildAccelerationStructure:ptIas descriptor:ptIasDesc
                           scratchBuffer:ptScratch scratchBufferOffset:0];
      [ptEnc endEncoding];
      [ptCmd commit];
      [ptCmd waitUntilCompleted];
      return ptIas;
    };
    pointAcceleration_ = buildPointIas();

    NSUInteger mcInstanceCount = 1 + (hasCurves ? 1 : 0);
    std::vector<MTLAccelerationStructureInstanceDescriptor> mcInstances(mcInstanceCount);
    std::memset(mcInstances.data(), 0, mcInstances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor));

    for (NSUInteger i = 0; i < mcInstanceCount; ++i) {
      mcInstances[i].transformationMatrix.columns[0] = {1.0f, 0.0f, 0.0f};
      mcInstances[i].transformationMatrix.columns[1] = {0.0f, 1.0f, 0.0f};
      mcInstances[i].transformationMatrix.columns[2] = {0.0f, 0.0f, 1.0f};
      mcInstances[i].transformationMatrix.columns[3] = {0.0f, 0.0f, 0.0f};
      mcInstances[i].mask = 0xFF;
      mcInstances[i].options = MTLAccelerationStructureInstanceOptionNone;
      mcInstances[i].intersectionFunctionTableOffset = 0;
    }
    NSUInteger nextMcInstance = 0;
    mcInstances[nextMcInstance++].accelerationStructureIndex = 0;

    NSMutableArray* mcBlasArray = [NSMutableArray arrayWithObject:triangleBLAS_];
    if (hasCurves) {
      mcInstances[nextMcInstance++].accelerationStructureIndex = static_cast<uint32_t>(mcBlasArray.count);
      [mcBlasArray addObject:curveBLAS_];
    }

    id<MTLBuffer> mcInstanceBuf = [device_ newBufferWithBytes:mcInstances.data()
                                                        length:mcInstances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor)
                                                       options:MTLResourceStorageModeShared];

    auto* mcIasDesc = [[MTLInstanceAccelerationStructureDescriptor alloc] init];
    mcIasDesc.instancedAccelerationStructures = mcBlasArray;
    mcIasDesc.instanceCount = mcInstanceCount;
    mcIasDesc.instanceDescriptorBuffer = mcInstanceBuf;
    mcIasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
    mcIasDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

    MTLAccelerationStructureSizes mcSizes = [device_ accelerationStructureSizesWithDescriptor:mcIasDesc];
    meshCurveAcceleration_ = [device_ newAccelerationStructureWithSize:mcSizes.accelerationStructureSize];
    id<MTLBuffer> mcScratch = [device_ newBufferWithLength:mcSizes.buildScratchBufferSize
                                                   options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> mcCmdBuf = [commandQueue_ commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> mcEnc = [mcCmdBuf accelerationStructureCommandEncoder];
    [mcEnc buildAccelerationStructure:meshCurveAcceleration_
                           descriptor:mcIasDesc
                         scratchBuffer:mcScratch
                   scratchBufferOffset:0];
    [mcEnc endEncoding];
    [mcCmdBuf commit];
    [mcCmdBuf waitUntilCompleted];
  }
}

} // namespace rt::metal_backend_internal
