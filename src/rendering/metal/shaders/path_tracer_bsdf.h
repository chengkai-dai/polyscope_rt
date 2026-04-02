#pragma once

float3 fresnelSchlick(float cosTheta, float3 F0) {
  return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

float3 fresnelSchlick(float cosTheta, float3 F0, float grazingMax) {
  return F0 + (float3(grazingMax) - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

float distributionGGX(float3 N, float3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0f);
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
  return a2 / max(3.14159265f * denom * denom, 1e-5f);
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0f;
  float k = (r * r) / 8.0f;
  return NdotV / max(NdotV * (1.0f - k) + k, 1e-5f);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

float3 evaluateEnvironmentRadiance(float3 dir, constant GPULighting& lighting) {
  return rtEvaluateEnvironmentRadiance(lighting.backgroundColor.xyz,
                                       lighting.environmentTintIntensity.xyz,
                                       lighting.environmentTintIntensity.w,
                                       lighting.sceneUpDir.xyz,
                                       normalize(dir));
}

float3 sampleGGXBounce(float3 N, float3 V, float roughness, thread uint& rng);

float luminance(float3 c) {
  return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

OpaqueBsdfSamplingWeights computeOpaqueBsdfSamplingWeights(float3 F, float3 baseColor, float metallic) {
  OpaqueBsdfSamplingWeights weights;
  float3 kD = (1.0f - F) * (1.0f - metallic);
  weights.specularWeight = luminance(max(F, float3(0.0f)));
  weights.diffuseWeight = luminance(max(baseColor * kD, float3(0.0f)));

  float totalWeight = weights.specularWeight + weights.diffuseWeight;
  if (totalWeight <= 1e-6f || weights.diffuseWeight <= 1e-6f) {
    weights.diffuseProb = 0.0f;
    weights.specularProb = 1.0f;
    return weights;
  }

  weights.specularProb = clamp(weights.specularWeight / totalWeight, 0.04f, 0.999f);
  weights.diffuseProb = 1.0f - weights.specularProb;
  return weights;
}

bool shouldSkipLightHitMISForOpaqueSample(bool sampledSpecular, OpaqueBsdfSamplingWeights weights) {
  // Single-lobe opaque materials only continue through the specular branch in
  // this BSDF model, so visible-light hits should be attributed to the BSDF
  // continuation directly instead of being penalized against a non-existent
  // diffuse light-sampling alternative.
  return sampledSpecular && weights.diffuseWeight <= 1e-6f;
}

float3 cosineWeightedHemisphere(float3 normal, thread uint& rng) {
  float u1 = rand01(rng);
  float u2 = rand01(rng);
  float r = sqrt(u1);
  float theta = 2.0f * 3.14159265f * u2;
  float z = sqrt(max(0.0f, 1.0f - u1));
  float3 up = abs(normal.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
  float3 tangent = normalize(cross(up, normal));
  float3 bitangent = cross(normal, tangent);
  return normalize(tangent * (r * cos(theta)) + bitangent * (r * sin(theta)) + normal * z);
}

// GGX Visible Normal Distribution Function (VNDF) importance sampling
// Reference: "Sampling the GGX Distribution of Visible Normals" (Heitz 2018)
float3 sampleGGXVNDF(float3 Ve, float alpha, float u1, float u2) {
  float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
  float3 T2 = cross(Vh, T1);
  float r = sqrt(u1);
  float phi = 2.0f * 3.14159265f * u2;
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

float3 sampleGGXBounce(float3 N, float3 V, float roughness, thread uint& rng) {
  float alpha = max(roughness * roughness, 0.001f);
  float3 up = abs(N.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
  float3 T = normalize(cross(up, N));
  float3 B = cross(N, T);
  float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));
  float3 H_local = sampleGGXVNDF(Ve, alpha, rand01(rng), rand01(rng));
  float3 H = T * H_local.x + B * H_local.y + N * H_local.z;
  float3 L = reflect(-V, H);
  return L;
}

float powerHeuristic(float pdfA, float pdfB) {
  float a2 = pdfA * pdfA;
  float b2 = pdfB * pdfB;
  return a2 / max(a2 + b2, 1e-8f);
}

float ggxReflectionPdf(float3 N, float3 V, float3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

  float3 H = V + L;
  float HLen2 = dot(H, H);
  if (HLen2 <= 1e-8f) return 0.0f;
  H *= rsqrt(HLen2);

  float NdotH = max(dot(N, H), 0.0f);
  if (NdotH <= 0.0f) return 0.0f;

  float D = distributionGGX(N, H, roughness);
  float G1 = geometrySchlickGGX(NdotV, roughness);
  return D * G1 * NdotH / max(4.0f * NdotV, 1e-6f);
}

float environmentBinSolidAngle(uint row) {
  float theta0 = kPi * float(row) / float(kEnvironmentSampleHeight);
  float theta1 = kPi * float(row + 1u) / float(kEnvironmentSampleHeight);
  return (2.0f * kPi / float(kEnvironmentSampleWidth)) * max(cos(theta0) - cos(theta1), 1e-6f);
}

uint environmentCellIndex(float3 dir) {
  float3 sampleDir = normalize(dir);
  float phi = atan2(sampleDir.z, sampleDir.x);
  if (phi < 0.0f) phi += 2.0f * kPi;
  float theta = acos(clamp(sampleDir.y, -1.0f, 1.0f));
  uint col = min(uint(phi / (2.0f * kPi) * float(kEnvironmentSampleWidth)), kEnvironmentSampleWidth - 1u);
  uint row = min(uint(theta / kPi * float(kEnvironmentSampleHeight)), kEnvironmentSampleHeight - 1u);
  return row * kEnvironmentSampleWidth + col;
}

float environmentDirectionalPdf(float3 dir, device const GPUEnvironmentSampleCell* environmentCells) {
  uint idx = environmentCellIndex(dir);
  uint row = idx / kEnvironmentSampleWidth;
  float pmf = environmentCells[idx].data.x;
  if (pmf <= 0.0f) return 0.0f;
  return pmf / environmentBinSolidAngle(row);
}

BsdfEvalResult evaluateOpaqueBsdf(float3 N, float3 V, float3 L, float3 albedo, float metallic,
                                  float roughness, float dielectricF0) {
  BsdfEvalResult out;
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  if (NdotV <= 0.0f || NdotL <= 0.0f) return out;

  float3 F0 = mix(float3(dielectricF0), albedo, metallic);
  float grazingMax = mix(clamp(dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, metallic);
  float3 H = normalize(V + L);
  float NDF = distributionGGX(N, H, roughness);
  float G = geometrySmith(N, V, L, roughness);
  float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0, grazingMax);
  float3 specular = (NDF * G * F) / max(4.0f * NdotV * NdotL, 1e-4f);
  float3 kS = F;
  float3 kD = (1.0f - kS) * (1.0f - metallic);
  float3 diffuse = kD * albedo / kPi;

  float3 viewF = fresnelSchlick(NdotV, F0, grazingMax);
  OpaqueBsdfSamplingWeights weights = computeOpaqueBsdfSamplingWeights(viewF, albedo, metallic);

  out.value = diffuse + specular;
  out.diffuse = diffuse;
  out.specular = specular;
  out.fresnel = F;
  out.diffusePdf = NdotL / kPi;
  out.specularPdf = ggxReflectionPdf(N, V, L, roughness);
  out.pdf = weights.diffuseProb * out.diffusePdf + weights.specularProb * out.specularPdf;
  out.NdotL = NdotL;
  return out;
}

BsdfEvalResult evaluateDielectricSpecularBsdf(float3 N, float3 V, float3 L, float roughness, float ior) {
  BsdfEvalResult out;
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  if (NdotV <= 0.0f || NdotL <= 0.0f) return out;

  float f0Scalar = pow((ior - 1.0f) / (ior + 1.0f), 2.0f);
  float3 F0 = float3(f0Scalar);
  float3 H = normalize(V + L);
  float NDF = distributionGGX(N, H, roughness);
  float G = geometrySmith(N, V, L, roughness);
  float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);

  out.value = (NDF * G * F) / max(4.0f * NdotV * NdotL, 1e-4f);
  out.specular = out.value;
  out.fresnel = F;
  out.specularPdf = ggxReflectionPdf(N, V, L, roughness);
  out.pdf = out.specularPdf;
  out.NdotL = NdotL;
  return out;
}

BsdfSampleResult sampleOpaqueBsdf(float3 N, float3 V, float3 albedo, float metallic,
                                  float roughness, float dielectricF0, thread uint& rng) {
  BsdfSampleResult out;
  float3 F0 = mix(float3(dielectricF0), albedo, metallic);
  float grazingMax = mix(clamp(dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, metallic);
  float NdotV = max(dot(N, V), 0.0f);
  float3 viewF = fresnelSchlick(NdotV, F0, grazingMax);
  OpaqueBsdfSamplingWeights weights = computeOpaqueBsdfSamplingWeights(viewF, albedo, metallic);
  bool sampledSpecular = rand01(rng) < weights.specularProb;
  if (sampledSpecular) {
    out.direction = sampleGGXBounce(N, V, roughness, rng);
    if (dot(out.direction, N) <= 0.0f) out.direction = reflect(-V, N);
  } else {
    out.direction = cosineWeightedHemisphere(N, rng);
  }

  BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, out.direction, albedo, metallic, roughness, dielectricF0);
  if (eval.NdotL <= 0.0f || eval.pdf <= 0.0f || all(eval.value <= 0.0f)) return out;

  out.valid = true;
  out.skipLightHitMIS = shouldSkipLightHitMISForOpaqueSample(sampledSpecular, weights);
  out.pdf = eval.pdf;
  out.throughput = eval.value * eval.NdotL / eval.pdf;
  return out;
}
