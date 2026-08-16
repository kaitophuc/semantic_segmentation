#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace
{

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ T normalize_channel(uint8_t value, int channel);

template <>
__device__ __forceinline__ float normalize_channel<float>(uint8_t value, int channel)
{
    constexpr float scale[3] = {
        1.0F / (255.0F * 0.229F),
        1.0F / (255.0F * 0.224F),
        1.0F / (255.0F * 0.225F),
    };
    constexpr float bias[3] = {
        -0.485F / 0.229F,
        -0.456F / 0.224F,
        -0.406F / 0.225F,
    };
    return static_cast<float>(value) * scale[channel] + bias[channel];
}

template <>
__device__ __forceinline__ half normalize_channel<half>(uint8_t value, int channel)
{
    return __float2half_rn(normalize_channel<float>(value, channel));
}

template <typename T>
__global__ void preprocess_bgr_to_rgb_chw_kernel(
    uint8_t const* __restrict__ frame,
    T* __restrict__ output,
    int pixels,
    int frame_channels)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels)
    {
        return;
    }

    uint8_t const* pixel = frame + static_cast<int64_t>(index) * frame_channels;
    output[index] = normalize_channel<T>(pixel[2], 0);
    output[pixels + index] = normalize_channel<T>(pixel[1], 1);
    output[2 * pixels + index] = normalize_channel<T>(pixel[0], 2);
}

template <typename T>
__device__ __forceinline__ float as_float(T value);

template <>
__device__ __forceinline__ float as_float<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ float as_float<half>(half value)
{
    return __half2float(value);
}

__device__ __forceinline__ uint8_t blend_channel(
    uint8_t source,
    float foreground,
    float alpha)
{
    float const value = static_cast<float>(source) * (1.0F - alpha) + foreground * alpha;
    return static_cast<uint8_t>(__float2uint_rn(fminf(fmaxf(value, 0.0F), 255.0F)));
}

template <typename T>
__global__ void overlay_logits_kernel(
    uint8_t const* __restrict__ frame,
    T const* __restrict__ logits,
    uint8_t* __restrict__ overlay,
    int frame_height,
    int frame_width,
    int frame_channels,
    int logits_height,
    int logits_width,
    float alpha)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    int const pixels = frame_height * frame_width;
    if (index >= pixels)
    {
        return;
    }

    int const y = index / frame_width;
    int const x = index - y * frame_width;
    int const logits_y = min(
        static_cast<int>((static_cast<int64_t>(y) * logits_height) / frame_height),
        logits_height - 1);
    int const logits_x = min(
        static_cast<int>((static_cast<int64_t>(x) * logits_width) / frame_width),
        logits_width - 1);
    int const logits_index = logits_y * logits_width + logits_x;
    int const logits_pixels = logits_height * logits_width;
    bool const drivable =
        as_float(logits[logits_pixels + logits_index]) > as_float(logits[logits_index]);

    uint8_t const* source = frame + static_cast<int64_t>(index) * frame_channels;
    uint8_t* destination = overlay + static_cast<int64_t>(index) * frame_channels;
    if (drivable)
    {
        destination[0] = blend_channel(source[0], 0.0F, alpha);
        destination[1] = blend_channel(source[1], 255.0F, alpha);
        destination[2] = blend_channel(source[2], 0.0F, alpha);
    }
    else
    {
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
    }

    if (frame_channels == 4)
    {
        destination[3] = 255;
    }
}

template <typename T>
cudaError_t launch_preprocess(
    uint8_t const* frame,
    void* output,
    int height,
    int width,
    int frame_channels,
    cudaStream_t stream)
{
    int const pixels = height * width;
    int const blocks = (pixels + kThreads - 1) / kThreads;
    preprocess_bgr_to_rgb_chw_kernel<T><<<blocks, kThreads, 0, stream>>>(
        frame,
        static_cast<T*>(output),
        pixels,
        frame_channels);
    return cudaPeekAtLastError();
}

template <typename T>
cudaError_t launch_overlay(
    uint8_t const* frame,
    void const* logits,
    uint8_t* overlay,
    int frame_height,
    int frame_width,
    int frame_channels,
    int logits_height,
    int logits_width,
    float alpha,
    cudaStream_t stream)
{
    int const pixels = frame_height * frame_width;
    int const blocks = (pixels + kThreads - 1) / kThreads;
    overlay_logits_kernel<T><<<blocks, kThreads, 0, stream>>>(
        frame,
        static_cast<T const*>(logits),
        overlay,
        frame_height,
        frame_width,
        frame_channels,
        logits_height,
        logits_width,
        alpha);
    return cudaPeekAtLastError();
}

} // namespace

extern "C" cudaError_t preprocess_bgr_to_rgb_chw(
    uint8_t const* frame,
    void* output,
    int height,
    int width,
    int frame_channels,
    int output_is_fp16,
    cudaStream_t stream)
{
    if (frame == nullptr || output == nullptr || height <= 0 || width <= 0
        || (frame_channels != 3 && frame_channels != 4))
    {
        return cudaErrorInvalidValue;
    }

    if (output_is_fp16 != 0)
    {
        return launch_preprocess<half>(
            frame, output, height, width, frame_channels, stream);
    }
    return launch_preprocess<float>(
        frame, output, height, width, frame_channels, stream);
}

extern "C" cudaError_t overlay_binary_logits(
    uint8_t const* frame,
    void const* logits,
    uint8_t* overlay,
    int frame_height,
    int frame_width,
    int frame_channels,
    int logits_height,
    int logits_width,
    int logits_are_fp16,
    float alpha,
    cudaStream_t stream)
{
    if (frame == nullptr || logits == nullptr || overlay == nullptr
        || frame_height <= 0 || frame_width <= 0
        || logits_height <= 0 || logits_width <= 0
        || (frame_channels != 3 && frame_channels != 4))
    {
        return cudaErrorInvalidValue;
    }

    if (logits_are_fp16 != 0)
    {
        return launch_overlay<half>(
            frame,
            logits,
            overlay,
            frame_height,
            frame_width,
            frame_channels,
            logits_height,
            logits_width,
            alpha,
            stream);
    }
    return launch_overlay<float>(
        frame,
        logits,
        overlay,
        frame_height,
        frame_width,
        frame_channels,
        logits_height,
        logits_width,
        alpha,
        stream);
}
