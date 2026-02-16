# DeepFilterNet3 Standalone

A standalone implementation of DeepFilterNet3 audio denoising that provides **direct buffer-to-buffer processing** without requiring AudioWorklet or LiveKit infrastructure.

## 🎯 Key Features

- **Zero AudioWorklet overhead** - Direct WASM-based processing
- **Simple API** - Just `processAudio(buffer)` to denoise
- **Pre-initialization** - Load WASM once at app startup, instant processing thereafter
- **Perfect for push-to-talk** - No activation delay after initialization

## 📦 Installation

```bash
npm install
```

## 🚀 Quick Start

```typescript
import { StandaloneDeepFilter } from './src/StandaloneDeepFilter';

// Initialize once at app startup
const denoiser = new StandaloneDeepFilter({
  attenuationLimit: 50,  // Noise reduction strength (0-100 dB)
  postFilterBeta: 0.02   // Post filter (0-0.05, 0 disables)
});

await denoiser.initialize();
console.log('Denoiser ready!');

// Later, when you have audio to process:
const noisyAudio = new Float32Array(/* your 48kHz mono audio */);
const cleanAudio = denoiser.processAudio(noisyAudio);

// Or with statistics:
const { audio, stats } = denoiser.processAudioWithStats(noisyAudio);
console.log(`Processed ${stats.inputSamples} samples in ${stats.processingTimeMs}ms`);
```

## 🛠️ API Reference

### `StandaloneDeepFilter`

#### Constructor Options

```typescript
interface DeepFilterConfig {
  cdnUrl?: string;           // CDN for WASM/model (default: official CDN)
  attenuationLimit?: number; // Noise reduction in dB (default: 50)
  postFilterBeta?: number;   // Post filter strength (default: 0.02)
}
```

#### Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Load WASM and model. Call once at startup. |
| `processAudio(input: Float32Array)` | Denoise audio buffer (48kHz mono) |
| `processAudioWithStats(input)` | Denoise with timing statistics |
| `setAttenuationLimit(db: number)` | Adjust noise reduction strength |
| `setPostFilterBeta(beta: number)` | Adjust post filter |
| `getFrameLength()` | Get the WASM model's frame size |
| `isReady()` | Check if initialized |
| `destroy()` | Cleanup resources |

## 🎮 Demo

Run the interactive demo:

```bash
npm run demo
```

This opens a browser with:
- 🎙️ Microphone recording with instant denoising
- 📁 Audio file upload and processing
- ⚙️ Real-time parameter adjustment
- 📊 Processing statistics

## 📐 Architecture

This package extracts the core WASM processing from [livekit-deepfilternet3-noise-filter](https://github.com/phuvinh010701/livekit-deepfilternet3-noise-filter) and provides a simpler interface:

```
Original Package:
┌─────────────────────────────────────────────┐
│  LiveKit TrackProcessor                      │
│  └── AudioWorkletNode                        │
│      └── AudioWorkletProcessor               │
│          └── WASM (df_process_frame)         │  ← We use this directly!
└─────────────────────────────────────────────┘

This Package:
┌─────────────────────────────────────────────┐
│  StandaloneDeepFilter                        │
│  └── WASM (df_process_frame)                 │  ← Direct access
└─────────────────────────────────────────────┘
```

## ⚠️ Requirements

- **Sample Rate**: 48kHz (resample if needed)
- **Channels**: Mono (mix down stereo if needed)
- **Browser**: Modern browser with WebAssembly support

## 📄 License


MIT License

## Attribution & License Compliance

This package reuses and builds upon code from [livekit-deepfilternet3-noise-filter](https://github.com/phuvinh010701/livekit-deepfilternet3-noise-filter), which is licensed under the MIT License. Portions of WASM processing and model loading logic are directly adapted from the original project.

We gratefully acknowledge the authors and contributors of livekit-deepfilternet3-noise-filter. Please see their repository for full license and copyright information.

You must retain the original MIT license and attribution if redistributing or modifying this package.

---
MIT - Based on DeepFilterNet3 and livekit-deepfilternet3-noise-filter
