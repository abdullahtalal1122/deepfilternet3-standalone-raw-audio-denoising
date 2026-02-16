/**
 * StandaloneDeepFilter - Direct buffer-to-buffer audio denoising using DeepFilterNet3
 * 
 * This class provides a simple interface to denoise audio without requiring
 * AudioWorklet or LiveKit infrastructure. Perfect for push-to-talk applications.
 * 
 * NEW: Streaming mode for continuous audio processing with maintained neural network state.
 * 
 * Usage (batch mode - for complete audio files):
 *   const denoiser = new StandaloneDeepFilter();
 *   await denoiser.initialize();
 *   const cleanAudio = denoiser.processAudio(noisyAudioBuffer);
 * 
 * Usage (streaming mode - for real-time audio chunks):
 *   const denoiser = new StandaloneDeepFilter();
 *   await denoiser.initialize();
 *   denoiser.startStreaming(); // Call once at start of recording session
 *   // For each incoming audio chunk:
 *   const cleanChunk = denoiser.processStreaming(noisyChunk);
 *   denoiser.stopStreaming(); // Call when recording ends
 */

import * as wasm_bindgen from './df3/df.js';

export interface DeepFilterConfig {
  /** CDN URL for assets (default: https://cdn.laptrinhai.id.vn/deepfilternet3) */
  cdnUrl?: string;
  /** Attenuation limit in dB - higher = more aggressive noise reduction (default: 100) */
  attenuationLimit?: number;
  /** Post filter beta - 0 disables post filter, range 0-0.05 (default: 0.02) */
  postFilterBeta?: number;
}

export interface ProcessingStats {
  inputSamples: number;
  outputSamples: number;
  framesProcessed: number;
  processingTimeMs: number;
}

export class StandaloneDeepFilter {
  private config: Required<DeepFilterConfig>;
  private modelBytes: ArrayBuffer | null = null;
  private dfHandle: number | null = null;
  private frameLength: number = 0;
  private isInitialized: boolean = false;
  
  // Ring buffers for frame-based processing
  private inputBuffer: Float32Array | null = null;
  private outputBuffer: Float32Array | null = null;
  private inputWritePos: number = 0;
  private outputWritePos: number = 0;
  private outputReadPos: number = 0;
  private bufferSize: number = 0;

  // ============================================================================
  // STREAMING MODE: Maintains state across calls for better audio quality
  // ============================================================================
  private isStreaming: boolean = false;
  private streamingAccumulator: Float32Array = new Float32Array(0);
  private streamingFramesProcessed: number = 0;
  private streamingTotalProcessingMs: number = 0;

  constructor(config: DeepFilterConfig = {}) {
    this.config = {
      cdnUrl: config.cdnUrl ?? 'https://cdn.laptrinhai.id.vn/deepfilternet3',
      attenuationLimit: config.attenuationLimit ?? 50,
      postFilterBeta: config.postFilterBeta ?? 0.02,
    };
  }

  /**
   * Initialize the denoiser - loads WASM and model from CDN
   * Call this once at app startup to avoid delay during push-to-talk
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.warn('StandaloneDeepFilter already initialized');
      return;
    }

    console.log('[DeepFilter] Loading WASM and model...');
    const startTime = performance.now();

    // Fetch WASM and model in parallel
    const wasmUrl = `${this.config.cdnUrl}/pkg/df_bg.wasm`;
    const modelUrl = `${this.config.cdnUrl}/models/DeepFilterNet3_onnx.tar.gz`;

    const [wasmBytes, modelBytes] = await Promise.all([
      this.fetchAsset(wasmUrl),
      this.fetchAsset(modelUrl),
    ]);

    this.modelBytes = modelBytes;

    // Use ASYNC initialization to avoid main thread blocking for large WASM
    console.log('[DeepFilter] Compiling WASM module asynchronously...');
    await wasm_bindgen.initAsync(wasmBytes);

    // Create the DeepFilter model
    console.log('[DeepFilter] Creating DeepFilter model...');
    const modelBytesArray = new Uint8Array(this.modelBytes);
    this.dfHandle = wasm_bindgen.df_create(modelBytesArray, this.config.attenuationLimit);
    
    // Get frame length from WASM
    this.frameLength = wasm_bindgen.df_get_frame_length(this.dfHandle);
    
    // Set post filter beta
    wasm_bindgen.df_set_post_filter_beta(this.dfHandle, this.config.postFilterBeta);

    // Initialize ring buffers (4x frame length for safety)
    this.bufferSize = this.frameLength * 8;
    this.inputBuffer = new Float32Array(this.bufferSize);
    this.outputBuffer = new Float32Array(this.bufferSize);
    this.inputWritePos = 0;
    this.outputWritePos = 0;
    this.outputReadPos = 0;

    this.isInitialized = true;

    const loadTime = performance.now() - startTime;
    console.log(`[DeepFilter] Initialized in ${loadTime.toFixed(0)}ms (frame length: ${this.frameLength} samples)`);
  }

  // ============================================================================
  // STREAMING MODE API - Use this for real-time audio processing
  // ============================================================================

  /**
   * Start streaming mode - call this at the beginning of a recording session
   * This resets the accumulator but maintains the neural network state
   */
  startStreaming(): void {
    if (!this.isInitialized) {
      throw new Error('DeepFilter not initialized. Call initialize() first.');
    }
    
    this.isStreaming = true;
    this.streamingAccumulator = new Float32Array(0);
    this.streamingFramesProcessed = 0;
    this.streamingTotalProcessingMs = 0;
    
    console.log('[DeepFilter] Streaming mode started');
  }

  /**
   * Process an audio chunk in streaming mode
   * Accumulates samples and processes complete frames, maintaining state between calls
   * 
   * @param input - Float32Array of audio samples (mono, 48kHz expected)
   * @returns Denoised audio as Float32Array (may be different length than input due to frame accumulation)
   */
  processStreaming(input: Float32Array): Float32Array {
    if (!this.isInitialized || !this.dfHandle) {
      throw new Error('DeepFilter not initialized. Call initialize() first.');
    }
    
    if (!this.isStreaming) {
      // Auto-start streaming if not already started
      this.startStreaming();
    }

    const startTime = performance.now();
    
    // Append new samples to the accumulator
    const newAccumulator = new Float32Array(this.streamingAccumulator.length + input.length);
    newAccumulator.set(this.streamingAccumulator);
    newAccumulator.set(input, this.streamingAccumulator.length);
    this.streamingAccumulator = newAccumulator;

    // Process complete frames
    const outputChunks: Float32Array[] = [];
    
    while (this.streamingAccumulator.length >= this.frameLength) {
      // Extract one frame
      const frame = this.streamingAccumulator.slice(0, this.frameLength);
      
      // Keep the remainder
      this.streamingAccumulator = this.streamingAccumulator.slice(this.frameLength);
      
      // Process through WASM - this maintains internal neural network state
      const processed = wasm_bindgen.df_process_frame(this.dfHandle, frame);
      outputChunks.push(new Float32Array(processed));
      this.streamingFramesProcessed++;
    }

    // Combine output chunks
    if (outputChunks.length === 0) {
      // Not enough samples for a complete frame yet
      return new Float32Array(0);
    }

    const totalLength = outputChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const output = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of outputChunks) {
      output.set(chunk, offset);
      offset += chunk.length;
    }

    const processingTime = performance.now() - startTime;
    this.streamingTotalProcessingMs += processingTime;

    // Log occasionally (every 100 frames)
    if (this.streamingFramesProcessed % 100 === 0) {
      const avgTime = this.streamingTotalProcessingMs / this.streamingFramesProcessed;
      console.log(`[DeepFilter] Streaming: ${this.streamingFramesProcessed} frames, avg ${avgTime.toFixed(2)}ms/frame`);
    }

    return output;
  }

  /**
   * Flush any remaining samples in the accumulator
   * Call this when recording ends to get the final audio
   * 
   * @returns Denoised audio of remaining samples (may be zero-padded)
   */
  flushStreaming(): Float32Array {
    if (!this.isInitialized || !this.dfHandle) {
      throw new Error('DeepFilter not initialized.');
    }

    if (this.streamingAccumulator.length === 0) {
      return new Float32Array(0);
    }

    // Pad remaining samples to frame length
    const paddedFrame = new Float32Array(this.frameLength);
    paddedFrame.set(this.streamingAccumulator);
    
    // Process the final padded frame
    const processed = wasm_bindgen.df_process_frame(this.dfHandle, paddedFrame);
    
    // Return only the valid samples (not the padding)
    return new Float32Array(processed).slice(0, this.streamingAccumulator.length);
  }

  /**
   * Stop streaming mode and get stats
   * Call this at the end of a recording session
   */
  stopStreaming(): { framesProcessed: number; totalProcessingMs: number; avgMsPerFrame: number } {
    const stats = {
      framesProcessed: this.streamingFramesProcessed,
      totalProcessingMs: this.streamingTotalProcessingMs,
      avgMsPerFrame: this.streamingFramesProcessed > 0 
        ? this.streamingTotalProcessingMs / this.streamingFramesProcessed 
        : 0,
    };
    
    console.log(`[DeepFilter] Streaming stopped: ${stats.framesProcessed} frames, ${stats.totalProcessingMs.toFixed(0)}ms total, ${stats.avgMsPerFrame.toFixed(2)}ms/frame avg`);
    
    this.isStreaming = false;
    this.streamingAccumulator = new Float32Array(0);
    this.streamingFramesProcessed = 0;
    this.streamingTotalProcessingMs = 0;
    
    return stats;
  }

  /**
   * Check if currently in streaming mode
   */
  isStreamingMode(): boolean {
    return this.isStreaming;
  }

  /**
   * Get the number of samples currently buffered in the streaming accumulator
   */
  getStreamingBufferSize(): number {
    return this.streamingAccumulator.length;
  }

  // ============================================================================
  // BATCH MODE API - Use this for processing complete audio files
  // ============================================================================

  /**
   * Process a complete audio buffer through the denoiser (batch mode)
   * NOTE: This resets internal state - for real-time streaming use processStreaming() instead
   * 
   * @param input - Float32Array of audio samples (mono, 48kHz expected)
   * @returns Denoised audio as Float32Array
   */
  processAudio(input: Float32Array): Float32Array {
    if (!this.isInitialized || !this.dfHandle || !this.inputBuffer || !this.outputBuffer) {
      throw new Error('DeepFilter not initialized. Call initialize() first.');
    }

    const startTime = performance.now();
    
    // Reset buffers for fresh processing
    this.inputWritePos = 0;
    this.outputWritePos = 0;
    this.outputReadPos = 0;
    this.inputBuffer.fill(0);
    this.outputBuffer.fill(0);

    // Process in frame-sized chunks
    const frameLength = this.frameLength;
    const tempFrame = new Float32Array(frameLength);
    const outputChunks: Float32Array[] = [];
    let framesProcessed = 0;

    // Add input samples and process frames
    for (let i = 0; i < input.length; i += frameLength) {
      const remaining = input.length - i;
      const chunkSize = Math.min(frameLength, remaining);
      
      // Copy to temp frame (pad with zeros if needed)
      tempFrame.fill(0);
      tempFrame.set(input.subarray(i, i + chunkSize));

      // Process through WASM
      const processed = wasm_bindgen.df_process_frame(this.dfHandle, tempFrame);
      outputChunks.push(new Float32Array(processed));
      framesProcessed++;
    }

    // Combine output chunks
    const totalLength = outputChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const output = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of outputChunks) {
      output.set(chunk, offset);
      offset += chunk.length;
    }

    // Trim to original length (remove padding)
    const result = output.subarray(0, input.length);

    const processingTime = performance.now() - startTime;
    console.log(`[DeepFilter] Processed ${input.length} samples in ${processingTime.toFixed(1)}ms (${framesProcessed} frames)`);

    return new Float32Array(result);
  }

  /**
   * Process audio with detailed statistics
   */
  processAudioWithStats(input: Float32Array): { audio: Float32Array; stats: ProcessingStats } {
    const startTime = performance.now();
    const audio = this.processAudio(input);
    const processingTimeMs = performance.now() - startTime;

    return {
      audio,
      stats: {
        inputSamples: input.length,
        outputSamples: audio.length,
        framesProcessed: Math.ceil(input.length / this.frameLength),
        processingTimeMs,
      },
    };
  }

  /**
   * Set the attenuation limit (noise reduction strength)
   * @param limitDb - Attenuation limit in dB (0-100, higher = more aggressive)
   */
  setAttenuationLimit(limitDb: number): void {
    if (!this.dfHandle) {
      throw new Error('DeepFilter not initialized');
    }
    const clamped = Math.max(0, Math.min(100, Math.floor(limitDb)));
    wasm_bindgen.df_set_atten_lim(this.dfHandle, clamped);
    console.log(`[DeepFilter] Attenuation limit set to ${clamped}dB`);
  }

  /**
   * Set the post filter beta
   * @param beta - Post filter attenuation (0-0.05, 0 disables)
   */
  setPostFilterBeta(beta: number): void {
    if (!this.dfHandle) {
      throw new Error('DeepFilter not initialized');
    }
    wasm_bindgen.df_set_post_filter_beta(this.dfHandle, beta);
    console.log(`[DeepFilter] Post filter beta set to ${beta}`);
  }

  /**
   * Get the frame length used by the model
   */
  getFrameLength(): number {
    return this.frameLength;
  }

  /**
   * Check if the denoiser is ready
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Destroy and cleanup resources
   */
  destroy(): void {
    if (this.isStreaming) {
      this.stopStreaming();
    }
    this.dfHandle = null;
    this.modelBytes = null;
    this.inputBuffer = null;
    this.outputBuffer = null;
    this.streamingAccumulator = new Float32Array(0);
    this.isInitialized = false;
    console.log('[DeepFilter] Destroyed');
  }

  private async fetchAsset(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return response.arrayBuffer();
  }
}

export default StandaloneDeepFilter;
