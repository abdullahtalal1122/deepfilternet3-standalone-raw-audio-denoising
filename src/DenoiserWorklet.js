/**
 * AudioWorklet processor for DeepFilter streaming
 * This runs in a separate audio thread and collects raw audio chunks
 */
class DenoiserWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.isCollecting = true;
    
    // Listen for messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') {
        this.isCollecting = false;
      } else if (event.data.type === 'start') {
        this.isCollecting = true;
        this.buffer = [];
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const inputChannel = input[0];
    
    if (this.isCollecting) {
      // Send raw audio chunk to main thread for processing
      // We copy the data since AudioWorklet buffers are recycled
      this.port.postMessage({
        type: 'audioChunk',
        samples: new Float32Array(inputChannel)
      });
    }

    // Pass through input to output (we'll handle denoised playback separately)
    const output = outputs[0];
    if (output && output[0]) {
      output[0].set(inputChannel);
    }

    return true;
  }
}

registerProcessor('denoiser-worklet', DenoiserWorklet);
