import type { Analysis, AnalysisStats, AgentResult, SymbolicReasoning, Visualizations, TimelineEvent, FrameAnomaly, ReasoningRule, OverriddenPrediction } from '@/types/analysis';

export const generateMockAgentResults = (isDeepfake: boolean): AgentResult[] => {
  const base = isDeepfake ? 30 : 85;
  const variance = 15;
  
  return [
    {
      agent: 'vision',
      score: Math.min(100, Math.max(0, base + (Math.random() * variance * 2 - variance))),
      confidence: 75 + Math.random() * 20,
      status: 'completed',
      details: isDeepfake 
        ? 'Detected facial inconsistencies in temporal frames. Blending artifacts around jaw line.'
        : 'No significant visual anomalies detected. Consistent lighting and shadow patterns.',
      anomalies: isDeepfake 
        ? ['Face swap artifacts', 'Temporal inconsistency', 'Unnatural eye movement']
        : []
    },
    {
      agent: 'audio',
      score: Math.min(100, Math.max(0, base + (Math.random() * variance * 2 - variance))),
      confidence: 70 + Math.random() * 25,
      status: 'completed',
      details: isDeepfake
        ? 'Voice synthesis patterns detected. Unnatural prosody in speech segments.'
        : 'Natural voice characteristics. Consistent acoustic fingerprint.',
      anomalies: isDeepfake
        ? ['Synthetic voice markers', 'Lip-sync delay', 'Frequency artifacts']
        : []
    },
    {
      agent: 'bio',
      score: Math.min(100, Math.max(0, base + (Math.random() * variance * 2 - variance))),
      confidence: 80 + Math.random() * 15,
      status: 'completed',
      details: isDeepfake
        ? 'Physiological signals inconsistent. Missing micro-expressions in key frames.'
        : 'Natural physiological patterns. Consistent pulse and micro-expression data.',
      anomalies: isDeepfake
        ? ['Irregular blink patterns', 'Missing micro-expressions', 'Unnatural skin texture']
        : []
    }
  ];
};

export const generateMockSymbolicReasoning = (agentResults: AgentResult[], isDeepfake: boolean): SymbolicReasoning => {
  const rules: ReasoningRule[] = [
    {
      id: 'R1',
      description: 'Temporal consistency check',
      condition: 'IF frame_t != frame_t+1 > threshold THEN flag_inconsistency',
      result: isDeepfake ? 'fail' : 'pass',
      weight: 0.25
    },
    {
      id: 'R2',
      description: 'Audio-visual synchronization',
      condition: 'IF audio_sync_offset > 50ms THEN flag_desync',
      result: isDeepfake ? 'warning' : 'pass',
      weight: 0.20
    },
    {
      id: 'R3',
      description: 'Physiological plausibility',
      condition: 'IF blink_rate < 3/min OR > 25/min THEN flag_implausible',
      result: isDeepfake ? 'fail' : 'pass',
      weight: 0.25
    },
    {
      id: 'R4',
      description: 'Facial geometry validation',
      condition: 'IF facial_landmarks_variance > threshold THEN flag_manipulation',
      result: isDeepfake ? 'fail' : 'pass',
      weight: 0.15
    },
    {
      id: 'R5',
      description: 'Voice naturalness verification',
      condition: 'IF formant_ratio NOT IN natural_range THEN flag_synthetic',
      result: isDeepfake ? 'warning' : 'pass',
      weight: 0.15
    }
  ];

  const overriddenPredictions: OverriddenPrediction[] = isDeepfake
    ? [
        {
          agent: 'audio',
          originalScore: agentResults.find(a => a.agent === 'audio')?.score || 50,
          overriddenTo: 25,
          reason: 'Logical impossibility: detected voice without corresponding lip movement in frames 1240-1380'
        }
      ]
    : [];

  return {
    conclusion: isDeepfake
      ? 'Neuro-symbolic analysis confirms high probability of synthetic manipulation. Multiple logical rules violated.'
      : 'Analysis passed all symbolic validation rules. Content appears authentic.',
    rules,
    overriddenPredictions,
    logicScore: isDeepfake ? 25 + Math.random() * 20 : 85 + Math.random() * 10
  };
};

export const generateMockVisualizations = (duration: number, isDeepfake: boolean): Visualizations => {
  const frameCount = Math.floor(duration * 30); // 30 fps
  
  const frameAnomalies: FrameAnomaly[] = isDeepfake
    ? Array.from({ length: Math.floor(Math.random() * 8) + 3 }, () => ({
        frameNumber: Math.floor(Math.random() * frameCount),
        timestamp: Math.random() * duration,
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as 'low' | 'medium' | 'high',
        type: ['Face swap artifact', 'Blending error', 'Temporal glitch', 'Lighting inconsistency'][Math.floor(Math.random() * 4)],
        x: Math.random() * 0.4 + 0.3,
        y: Math.random() * 0.3 + 0.2,
        width: Math.random() * 0.2 + 0.1,
        height: Math.random() * 0.2 + 0.1
      }))
    : [];

  const audioSpectrum = Array.from({ length: 64 }, () => 
    isDeepfake 
      ? Math.random() * 0.8 + 0.2 
      : Math.random() * 0.4 + 0.1
  );

  const timeline: TimelineEvent[] = [];
  const eventCount = isDeepfake ? Math.floor(Math.random() * 12) + 5 : Math.floor(Math.random() * 3);
  
  for (let i = 0; i < eventCount; i++) {
    timeline.push({
      timestamp: Math.random() * duration,
      type: ['visual', 'audio', 'bio'][Math.floor(Math.random() * 3)] as 'visual' | 'audio' | 'bio',
      severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as 'low' | 'medium' | 'high',
      description: [
        'Facial landmark deviation detected',
        'Audio frequency anomaly',
        'Blink pattern irregularity',
        'Lip sync desynchronization',
        'Skin texture inconsistency'
      ][Math.floor(Math.random() * 5)]
    });
  }
  
  timeline.sort((a, b) => a.timestamp - b.timestamp);

  // Generate heatmap data (10x10 grid)
  const heatmapData = Array.from({ length: 10 }, () =>
    Array.from({ length: 10 }, () => 
      isDeepfake ? Math.random() : Math.random() * 0.3
    )
  );

  return {
    frameAnomalies,
    audioSpectrum,
    timeline,
    heatmapData
  };
};

export const generateMockAnalysis = (id: string, isDeepfake: boolean = Math.random() > 0.5): Analysis => {
  const duration = 15 + Math.random() * 120;
  const agentResults = generateMockAgentResults(isDeepfake);
  const avgScore = agentResults.reduce((sum, r) => sum + r.score, 0) / agentResults.length;
  
  const verdict = avgScore > 70 ? 'authentic' : avgScore < 40 ? 'deepfake' : 'inconclusive';
  
  return {
    id,
    filename: `video_${id}.mp4`,
    fileSize: Math.floor(Math.random() * 100 + 10) * 1024 * 1024,
    duration,
    createdAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
    completedAt: new Date(),
    status: 'completed',
    verdict: verdict as any,
    overallScore: avgScore,
    agentResults,
    symbolicReasoning: generateMockSymbolicReasoning(agentResults, isDeepfake),
    visualizations: generateMockVisualizations(duration, isDeepfake),
    thumbnailUrl: `https://picsum.photos/seed/${id}/320/180`,
    processingTime: Math.floor(Math.random() * 30000 + 5000)
  };
};

export const generateMockAnalyses = (count: number): Analysis[] => {
  return Array.from({ length: count }, (_, i) => 
    generateMockAnalysis(`analysis-${i + 1}`, i % 3 !== 0)
  );
};

export const generateMockStats = (): AnalysisStats => {
  const total = Math.floor(Math.random() * 500 + 100);
  const deepfakes = Math.floor(total * (0.3 + Math.random() * 0.2));
  const inconclusive = Math.floor(total * (0.05 + Math.random() * 0.1));
  
  return {
    totalAnalyses: total,
    deepfakesDetected: deepfakes,
    authenticVideos: total - deepfakes - inconclusive,
    inconclusiveResults: inconclusive,
    averageProcessingTime: Math.floor(Math.random() * 20000 + 8000),
    todayAnalyses: Math.floor(Math.random() * 20 + 5),
    weeklyTrend: Math.floor(Math.random() * 40 - 10)
  };
};

export const mockAnalyses = generateMockAnalyses(25);
export const mockStats = generateMockStats();
