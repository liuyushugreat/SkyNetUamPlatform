import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

// Synthetic data generation to mimic the "Torus" manifold
// In a real app, this would load from a JSON file exported by the Python script
const generateTorusData = (count = 2000, stability = 1.0) => {
    const points = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
        // Torus parameterization:
        // x = (R + r*cos(theta)) * cos(phi)
        // y = (R + r*cos(theta)) * sin(phi)
        // z = r * sin(theta)
        
        const u = Math.random() * Math.PI * 2; // theta
        const v = Math.random() * Math.PI * 2; // phi
        
        const R = 3; // Major radius
        const r = 1; // Minor radius
        
        // Add noise based on stability (lower stability = more noise/tears)
        const noise = (1 - stability) * 0.5;
        const uNoise = (Math.random() - 0.5) * noise;
        const vNoise = (Math.random() - 0.5) * noise;
        
        const x = (R + r * Math.cos(u + uNoise)) * Math.cos(v + vNoise);
        const y = (R + r * Math.cos(u + uNoise)) * Math.sin(v + vNoise);
        const z = r * Math.sin(u + uNoise);
        
        points[i * 3] = x;
        points[i * 3 + 1] = y;
        points[i * 3 + 2] = z;
        
        // Color gradient based on angle (similar to Python visualization)
        const color = new THREE.Color().setHSL(v / (Math.PI * 2), 1.0, 0.5);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }
    
    return { points, colors };
};

const TorusPointCloud = ({ stability = 1.0 }) => {
    const meshRef = useRef<THREE.Points>(null);
    const { points, colors } = useMemo(() => generateTorusData(3000, stability), [stability]);
    
    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += 0.002;
            meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
        }
    });

    return (
        <points ref={meshRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={points.length / 3}
                    array={points}
                    itemSize={3}
                />
                <bufferAttribute
                    attach="attributes-color"
                    count={colors.length / 3}
                    array={colors}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial
                vertexColors
                size={0.08}
                sizeAttenuation
                transparent
                opacity={0.8}
            />
        </points>
    );
};

const MarketTopologyMonitor: React.FC = () => {
    // Mock metric for market stability (1.0 = perfect torus, <1.0 = arbitrage/noise)
    // This would eventually come from the Python backend Betti number analysis
    const stability = 1.0; 

    return (
        <div className="w-full h-64 bg-slate-900 rounded-xl overflow-hidden relative border border-slate-700 shadow-inner">
            <div className="absolute top-3 left-3 z-10">
                <h3 className="text-white text-sm font-bold flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                    Market Topology Monitor
                </h3>
                <p className="text-[10px] text-slate-400 font-mono mt-1">
                    Betti Numbers: $\beta_1=2, \beta_2=1$ (Stable)
                </p>
            </div>

            {/* Scientific Annotation Tooltip */}
            <div className="absolute bottom-3 right-3 z-10 max-w-[200px] text-right">
                <div className="group relative inline-block">
                    <span className="text-[10px] text-blue-400 cursor-help border-b border-dashed border-blue-400">
                        Methodology: Geometric RWA
                    </span>
                    <div className="invisible group-hover:visible absolute bottom-full right-0 mb-2 w-48 bg-slate-800 text-slate-200 text-[10px] p-2 rounded border border-slate-600 shadow-xl z-20">
                        "Market latent geometry based on modular representations." [Cite: 18]
                        <br/>
                        <span className="italic text-slate-400 mt-1 block">
                            Verify fairness by checking if pricing manifold forms a perfect torus.
                        </span>
                    </div>
                </div>
            </div>

            <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
                <color attach="background" args={['#0f172a']} />
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                
                <TorusPointCloud stability={stability} />
                
                <OrbitControls enableZoom={false} autoRotate={false} />
            </Canvas>
        </div>
    );
};

export default MarketTopologyMonitor;

