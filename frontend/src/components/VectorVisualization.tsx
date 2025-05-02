import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js';

interface VectorData {
  id: number;
  vector: number[];
  metadata: Record<string, any>;
}

const VectorVisualization: React.FC = () => {
  const plotRef = useRef<HTMLDivElement>(null);
  const [vectors, setVectors] = useState<VectorData[]>([]);

  useEffect(() => {
    const fetchVectors = async () => {
      try {
        const response = await axios.get('http://localhost:8000/vectors');
        console.log('Raw response data:', response.data);
        // Assuming the response has a 'vectors' field containing the array
        const vectorsData = response.data.vectors || response.data;
        console.log('Processed vectors data:', vectorsData);
        setVectors(Array.isArray(vectorsData) ? vectorsData : []);
      } catch (error) {
        console.error('Error fetching vectors:', error);
      }
    };

    fetchVectors();
  }, []);

  useEffect(() => {
    if (!plotRef.current || vectors.length === 0) {
      console.log('No vectors to visualize:', vectors);
      return;
    }

    console.log('Preparing to visualize vectors:', vectors);

    // Simple PCA implementation
    const performPCA = (vectors: number[][], dimensions: number = 3) => {
      const n = vectors.length;
      if (n === 0) return [];

      console.log('Input vectors for PCA:', vectors);
      console.log('Vector dimensions:', vectors[0].length);

      // Center the data
      const mean = vectors[0].map((_, i) => 
        vectors.reduce((sum, v) => sum + v[i], 0) / n
      );
      
      const centered = vectors.map(v => 
        v.map((val, i) => val - mean[i])
      );

      // Compute covariance matrix
      const cov = centered[0].map((_, i) =>
        centered[0].map((_, j) =>
          centered.reduce((sum, v) => sum + v[i] * v[j], 0) / (n - 1)
        )
      );

      // Power iteration for eigenvectors
      const getEigenvectors = (matrix: number[][], numComponents: number) => {
        const n = matrix.length;
        const eigenvectors = Array(numComponents).fill(null).map(() => 
          Array(n).fill(0).map(() => Math.random())
        );

        for (let i = 0; i < 10; i++) {
          eigenvectors.forEach((eigenvector, idx) => {
            const newVector = matrix.map(row =>
              row.reduce((sum, val, j) => sum + val * eigenvector[j], 0)
            );
            
            const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
            eigenvectors[idx] = newVector.map(val => val / norm);
          });
        }

        return eigenvectors;
      };

      const eigenvectors = getEigenvectors(cov, dimensions);
      
      // Project data onto principal components
      return centered.map(v => 
        eigenvectors.map(eigenvector =>
          v.reduce((sum, val, i) => sum + val * eigenvector[i], 0)
        )
      );
    };

    try {
      const vectorArrays = vectors.map(v => v.vector);
      console.log('Vector arrays for PCA:', vectorArrays);
      const pcaVectors = performPCA(vectorArrays);
      console.log('PCA result:', pcaVectors);

      const data: Data[] = [{
        type: 'scatter3d',
        mode: 'markers',
        x: pcaVectors.map(v => v[0]),
        y: pcaVectors.map(v => v[1]),
        z: pcaVectors.map(v => v[2]),
        marker: {
          size: 8,
          color: vectors.map((_, i) => i),
          colorscale: 'Viridis',
          opacity: 0.8
        },
        text: vectors.map(v => `ID: ${v.id}`),
        hoverinfo: 'text'
      }];

      const layout: Partial<Layout> = {
        title: 'Vector Visualization (PCA)',
        scene: {
          aspectmode: 'cube',
          xaxis: { title: 'PC1' },
          yaxis: { title: 'PC2' },
          zaxis: { title: 'PC3' }
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        width: window.innerWidth,
        height: window.innerHeight
      };

      Plotly.newPlot(plotRef.current, data, layout, {
        responsive: true,
        displayModeBar: true
      });
    } catch (error) {
      console.error('Error during visualization:', error);
    }

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [vectors]);

  return (
    <div 
      ref={plotRef} 
      style={{ 
        width: '100%', 
        height: '100vh',
        border: '1px solid #e0e0e0',
        borderRadius: '8px',
        overflow: 'hidden',
        backgroundColor: '#ffffff'
      }} 
    />
  );
};

export default VectorVisualization; 