import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js';

interface VectorData {
  id: number;
  text: string;  // Changed from optional to required since we know it exists
  vector: number[];
  metadata: {
    source: string;
    category: string;
    [key: string]: any;  // Allow for additional metadata fields
  };
}

const VectorVisualization: React.FC = () => {
  const plotRef = useRef<HTMLDivElement>(null);
  const [vectors, setVectors] = useState<VectorData[]>([]);
  const [selectedVector, setSelectedVector] = useState<VectorData | null>(null);
  const plotInitialized = useRef(false);

  useEffect(() => {
    const fetchVectors = async () => {
      try {
        const response = await axios.get('http://localhost:8000/vectors');
        console.log('Raw response data:', response.data);
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
          zaxis: { title: 'PC3' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 },
            center: { x: 0, y: 0, z: 0 }
          }
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        width: window.innerWidth,
        height: window.innerHeight,
        uirevision: 'constant'
      };

      const plotElement = plotRef.current;
      if (plotElement) {
        if (!plotInitialized.current) {
          Plotly.newPlot(plotElement, data, layout, {
            responsive: true,
            displayModeBar: true
          });
          plotInitialized.current = true;

          // Add click handler
          (plotElement as any).on('plotly_click', (eventData: any) => {
            if (eventData && eventData.points && eventData.points[0]) {
              const point = eventData.points[0];
              const pointIndex = point.pointNumber;
              setSelectedVector(vectors[pointIndex]);
            }
          });
        } else {
          // Update only the data
          Plotly.newPlot(plotElement, data, layout, {
            responsive: true,
            displayModeBar: true
          });
        }
      }

    } catch (error) {
      console.error('Error during visualization:', error);
    }

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
        plotInitialized.current = false;
      }
    };
  }, [vectors]);

  // Handle container width changes separately
  useEffect(() => {
    if (plotRef.current && plotInitialized.current) {
      const width = selectedVector ? window.innerWidth * 0.7 : window.innerWidth;
      const layout = {
        width,
        uirevision: 'constant'
      };
      (plotRef.current as any).style.width = `${width}px`;
    }
  }, [selectedVector]);

  return (
    <div style={{ 
      display: 'flex', 
      width: '100%', 
      height: '100vh',
      backgroundColor: '#ffffff'
    }}>
      <div 
        ref={plotRef} 
        style={{ 
          width: selectedVector ? '70%' : '100%',
          height: '100vh',
          border: '1px solid #e0e0e0',
          borderRadius: '8px',
          overflow: 'hidden',
          transition: 'width 0.3s ease'
        }} 
      />
      {selectedVector && (
        <div style={{
          width: '30%',
          padding: '20px',
          borderLeft: '1px solid #e0e0e0',
          backgroundColor: '#fafafa',
          overflowY: 'auto',
          color: '#000000'  // Set default text color to black
        }}>
          <h2 style={{ color: '#000000', marginBottom: '20px' }}>Vector Details</h2>
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ color: '#000000' }}>ID: {selectedVector.id}</h3>
          </div>
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ color: '#000000' }}>Text</h3>
            <div style={{ 
              backgroundColor: '#fff', 
              padding: '10px', 
              borderRadius: '4px',
              maxHeight: '200px',
              overflowY: 'auto',
              border: '1px solid #e0e0e0'
            }}>
              <pre style={{ 
                margin: 0, 
                whiteSpace: 'pre-wrap',
                color: '#000000',
                fontFamily: 'inherit'
              }}>
                {selectedVector.text}
              </pre>
            </div>
          </div>
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ color: '#000000' }}>Metadata</h3>
            <div style={{ 
              backgroundColor: '#fff', 
              padding: '10px', 
              borderRadius: '4px',
              border: '1px solid #e0e0e0'
            }}>
              <div style={{ marginBottom: '8px', color: '#000000' }}>
                <strong>Source:</strong> {selectedVector.metadata.source}
              </div>
              <div style={{ marginBottom: '8px', color: '#000000' }}>
                <strong>Category:</strong> {selectedVector.metadata.category}
              </div>
              {Object.entries(selectedVector.metadata)
                .filter(([key]) => !['source', 'category'].includes(key))
                .map(([key, value]) => (
                  <div key={key} style={{ marginBottom: '8px', color: '#000000' }}>
                    <strong>{key}:</strong> {JSON.stringify(value)}
                  </div>
                ))
              }
            </div>
          </div>
          <div>
            <h3 style={{ color: '#000000' }}>Vector Values</h3>
            <div style={{ 
              backgroundColor: '#fff', 
              padding: '10px', 
              borderRadius: '4px',
              maxHeight: '200px',
              overflowY: 'auto',
              border: '1px solid #e0e0e0'
            }}>
              <pre style={{ 
                margin: 0,
                color: '#000000',
                fontFamily: 'inherit'
              }}>
                {JSON.stringify(selectedVector.vector, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VectorVisualization; 