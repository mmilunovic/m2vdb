import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js';
import { UMAP } from 'umap-js';
import VectorDetailsPanel from './VectorDetailsPanel';
import SearchResultsPanel from './SearchResultsPanel';

interface VectorData {
  id: number;
  text: string;
  vector: number[];
  metadata: {
    id: string;
    name: string;
    category: string;
    subcategory: string;
    brand: string;
    price_usd: number | null;
    source: string;
  };
}

interface NeighborData {
  id: number;
  distance: number;
  text: string;
  metadata: {
    id: string;
    name: string;
    category: string;
    subcategory: string;
    brand: string;
    price_usd: number | null;
    source: string;
  };
}

interface SearchResult {
  id: number;
  text: string;
  similarity_score: number;
  metadata: {
    id: string;
    name: string;
    category: string;
    subcategory: string;
    brand: string;
    price_usd: number | null;
    source: string;
  };
}

interface VectorVisualizationProps {
  vectors: VectorData[];
  setVectors: React.Dispatch<React.SetStateAction<VectorData[]>>;
  selectedVector: VectorData | null;
  setSelectedVector: React.Dispatch<React.SetStateAction<VectorData | null>>;
  searchResults: SearchResult[];
  setSearchResults: React.Dispatch<React.SetStateAction<SearchResult[]>>;
  searchQuery: string;
}

const VectorVisualization: React.FC<VectorVisualizationProps> = ({
  vectors,
  setVectors,
  selectedVector,
  setSelectedVector,
  searchResults,
  setSearchResults,
  searchQuery,
}) => {
  type VizMethod = 'pca' | 'umap' | 'tsne';
  const [vizMethod, setVizMethod] = useState<VizMethod>('pca');
  const plotRef = useRef<HTMLDivElement>(null);
  const plotInitialized = useRef(false);
  const searchResultsRef = useRef<SearchResult[]>(searchResults);

  useEffect(() => {
    const fetchVectors = async () => {
      try {
        const response = await axios.get<{ vectors: VectorData[] }>('http://localhost:8000/vectors');
        const vectorsData = response.data.vectors || (response.data as unknown as VectorData[]);
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
    console.log('Current search results:', searchResults);

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

    // UMAP dimensionality reduction
    const performUMAP = (vectors: number[][], dimensions: number = 3) => {
      if (vectors.length === 0) return [];
      const umap = new UMAP({ nComponents: dimensions });
      // umap-js expects number[][] where rows are items.
      const embedding: number[][] = umap.fit(vectors);
      return embedding;
    };

    try {
      const vectorArrays = vectors.map(v => v.vector);
      console.log(`Vector arrays for ${vizMethod.toUpperCase()}:`, vectorArrays);
      let embeddedVectors: number[][] = [];
      if (vizMethod === 'umap') {
        embeddedVectors = performUMAP(vectorArrays);
      } else {
        embeddedVectors = performPCA(vectorArrays);
      }
      console.log(`${vizMethod.toUpperCase()} result:`, embeddedVectors);

      // Split vectors into returned and non-returned
      const returnedIndices = searchResults.map(result => 
        vectors.findIndex(v => v.id === result.id)
      ).filter(idx => idx !== -1);

      console.log('Current search results:', searchResults);
      console.log('Returned indices:', returnedIndices);

      const data: Data[] = [
        // Non-returned vectors (transparent)
        {
          type: 'scatter3d',
          mode: 'markers',
          x: embeddedVectors.map((v, i) => !returnedIndices.includes(i) ? v[0] : null).filter(x => x !== null),
          y: embeddedVectors.map((v, i) => !returnedIndices.includes(i) ? v[1] : null).filter(y => y !== null),
          z: embeddedVectors.map((v, i) => !returnedIndices.includes(i) ? v[2] : null).filter(z => z !== null),
          marker: {
            size: 8,
            color: vectors.map((_, i) => !returnedIndices.includes(i) ? i : null).filter(c => c !== null),
            colorscale: 'Viridis',
            opacity: searchResults.length > 0 ? 0.1 : 0.8
          },
          text: vectors.map((v, i) => !returnedIndices.includes(i) ? `ID: ${v.id}` : null).filter(t => t !== null),
          hoverinfo: 'text',
          name: 'non-returned',
          customdata: vectors
            .map((_, i) => !returnedIndices.includes(i) ? i : null)
            .filter((d): d is number => d !== null)
        },
        // Returned vectors (full opacity)
        {
          type: 'scatter3d',
          mode: 'markers',
          x: embeddedVectors.map((v, i) => returnedIndices.includes(i) ? v[0] : null).filter(x => x !== null),
          y: embeddedVectors.map((v, i) => returnedIndices.includes(i) ? v[1] : null).filter(y => y !== null),
          z: embeddedVectors.map((v, i) => returnedIndices.includes(i) ? v[2] : null).filter(z => z !== null),
          marker: {
            size: 8,
            color: vectors.map((_, i) => returnedIndices.includes(i) ? i : null).filter(c => c !== null),
            colorscale: 'Viridis',
            opacity: 0.8
          },
          text: vectors.map((v, i) => returnedIndices.includes(i) ? `ID: ${v.id}` : null).filter(t => t !== null),
          hoverinfo: 'text',
          name: 'returned',
          customdata: vectors
            .map((_, i) => returnedIndices.includes(i) ? i : null)
            .filter((d): d is number => d !== null)
        }
      ];

      console.log('Plot data:', data);

      const layout: Partial<Layout> = {
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
          console.log('Initializing plot...');
          // Initial plot creation
          Plotly.newPlot(plotElement, data, layout, {
            responsive: true,
            displayModeBar: true
          });
          plotInitialized.current = true;

          // Add click handler
          (plotElement as any).on('plotly_click', (eventData: any) => {
            console.log('Plot clicked:', eventData);
            if (eventData && eventData.points && eventData.points[0]) {
              const point = eventData.points[0];
              // Retrieve the original vector index that we stored in `customdata`
              const originalIndex: number | undefined = point.customdata;

              // Fallback to pointNumber (legacy behaviour) if for some reason customdata is undefined
              const vectorIndex = (typeof originalIndex === 'number') ? originalIndex : point.pointNumber;

              const clickedVector = vectors[vectorIndex];
              console.log('Clicked vector:', clickedVector);
              
              const currentResults = searchResultsRef.current;

              if (currentResults.length > 0) {
                const isReturnedVector = currentResults.some(result => result.id === clickedVector.id);
                console.log('Is returned vector:', isReturnedVector);
                if (!isReturnedVector) {
                  console.log('Clearing search results because clicked blurred vector');
                  // Only clear search results if clicking a blurred vector
                  setSearchResults([]);
                }
              }
              setSelectedVector(clickedVector);
            }
          });
        } else {
          console.log('Updating plot...');
          // Preserve the current camera position before updating
          const currentCamera = (plotElement as any)._fullLayout?.scene?.camera;

          if (currentCamera) {
            (layout.scene as any).camera = currentCamera;
          }

          // Use Plotly.react to update data/layout without reinitialising the scene
          (Plotly as any).react(plotElement, data, layout as any, {
            responsive: true,
            displayModeBar: true
          });
        }
      }

    } catch (error) {
      console.error('Error during visualization:', error);
    }

    // Do not purge here; we maintain the same plot instance across updates
    return undefined;
  }, [vectors, searchResults, vizMethod]);

  // Purge the plot only when the component is unmounted
  useEffect(() => {
    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
        plotInitialized.current = false;
      }
    };
  }, []);

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

  // Update the ref whenever searchResults changes
  useEffect(() => {
    searchResultsRef.current = searchResults;
  }, [searchResults]);

  return (
    <div style={{ 
      display: 'flex', 
      width: '100%', 
      height: '100vh',
      backgroundColor: '#ffffff',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        zIndex: 1000
      }}>
        <div style={{
          position: 'relative',
          display: 'inline-block'
        }}>
          <button
            style={{
              background: 'none',
              border: 'none',
              padding: '6px 14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              color: '#37352f',
              fontSize: '12px',
              fontWeight: 600,
              fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}
            onMouseOver={e => {
              const dropdown = e.currentTarget.nextElementSibling as HTMLElement;
              if (dropdown) dropdown.style.display = 'block';
            }}
            onMouseOut={e => {
              const dropdown = e.currentTarget.nextElementSibling as HTMLElement;
              if (dropdown) dropdown.style.display = 'none';
            }}
          >
            <span style={{ fontSize: '18px', transform: 'rotate(-90deg)' }}>▼</span>
            <span>
              {vizMethod === 'pca' && 'PCA Visualization (default/global)'}
              {vizMethod === 'umap' && 'UMAP Visualization (best of both worlds)'}
              {vizMethod === 'tsne' && 't-SNE Visualization (local-ish)'}
            </span>
          </button>
          <div
            style={{
              display: 'none',
              position: 'absolute',
              top: '100%',
              left: '0',
              backgroundColor: '#ffffff',
              border: '1px solid rgba(0, 0, 0, 0.1)',
              borderRadius: '4px',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              padding: '4px 0',
              minWidth: '270px',
              zIndex: 1001
            }}
            onMouseOver={e => e.currentTarget.style.display = 'block'}
            onMouseOut={e => e.currentTarget.style.display = 'none'}
          >
            {([
              { key: 'pca', label: 'PCA Visualization (default/global)' },
              { key: 'umap', label: 'UMAP Visualization (best of both worlds)' },
              { key: 'tsne', label: 't-SNE Visualization (local-ish)' },
            ] as { key: VizMethod; label: string }[]).map(method => (
              <div
                key={method.key}
                onClick={() => {
                  setVizMethod(method.key);
                }}
                style={{
                  padding: '10px 18px',
                  fontSize: '10px',
                  color: vizMethod === method.key ? '#37352f' : '#787774',
                  cursor: 'pointer',
                  backgroundColor: vizMethod === method.key ? '#f1f1ef' : '#ffffff',
                  fontWeight: 600
                }}
              >
                {method.label}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div 
        ref={plotRef} 
        style={{ 
          width: selectedVector || searchResults.length > 0 ? '70%' : '100%',
          height: '100vh',
          border: 'none',
          borderRadius: '0',
          overflow: 'hidden',
          transition: 'width 0.3s ease',
          boxShadow: 'none',
          position: 'fixed',
          left: 0,
          top: 0
        }} 
      />
      {selectedVector && (
        <VectorDetailsPanel
          selectedVector={selectedVector}
          setSelectedVector={setSelectedVector}
        />
      )}
      {!selectedVector && searchResults.length > 0 && (
        <SearchResultsPanel
          vectors={vectors}
          searchResults={searchResults}
          searchQuery={searchQuery}
          setSelectedVector={setSelectedVector}
          setSearchResults={setSearchResults}
        />
      )}
    </div>
  );
};

export default VectorVisualization; 